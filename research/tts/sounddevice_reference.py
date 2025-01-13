#!/usr/bin/env python3
"""
Creating an asyncio generator for blocks of audio data.

This example shows how a generator can be used to analyze audio input blocks.
In addition, it shows how a generator can be created that yields not only input
blocks but also output blocks where audio data can be written to.

You need Python 3.7 or newer to run this.
"""
import asyncio
import queue
import sys

import numpy as np
import sounddevice as sd
from queue import Queue
from datetime import datetime, timedelta
from faster_whisper import WhisperModel

import logging

logging.basicConfig(level=logging.INFO)


class LocalStreamingAda:
  def __init__(self,
               blocksize=1024,
               dtype='float32',
               channels=2,
               sttmodel="tiny.en"):
    self._stream = None
    self.output_queue = Queue()  # type: ignore[var-annotated]
    self.sttmodel = WhisperModel(sttmodel)
    self.logger = logging.getLogger(__name__)
    self.blocksize = blocksize
    self.dtype = dtype
    self.channels = channels
    self.CONVERSATION_START_DELAY = 0.005
    self.CONVERSATION_END_DELAY = 2.0
    self.CONVERSATION_DETECTION_LINE = 0.1

  async def stream_generator(self, *, pre_fill_blocks=10):
    """
    Generator that yields blocks of input/output data as NumPy arrays.

    The output blocks are uninitialized and have to be filled with
    appropriate audio signals.
    """
    assert self.blocksize != 0
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, outdata, frame_count, time_info, status):
      loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))
      if self.output_queue.not_empty:
        try:
          outdata[:] = self.output_queue.get_nowait()
        except asyncio.QueueEmpty:
          pass
          # outdata[:] = np.zeros((self.blocksize, self.channels), dtype=self.dtype)
      else:
        asyncio.sleep(0.1)

    # pre-fill output queue
    for _ in range(pre_fill_blocks):
      self.output_queue.put(np.zeros((self.blocksize, self.channels), dtype=self.dtype))

    stream = sd.Stream(blocksize=self.blocksize,
                       callback=callback,
                       dtype=self.dtype,
                       channels=self.channels)
    with stream:
      while True:
        indata, status = await q_in.get()
        outdata = np.empty((self.blocksize, self.channels), dtype=self.dtype)
        yield indata, outdata, status
        self.output_queue.put_nowait(outdata)

  async def stream_processor(self):
    """
    Create a connection between audio inputs and outputs.

    Asynchronously iterates over a stream generator and for each block
    simply copies the input data into the output block.
    """
    conversation_status = 'Not Started'
    last_started_set_time = datetime.now()
    async for indata, outdata, status in self.stream_generator():
      if status:
        self.logger.info(status)

      # The conversation may be starting if the input sound continues for the 
      # duration of the CONVERSATION_START_DELAY variable.
      if conversation_status == 'Not Started' and indata.max() >= self.CONVERSATION_DETECTION_LINE:
        self.logger.info(f"Conversation Started: {datetime.now() - last_started_set_time}")
        if (datetime.now() - last_started_set_time) > timedelta(seconds=self.CONVERSATION_START_DELAY):
          conversation_status = 'Started'
        last_started_set_time = datetime.now()
      
      # The conversation has already started and continues
      elif conversation_status == 'Started' and indata.max() >= self.CONVERSATION_DETECTION_LINE:
        self.logger.info(f"Recording Audio...")
        self.output_queue.put(indata)
        last_started_set_time = datetime.now()
      
      # The conversation has already started and may be ending if the conversation
      # stays silent for the duration of `CONVERSATION_END_DELAY`
      elif conversation_status == 'Started' and indata.max() < self.CONVERSATION_DETECTION_LINE:
        if (datetime.now() - last_started_set_time) > timedelta(seconds=self.CONVERSATION_END_DELAY):
          self.logger.info(f"Conversation Ended After {datetime.now() - last_started_set_time}")
          conversation_status = 'Not Started'
          # silence = np.zeros(int(0.25 * self.ttsmodel.generation_config.sample_rate))
          # silence = np.empty((1024, 1), dtype='float32')

          self.logger.info(f"Data Queue Size: {self.output_queue.qsize()}")
          text = ""
          data = []

          await asyncio.sleep(0.1)

          while self.output_queue.not_empty:
            # sd.play(b"".join(list(self.output_queue.get())))
            sd.play(self.output_queue.get())

          await asyncio.sleep(0.1)

          outdata[:] = np.zeros_like(indata)
          self.logger.info(f"Synthesized Text: {text}")
        else:
          self.logger.info("Recording Audio....")
          self.output_queue.put(np.zeros_like(indata) + indata + np.zeros_like(indata))
      else:
        self.logger.info("Listening for a Conversation")
        outdata[:] = np.zeros_like(indata)
      

  async def run(self):
    self.logger.info('Enough of that, activating wire ...')
    audio_task = asyncio.create_task(self.stream_processor())
    await asyncio.sleep(10)
    audio_task.cancel()
    try:
      await audio_task
    except asyncio.CancelledError:
      self.logger.info('wire was cancelled')


if __name__ == "__main__":
  try:
    ada = LocalStreamingAda(blocksize=2048)
    asyncio.run(ada.run())
  except KeyboardInterrupt:
    sys.exit('\nInterrupted by user')