import torch
import sys
from transformers import AutoProcessor, BarkModel
import logging
import queue
import asyncio
import numpy as np
import pyaudio
from IPython.display import Audio

logger = logging.getLogger(__name__)

# if torch.cuda.is_available(): device = "cuda" 
# elif sys.platform=='darwin' and torch.backends.mps.is_available(): device = "mps" 
# else: device = "cpu"

# processor = AutoProcessor.from_pretrained('suno/bark-small')
# ttsmodel = BarkModel.from_pretrained('suno/bark-small')
# ttsmodel.to(device)

SAMPLES_PER_SECOND=24000

class AsyncAudio:
    def __init__(self, 
                 chunk=1024,
                 format=pyaudio.paInt16,
                 channels=1,
                 rate=SAMPLES_PER_SECOND):
        self.chunk = chunk
        self.format = format
        self.channels = channels
        self.rate = rate
        self.p = pyaudio.PyAudio()
        self.input_stream = self.p.open(
            format=format,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
            stream_callback=self.is_callback,
        )
        self.output_stream = self.p.open(
            format=format,
            channels=channels,
            rate=rate,
            output=True,
            frames_per_buffer=chunk,
        )
        self.queue = queue.Queue()
        self.is_recording = False
        self.is_receiving = False
        self.exit_event = asyncio.Event()
        logger.info("AsyncMicrophone initialized")

    def is_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording and not self.is_receiving:
            self.queue.put(in_data)
        return (None, pyaudio.paContinue)

    def start_recording(self):
        self.is_recording = True
        logger.info("Started recording")

    def stop_recording(self):
        self.is_recording = False
        logger.info("Stopped recording")

    def start_receiving(self):
        self.is_receiving = True
        self.is_recording = False
        logger.info("Started receiving assistant response")

    def stop_receiving(self):
        self.is_receiving = False
        logger.info("Stopped receiving assistant response")

    def get_audio_data(self):
        data = b""
        while not self.queue.empty():
            data += self.queue.get()
        return data if data else None

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        logger.info("AsyncMicrophone closed")

    async def play(self):
      audio_data = self.get_audio_data()
      if audio_data:
          await self.play_audio(audio_data)

    async def play_audio(self, audio_data):
        # p = pyaudio.PyAudio()
        self.output_stream.write(audio_data)

        # Add a small delay of silence at the end to prevent popping, and weird cuts off sounds
        silence_duration = 0.4
        silence_frames = int(self.rate * silence_duration)
        silence = b"\x00" * ( silence_frames * self.channels * 2) # 2 bytes per sample for 16-bit audio
        self.output_stream.write(silence)

        # Add a small pause before closing the stream to make sure the audio is fully played
        await asyncio.sleep(0.5)

        self.output_stream.stop_stream()
        self.output_stream.close()
        # p.terminate()
        logger.debug("Audio playback completed")

    def stop_running(self):
        self.exit_event.is_set()
    
    async def run(self):
      logger.info("Starting recording...")
      self.start_receiving()
      self.start_recording()
      # while not self.exit_event.is_set():
      #   await asyncio.sleep(0.1)
      #   if not self.is_receiving and self.queue.get() is not None:
      #     await self.play_audio(self.get_audio_data())
      #   else:
      #     await asyncio.sleep(0.1)


async def main():
  input("Press Enter to start recording...")

  audio = AsyncAudio()
  sleep_timeout = 3
  try:
    await audio.run()
    logger.info(f"Sleeping for {sleep_timeout} seconds before stopping...")
    await asyncio.sleep(sleep_timeout)
    audio.stop_running()
    logger.info("Going to play audio")
    await audio.play()
  except KeyboardInterrupt:
    logger.info("Program terminated by user")
  except Exception as e:
    logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
  asyncio.run(main())