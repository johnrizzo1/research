{ pkgs, lib, config, inputs, ... }:

let 
  mypkgs = import inputs.nixpkgs-unstable {
    inherit (pkgs.stdenv) system;
    config.nixpkgs.allowUnfree = true;
    config.nixpkgs.cudaSupport = true;
  };
in {
  packages = with mypkgs; [ 
    git
    (python3.withPackages (pkgs-python: with pkgs-python; [
      arxiv
      ipykernel
      jupyter
      jupyterlab
      ipython
      python-dotenv
      pandas
      requests
      matplotlib
      langchain-openai
      langchain-ollama
      langchain-community
      pip
    ]))
  ];
  languages.python.enable = true;
  languages.python.package = mypkgs.python3;
  languages.python.uv.enable = true;
  languages.python.venv.enable = true;
  # https://devenv.sh/services/
  # services.postgres.enable = true;

  enterShell = ''
    git --version
  '';

  dotenv.enable = true;
  difftastic.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
