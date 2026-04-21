{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python3.withPackages (ps: with ps; [
    pip
    torch
    transformers
    datasets
    sentencepiece
    ruff
  ]);
in
pkgs.mkShell {
  buildInputs = [ python ];
}
