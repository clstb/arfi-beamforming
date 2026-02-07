let
  pkgs = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/nixos-25.11.tar.gz";
  }) { };
in

pkgs.mkShell {
  buildInputs = [
    pkgs.cmake
    pkgs.gnumake
    pkgs.gcc
    pkgs.cudatoolkit
    pkgs.hdf5
    pkgs.hdf5.dev
    pkgs.pkg-config
    pkgs.fftw
    pkgs.fftwFloat
    pkgs.python3
    pkgs.python3Packages.h5py
    pkgs.python3Packages.numpy
    pkgs.python3Packages.matplotlib
    pkgs.python3Packages.scipy
    pkgs.ffmpeg
  ];
}
