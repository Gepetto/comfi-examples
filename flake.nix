{
  description = "human-robot Collaboration Oriented Markerless For Industry dataset utils";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } (
      { self, lib, ... }:
      {
        systems = [ "x86_64-linux" ];
        flake.overlays.default = final: prev: {
          pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
            (
              python-final: python-prev:
              lib.filesystem.packagesFromDirectoryRecursive {
                inherit (python-final) callPackage;
                directory = ./pkgs;
              }
            )
          ];
        };
        perSystem =
          { pkgs, system, ... }:
          {
            _module.args.pkgs = import inputs.nixpkgs {
              inherit system;
              overlays = [ self.overlays.default ];
            };
            packages.default = pkgs.python3Packages.comfi-examples;
          };
      }
    );
}
