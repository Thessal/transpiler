let
  pkgs = import <nixpkgs> { };

  inherit (pkgs) lib;

  pyproject-nix = import (builtins.fetchGit {
    url = "https://github.com/pyproject-nix/pyproject.nix.git";
  }) {
    inherit lib;
  };
  
  project = pyproject-nix.lib.project.loadPyproject {
    projectRoot = ./.;
  };

  python = pkgs.python313.override {
    packageOverrides = self: super: {
      butterflow = self.buildPythonPackage rec {
        pname = "butterflow";
        version = "0.1.0";

        src = pkgs.fetchFromGitHub {
          owner = "Thessal";
          repo = "butterflow";
          rev = "main";
          sha256 = "sha256-meXlB4baqOq1aF3QqktyHxP1/U0fTscnYhQHxa1wkiA="; 
        };

        nativeBuildInputs = [ self.hatchling ];

        propagatedBuildInputs = [ 
          self.numpy
          self.scipy
        ];

        pyproject = true;
        doCheck = false;
      };
    };
  };

  arg = project.renderers.withPackages { inherit python; };
  pythonEnv = python.withPackages arg;


  morphoPackage = python.pkgs.buildPythonPackage rec {
    pname = "morpho"; 
    version = "0.1.0"; 
    src = ./.; 

    buildInputs = with python.pkgs; [
      hatchling
      pythonEnv
    ];
    
    format = "pyproject"; 
    doCheck = false; 
    doInstallCheck = false;
    dontCheck = true;
  };

in pkgs.mkShell { 
  packages = [ 
    pythonEnv
    morphoPackage
    pkgs.uv
    ]; 
}

