with import <nixpkgs> { };

let
  luajit_latest = stdenv.mkDerivation {
      name = "luajit";
      src = fetchgit {
        url = "https://github.com/LuaJIT/LuaJIT.git";
        sha256 = "sha256-pjNZg9W1q7AgLxIrRRMhFVs1y/aWL1asFv2JY6c8dnE=";
      };

      buildInputs = [ makeWrapper glibc ];

      makeFlags = [ "PREFIX=$(out)" ];

      installPhase = ''
        make install PREFIX=$out
        ln -sf $out/bin/luajit-2.1.0-beta3 $out/bin/luajit
      '';
    };
in
mkShell {
  buildInputs = [ 
    luajit
    cudaPackages.cudatoolkit
    cudaPackages.libcublas
    cudaPackages.cuda_cupti
    cudaPackages.cuda_nvrtc
    glibc
  ];
  shellHook = ''
    echo "hello"

    export LD_LIBRARY_PATH="${lib.makeLibraryPath [
      cudaPackages.cudatoolkit
      cudaPackages.libcublas
      cudaPackages.cuda_cupti
      cudaPackages.cuda_nvrtc
      linuxPackages.nvidia_x11
      glibc
    ]}:$LD_LIBRARY_PATH"
    export PATH=${pkgs.cudaPackages.cudatoolkit}/bin:$PATH
    export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
  '';
}
