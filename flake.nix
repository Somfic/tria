{
  description = "tria — Bevy-based Rust engine workspace";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
        };

        # Native deps Bevy links/dlopen's at build and run time.
        buildInputs = with pkgs; [
          udev
          alsa-lib
          vulkan-loader
          # X11
          libx11
          libxcursor
          libxi
          libxrandr
          libxkbcommon
          # Wayland
          wayland
        ];

        nativeBuildInputs = with pkgs; [
          pkg-config
        ];

        # Libraries Bevy loads at runtime via dlopen must be on the linker path.
        libraryPath = pkgs.lib.makeLibraryPath buildInputs;
      in
      {
        devShells.default = pkgs.mkShell {
          inherit buildInputs nativeBuildInputs;

          packages = [ rustToolchain ];

          LD_LIBRARY_PATH = libraryPath;
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";

          shellHook = ''
            echo "tria dev shell — $(rustc --version)"
          '';
        };
      });
}
