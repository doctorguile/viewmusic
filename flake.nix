{
  description = "ViewMusic macOS development and build environment";

  inputs = {
    nixpkgs.url = "https://flakehub.com/f/DeterminateSystems/nixpkgs-weekly/0.1";
  };

  outputs = { nixpkgs, ... }:
    let
      darwinSystems = [
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      forDarwin =
        f: nixpkgs.lib.genAttrs darwinSystems (system: f (import nixpkgs { inherit system; }));
    in
    {
      devShells = forDarwin (pkgs: {
        default = pkgs.mkShellNoCC {
          packages = with pkgs; [
            rustc
            cargo
            clippy
            rustfmt
            rust-analyzer
          ];
        };
      });
    };
}
