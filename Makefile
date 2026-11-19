.DEFAULT_GOAL := help

APP_NAME    := ViewMusic
BUNDLE_ID   := io.github.acrive82.viewmusic
BUNDLE_APP  := target/release/bundle/$(APP_NAME).app
INSTALL_DIR := /Applications
INSTALL_APP := $(INSTALL_DIR)/$(APP_NAME).app
SCHEMA_PATH := docs/reference/artifact.schema.json

# Run inside the Nix devShell automatically unless already inside one.
NIX_RUN := $(if $(IN_NIX_SHELL),,nix develop -c)

.PHONY: help shell build release install uninstall run test lint fmt schema clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

shell: ## Enter the Nix macOS development shell
	nix develop

build: ## Build the full workspace (debug)
	$(NIX_RUN) cargo build --workspace

release: ## Build and sign the release ViewMusic.app bundle
	$(NIX_RUN) ./packaging/bundle.sh

install: release ## Build, copy to /Applications, clear xattrs, and sign ViewMusic.app
	rm -rf "$(INSTALL_APP)"
	cp -R "$(BUNDLE_APP)" "$(INSTALL_APP)"
	xattr -cr "$(INSTALL_APP)"
	@IDENTITY="$$(security find-identity -v -p codesigning 2>/dev/null | awk -F'"' '/Apple Development/ {print $$2; exit}')"; \
	if [ -n "$$IDENTITY" ]; then \
		codesign --force --options runtime --identifier "$(BUNDLE_ID)" --sign "$$IDENTITY" "$(INSTALL_APP)"; \
	else \
		codesign --force --identifier "$(BUNDLE_ID)" --sign - "$(INSTALL_APP)"; \
	fi
	codesign --verify --deep --strict --verbose=2 "$(INSTALL_APP)"

uninstall: ## Remove /Applications/ViewMusic.app
	rm -rf "$(INSTALL_APP)"

run: ## Launch ViewMusic.app (/Applications if installed, else build bundle)
	@if [ -d "$(INSTALL_APP)" ]; then \
		open "$(INSTALL_APP)"; \
	elif [ -d "$(BUNDLE_APP)" ]; then \
		open "$(BUNDLE_APP)"; \
	else \
		echo "ViewMusic.app not found. Run 'make release' or 'make install' first." >&2; \
		exit 1; \
	fi

test: ## Run the workspace test suite
	$(NIX_RUN) cargo test --workspace

lint: ## Run clippy and rustfmt check across the workspace
	$(NIX_RUN) cargo clippy --workspace --all-targets
	$(NIX_RUN) cargo fmt --all -- --check

fmt: ## Format all Rust code with rustfmt
	$(NIX_RUN) cargo fmt --all

schema: ## Regenerate docs/reference/artifact.schema.json
	$(NIX_RUN) cargo run -p viz-contract --bin export-schema -- "$(SCHEMA_PATH)"

clean: ## Remove Cargo build artifacts
	$(NIX_RUN) cargo clean
