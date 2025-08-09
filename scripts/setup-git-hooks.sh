#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

hook_dir=".githooks"
hook_file="$hook_dir/pre-commit"

if [ ! -f "$hook_file" ]; then
  echo "Hook not found: $hook_file" >&2
  exit 1
fi

chmod +x "$hook_file"
git config core.hooksPath "$hook_dir"

echo "Configured git hooks path to $hook_dir and made pre-commit executable."

