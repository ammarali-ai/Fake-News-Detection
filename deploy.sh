#!/usr/bin/env bash
# Multilingual Fake News Detector — deploy.sh
# Push the current branch to a HuggingFace Space.
#
# Usage:
#   ./deploy.sh <hf-username> [space-name] [--force]
#
# Prerequisites:
#   1. pip install huggingface_hub
#   2. huggingface-cli login
#   3. Create the Space at https://huggingface.co/new-space (SDK: Gradio)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <hf-username> [space-name] [--force]" >&2
    exit 1
fi

SPACE_OWNER="$1"
SPACE_NAME="${2:-multilingual-fake-news-detector}"
FORCE_PUSH=""
for arg in "$@"; do
    if [[ "$arg" == "--force" ]]; then
        FORCE_PUSH="--force"
    fi
done

assert_command() {
    local name="$1"
    local hint="$2"
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "Required command '$name' not found on PATH. $hint" >&2
        exit 1
    fi
}

assert_command git "Install Git from https://git-scm.com/."
assert_command huggingface-cli "Install with: pip install huggingface_hub"

echo "Checking HuggingFace login status..."
if ! WHOAMI=$(huggingface-cli whoami 2>&1); then
    echo "Not logged into HuggingFace. Run: huggingface-cli login" >&2
    exit 1
fi
echo "Logged in as: $WHOAMI"

REMOTE_URL="https://huggingface.co/spaces/${SPACE_OWNER}/${SPACE_NAME}"

if EXISTING=$(git remote get-url hf 2>/dev/null); then
    if [[ "$EXISTING" != "$REMOTE_URL" ]]; then
        echo "Remote 'hf' already exists and points to: $EXISTING" >&2
        echo "Will overwrite to: $REMOTE_URL"
        read -r -p "Continue? (y/N) " confirm
        if [[ "$confirm" != "y" ]]; then
            echo "Aborted."
            exit 0
        fi
        git remote set-url hf "$REMOTE_URL"
    fi
else
    echo "Adding remote 'hf' -> $REMOTE_URL"
    git remote add hf "$REMOTE_URL"
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Pushing branch '$CURRENT_BRANCH' to Space '${SPACE_OWNER}/${SPACE_NAME}'..."

if ! git push hf "${CURRENT_BRANCH}:main" ${FORCE_PUSH}; then
    echo "git push failed. If this is the first push to an existing Space with unrelated history, re-run with --force." >&2
    exit 1
fi

echo
echo "Deployed."
echo "Space URL: $REMOTE_URL"
