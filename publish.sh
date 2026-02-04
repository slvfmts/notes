#!/bin/bash
# Publish notes: sync from Obsidian and push to GitHub

set -e

cd "$(dirname "$0")"

echo "📥 Syncing from Obsidian vault..."
./sync.sh

echo ""
echo "📤 Pushing to GitHub..."
git add content
git commit -m "Update notes" || echo "Nothing to commit"
git push

echo ""
echo "✅ Done! Site will update in ~1 minute"
echo "🔗 https://slvfmts.github.io/notes/"
