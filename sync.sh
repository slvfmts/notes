#!/bin/bash
# Sync public notes from Obsidian vault to Quartz content folder

VAULT_PUBLIC="/Users/slava/Library/Mobile Documents/iCloud~md~obsidian/Documents/slvfmts/public"
QUARTZ_CONTENT="/Users/slava/quartz/content"

# Sync files (delete removed, preserve newer)
rsync -av --delete "$VAULT_PUBLIC/" "$QUARTZ_CONTENT/"

echo "Synced from $VAULT_PUBLIC"
echo ""
echo "Files in content:"
ls -la "$QUARTZ_CONTENT"
