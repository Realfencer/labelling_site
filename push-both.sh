#!/usr/bin/env bash
# Push main to both GitHub remotes.
# Prerequisites:
#   1. Create empty repos on GitHub (no README):
#      - https://github.com/Dagger26323/labelling_site
#      - https://github.com/Realfencer/labelling_site
#   2. Authenticate: gh auth login  (for Realfencer)
#      For Dagger26323, log in with that account or use a PAT:
#        gh auth login   OR   echo "$DAGGER_GH_TOKEN" | gh auth login --with-token
set -euo pipefail
cd "$(dirname "$0")"

echo "→ Pushing to Realfencer/labelling_site ..."
git push -u realfencer main

echo "→ Pushing to Dagger26323/labelling_site ..."
git push -u dagger main

echo "Done. Both remotes updated."
