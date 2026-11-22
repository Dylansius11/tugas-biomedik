#!/usr/bin/env bash
# Downloads the Danilov et al. (2022) lung segmentation collection from Mendeley Data
# (DOI 10.17632/8gf9vpkhgy.2, CC BY 4.0) into data/raw and verifies SHA-256.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
BASE="https://data.mendeley.com/public-files/datasets/8gf9vpkhgy/files"
get() { # name id sha256
  if [ ! -f "$1" ] || ! echo "$3  $1" | sha256sum -c --status; then
    curl -L --fail --retry 3 -C - -o "$1" "$BASE/$2/file_downloaded"
  fi
  echo "$3  $1" | sha256sum -c
}
get Montgomery.zip 5e8272dc-353c-442d-bf8d-6eacd1690e62 2d2b0069531313f529a05408465c8639b3ca32612eac7e94a8a0e81a67471eed
get Shenzhen.zip   75fb6dcb-4bcf-4e72-916c-00251a184eec d060975a2d6dd2ca11e5ead8589651d7d9b95431e2d02f8ce67ee3134d1d720d
get utils.py       281197a6-4897-4eaa-a7d5-9d6676421eb0 d5fdf46985c3adf365c5f8312ce64b1c803f22eb6f04244d14a1de943d99ab45
get Darwin.zip     36924a7f-315c-4b1a-b5d1-9b406fab40db 7a10e501b8e13e1321fa9701f8823db48db1bb1441f8cde3e57dc425eb3169e4
