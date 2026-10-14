#!/usr/bin/env bash
# Rebuild the DDXPlus dataset from the split archive in data/ and unpack it to data/ddxplus/.
#
# The zip (171 MB) is cut into 45 MB parts because GitHub rejects files over 100 MB.
# Run from anywhere:  bash scripts/unpack_data.sh
set -euo pipefail

DATA="$(cd "$(dirname "$0")/.." && pwd)/data"
cd "$DATA"

if [ -f ddxplus/train.csv ]; then
  echo "data/ddxplus already unpacked - nothing to do."
  exit 0
fi

cat ddxplus.zip.part-* > ddxplus.zip                      # join the parts back into one zip

expected="$(cat ddxplus.zip.sha256)"                       # make sure nothing got corrupted
actual="$(shasum -a 256 ddxplus.zip 2>/dev/null || sha256sum ddxplus.zip)"
if [ "${actual%% *}" != "$expected" ]; then
  echo "Checksum mismatch - re-download the data/ddxplus.zip.part-* files." >&2
  rm -f ddxplus.zip
  exit 1
fi

unzip -q ddxplus.zip                                       # -> data/ddxplus/{train,validate,test}.csv + JSONs
rm ddxplus.zip
echo "Unpacked to $DATA/ddxplus"
