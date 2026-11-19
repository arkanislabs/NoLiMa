#!/usr/bin/env bash
set -euo pipefail

# Run from the directory containing this script, regardless of the caller's cwd.
cd "$(dirname "$0")"

BASE_URL="https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main"

# Download (resumable) using curl, which ships with macOS and most Linux distros.
fetch() {
    curl -fL --retry 3 -C - -O "$1"
}

mkdir -p needlesets
cd needlesets
fetch "$BASE_URL/needlesets/needle_set.json"
fetch "$BASE_URL/needlesets/needle_set_MC.json"
fetch "$BASE_URL/needlesets/needle_set_ONLYDirect.json"
fetch "$BASE_URL/needlesets/needle_set_hard.json"
fetch "$BASE_URL/needlesets/needle_set_w_CoT.json"
fetch "$BASE_URL/needlesets/needle_set_w_Distractor.json"

cd ..
mkdir -p haystack/rand_shuffle
cd haystack/rand_shuffle
fetch "$BASE_URL/haystack/rand_shuffle/rand_book_1.txt"
fetch "$BASE_URL/haystack/rand_shuffle/rand_book_2.txt"
fetch "$BASE_URL/haystack/rand_shuffle/rand_book_3.txt"
fetch "$BASE_URL/haystack/rand_shuffle/rand_book_4.txt"
fetch "$BASE_URL/haystack/rand_shuffle/rand_book_5.txt"

cd ..
mkdir -p rand_shuffle_long
cd rand_shuffle_long
fetch "$BASE_URL/haystack/rand_shuffle_long/rand_book_1.txt"
fetch "$BASE_URL/haystack/rand_shuffle_long/rand_book_2.txt"
fetch "$BASE_URL/haystack/rand_shuffle_long/rand_book_3.txt"
fetch "$BASE_URL/haystack/rand_shuffle_long/rand_book_4.txt"
fetch "$BASE_URL/haystack/rand_shuffle_long/rand_book_5.txt"
