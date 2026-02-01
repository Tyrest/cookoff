#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running basic_bot vs do_nothing_bot on all maps..."
echo "=================================================="

shopt -s nullglob globstar

for map in maps/*.txt maps/**/*.txt; do
    echo ""
    echo "Map: $map"
    echo "---"
    output=$(python src/game.py --red bots/do_nothing_bot.py --blue bots/basic_bot.py --map "$map" 2>&1)
    result=$(echo "$output" | grep -E "(GAME|winner|Winner|WIN|crashed|failed|Final)" | tail -5)
    if [ -z "$result" ]; then
        result=$(echo "$output" | tail -3)
    fi
    echo "$result"
done

echo ""
echo "=================================================="
echo "Done!"
