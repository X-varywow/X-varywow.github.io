#!/bin/bash
# 中英文字符、换行算一个
# 20240723：1322981


# chmod +x ./words.sh
# ./words.sh


src_dir="."
total_chars=0

for file in $(find "$src_dir" -name '*.md'); do
    if [[ $file != */ZZZ/* ]]; then
        chars=$(wc -m <"$file")
        total_chars=$((total_chars + chars))
    fi
done

echo "Total characters: $total_chars"
