#!/bin/bash
set -e

# 引数が渡された場合はそのまま実行
if [ "$#" -gt 0 ]; then
    exec conda run --no-capture-output -n pi0 "$@"
fi

# 引数なし → インタラクティブシェル
exec conda run --no-capture-output -n pi0 /bin/bash
