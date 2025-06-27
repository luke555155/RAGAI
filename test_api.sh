#!/bin/bash
# Usage: ./test_api.sh path_to_document.pdf
set -e
FILE=${1:-sample.pdf}

UPLOAD=$(curl -s -F "file=@${FILE}" http://localhost:8000/api/upload)
echo "$UPLOAD"
DOC_ID=$(echo "$UPLOAD" | jq -r '.document_id')

ASK=$(curl -s -X POST http://localhost:8000/api/ask \
    -H "Content-Type: application/json" \
    -d "{\"question\":\"文件內容是什麼？\",\"document_id\":\"$DOC_ID\"}")
echo "$ASK"

curl -s http://localhost:8000/api/docs/$DOC_ID | jq '.'
