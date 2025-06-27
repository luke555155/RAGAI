#!/usr/bin/env python
import argparse
import requests

BASE_URL = 'http://localhost:8000'


def upload(file_path: str):
    with open(file_path, 'rb') as f:
        resp = requests.post(f'{BASE_URL}/api/upload', files={'file': f})
    resp.raise_for_status()
    return resp.json()


def ask(question: str, document_id: str):
    resp = requests.post(
        f'{BASE_URL}/api/ask',
        json={'question': question, 'document_id': document_id},
    )
    resp.raise_for_status()
    return resp.json()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test backend API')
    parser.add_argument('file', help='path to document for upload')
    parser.add_argument('question', help='question to ask')
    args = parser.parse_args()

    up = upload(args.file)
    print('Upload:', up)
    doc_id = up['document_id']
    ans = ask(args.question, doc_id)
    print('Answer:', ans)

