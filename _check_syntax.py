#!/usr/bin/env python3
import ast, sys

files = [
    'agent/llm.py',
    'agent/quality.py',
    'agent/pipelines/web.py',
    'agent/pipelines/books.py',
    'agent/pipelines/youtube.py',
    'agent/pipelines/threads.py',
    'agent/pipelines/instagram.py',
]
errors = []
for f in files:
    try:
        with open(f) as fh:
            ast.parse(fh.read())
        print(f'  OK: {f}')
    except SyntaxError as e:
        print(f'  FAIL: {f} - {e}')
        errors.append(f)

if errors:
    print(f'\n{len(errors)} file(s) with syntax errors')
    sys.exit(1)
else:
    print('\nAll files syntax OK')
