"""
Extract summary and best code from the JSONL file.
Run: python _tmp_extract.py

Creates:
  _tmp_summary.txt   - summary of all attempts (readable)
  _tmp_best_code0.py - full code from the best attempt
"""
import json
import os
import sys

base = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(base,
    'program_synthesis', 'runs_step23_live_matrix', '20260225_224446',
    'runs', 'fn_o_seed2201', 'results_fn_o_L21_trial1.jsonl')

if not os.path.exists(filepath):
    print(f"ERROR: file not found: {filepath}")
    sys.exit(1)

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

out = []
out.append(f'Total lines (attempts): {len(lines)}')
out.append(f'Source: {filepath}')

best_val = -1
best_idx = -1
compiled_count = 0

for i, line in enumerate(lines):
    obj = json.loads(line.strip())
    va = obj.get('val_acc')
    ta = obj.get('test_acc')
    ce = obj.get('compile_error')

    out.append(f'\n=== ATTEMPT {obj.get("attempt", "?")} (line {i+1}) ===')
    out.append(f'  prompt_variant  : {obj.get("prompt_variant")}')
    out.append(f'  model           : {obj.get("model")}')
    out.append(f'  reasoning_effort: {obj.get("reasoning_effort")}')
    out.append(f'  val_acc         : {va}')
    out.append(f'  test_acc        : {ta}')
    out.append(f'  compile_error   : {ce}')
    out.append(f'  stopped_early   : {obj.get("stopped_early")}')
    out.append(f'  trial           : {obj.get("trial")}')
    out.append(f'  code_lines      : {obj.get("code_lines")}')
    out.append(f'  num_branches    : {obj.get("num_branches")}')
    out.append(f'  duration_ms     : {obj.get("duration_ms")}')
    code = obj.get('code') or ''
    out.append(f'  code_chars      : {len(code)}')

    if ce is None and code:
        compiled_count += 1

    if va is not None and va > best_val:
        best_val = va
        best_idx = i

out.append(f'\n\n{"="*50}')
out.append(f'COMPILED: {compiled_count} / {len(lines)}')
out.append(f'BEST ATTEMPT: line {best_idx + 1}')
out.append(f'BEST val_acc: {best_val}')

if best_idx >= 0:
    best_obj = json.loads(lines[best_idx].strip())
    out.append(f'BEST test_acc: {best_obj.get("test_acc")}')
    out.append(f'BEST prompt_variant: {best_obj.get("prompt_variant")}')
    code = best_obj.get('code', '') or ''
    code_path = os.path.join(base, '_tmp_best_code0.py')
    with open(code_path, 'w', encoding='utf-8') as f:
        f.write(code)
    out.append(f'Code written to: {code_path} ({len(code)} chars)')

summary_text = '\n'.join(out)
summary_path = os.path.join(base, '_tmp_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(summary_text)
print(f'\nSummary written to: {summary_path}')
