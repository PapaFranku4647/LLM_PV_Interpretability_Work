import json
import os

filepath = r'C:\Users\lucas\Desktop\Coding\TomerResearch\LLM_PV_Working_Copy\program_synthesis\runs_step23_live_matrix\20260225_224446\runs\fn_o_seed2201\results_fn_o_L21_trial1.jsonl'
outdir = r'C:\Users\lucas\Desktop\Coding\TomerResearch\LLM_PV_Working_Copy'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

summary_lines = []
summary_lines.append(f'Total lines: {len(lines)}')

best_val = -1
best_idx = -1

for i, line in enumerate(lines):
    obj = json.loads(line.strip())
    summary_lines.append(f'\n=== LINE {i+1} ===')
    for k in obj:
        if k == 'code':
            summary_lines.append(f'  code: [{len(obj[k])} chars]')
        elif k == 'compile_error':
            ce = obj[k]
            if ce:
                summary_lines.append(f'  compile_error: {ce[:500]}')
            else:
                summary_lines.append(f'  compile_error: None')
        elif k == 'raw_response':
            summary_lines.append(f'  raw_response: [{len(str(obj[k]))} chars]')
        else:
            summary_lines.append(f'  {k}: {obj[k]}')

    va = obj.get('val_acc')
    if va is not None and va > best_val:
        best_val = va
        best_idx = i

summary_lines.append(f'\n\n=== BEST ATTEMPT ===')
summary_lines.append(f'Line index: {best_idx + 1}, val_acc: {best_val}')

# Write summary
with open(os.path.join(outdir, '_tmp_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

# Write best code
best_obj = json.loads(lines[best_idx].strip())
code = best_obj.get('code', '')
with open(os.path.join(outdir, '_tmp_best_code0.py'), 'w', encoding='utf-8') as f:
    f.write(code)

print('Done')
