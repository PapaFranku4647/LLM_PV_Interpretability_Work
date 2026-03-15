import json
import sys

filepath = r'C:\Users\lucas\Desktop\Coding\TomerResearch\LLM_PV_Working_Copy\program_synthesis\runs_step23_live_matrix\20260225_224446\runs\fn_o_seed2201\results_fn_o_L21_trial1.jsonl'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f'Total lines: {len(lines)}')

best_val = -1
best_idx = -1

for i, line in enumerate(lines):
    obj = json.loads(line.strip())

    print(f'\n=== LINE {i+1} ===')

    # Print all keys except code (too large)
    for k in obj:
        if k == 'code':
            print(f'  code: [{len(obj[k])} chars]')
        elif k == 'compile_error':
            ce = obj[k]
            if ce:
                print(f'  compile_error: {ce[:300]}')
            else:
                print(f'  compile_error: None')
        elif k == 'raw_response':
            print(f'  raw_response: [{len(str(obj[k]))} chars]')
        else:
            print(f'  {k}: {obj[k]}')

    va = obj.get('val_acc')
    if va is not None and va > best_val:
        best_val = va
        best_idx = i

print(f'\n\n=== BEST ATTEMPT ===')
print(f'Line index: {best_idx + 1}, val_acc: {best_val}')

# Now write the best code to a separate file
best_obj = json.loads(lines[best_idx].strip())
code = best_obj.get('code', '')

outpath = r'C:\Users\lucas\Desktop\Coding\TomerResearch\LLM_PV_Working_Copy\_tmp_best_code.py'
with open(outpath, 'w', encoding='utf-8') as f:
    f.write(code)

print(f'Best code written to: {outpath}')
print(f'Code length: {len(code)} chars')
