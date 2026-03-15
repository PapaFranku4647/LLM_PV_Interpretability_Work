import os, shutil
base = r"C:\Users\lucas\Desktop\Coding\TomerResearch\LLM_PV_Working_Copy\program_synthesis\runs_step23_live_matrix"
renames = {
    "20260225_174131": "20260225_fn_o_diabetes_25s_thesis_v2_medium",
    "20260225_fn_o_diabetes_25_samples_gpt5mini_minimal": "20260225_fn_o_diabetes_25s_thesis_v2",
    "20260225_171616": "20260225_fn_o_diabetes_25s_thesis_v3",
    "20260225_181533": "20260225_fn_o_diabetes_25s_thesis_v3_medium",
    "20260225_184320": "20260225_fn_o_diabetes_25s_thesis_v2_high",
    "20260225_195043": "20260225_fn_o_diabetes_25s_thesis_v3_high",
    "20260225_224446": "20260225_fn_o_diabetes_25s_code0_thesis_aware_high",
}
for old, new in renames.items():
    src = os.path.join(base, old)
    dst = os.path.join(base, new)
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
        print(f"Renamed: {old} -> {new}")
    elif os.path.exists(dst):
        print(f"Already exists: {new}")
    else:
        print(f"Not found: {old}")

# Remove failed run (no valid Code0)
failed = os.path.join(base, "20260225_173155")
if os.path.exists(failed):
    shutil.rmtree(failed)
    print(f"Removed failed run: 20260225_173155")
