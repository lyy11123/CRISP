#!/usr/bin/env bash
# GPT prefill pilot (E2E_HANDOFF section 1). MHA A2 / A3h at 6 processes so that a single run stays near 1 hour.
cd /e/58Institute/PhD_TERM3/CRISP/e2e || exit 1
python e2e_pilot.py run --ops ffn --jobs 14 --tag _ffn > out/run_ffn.out 2>&1
python e2e_pilot.py run --ops mha --arms A1 --jobs 14 --tag _mha_a1 > out/run_mha_a1.out 2>&1
python e2e_pilot.py run --ops mha --arms A2 A3h --jobs 6 --tag _mha_cpcm > out/run_mha_cpcm.out 2>&1
tail -n 1 out/run_ffn.out out/run_mha_a1.out out/run_mha_cpcm.out
