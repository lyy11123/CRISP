#!/bin/bash
# Full speed-up regression with the current fast_hop / fast_cpcm (E2E_HANDOFF.md section 3): decode 4 mappings (C4 + hop seeds 1-4),
# prefill FFN A1 (C4 + hop seeds 1-4) and A3h (C4 + hop seed 1), then the 18 DSTU objects. Results in out/speedup/final/.
cd /e/58Institute/PhD_TERM3/CRISP/e2e
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
F=out/speedup/final; mkdir -p $F/dstu
python speedup_check.py pilot out/readonly/prefill_ffn_A1_spec.json $F/prefill_ffn_A1.json --seeds 1 2 3 4 --parts c4 hop > $F/prefill_ffn_A1.log 2>&1 &
python speedup_check.py pilot out/readonly/prefill_ffn_A3h_spec.json $F/prefill_ffn_A3h.json --seeds 1 --parts c4 hop > $F/prefill_ffn_A3h.log 2>&1 &
for m in decode_ffn_A1 decode_ffn_A3h; do python speedup_check.py pilot out/readonly/${m}_spec.json $F/$m.json --seeds 1 2 3 4 --parts c4 hop > $F/$m.log 2>&1; done
for m in decode_mha_A1 decode_mha_A3h; do python speedup_check.py pilot out/readonly_nobk/${m}.spec $F/$m.json --seeds 1 2 3 4 --parts c4 hop > $F/$m.log 2>&1; done
seq 0 17 | xargs -P 3 -I{} sh -c "python speedup_dstu.py one {} $F/dstu/item{}.json > $F/dstu/item{}.log 2>&1"
wait
grep -h 'identical' $F/*.log
for f in $F/dstu/item*.json; do python -c "import json,sys;r=json.load(open(sys.argv[1]));print(r['index'],r['name'],r['identical'],round(r['old_wall_s'],1),round(r['new_wall_s'],1))" $f; done
