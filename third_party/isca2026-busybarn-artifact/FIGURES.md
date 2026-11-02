# Per-Figure Reproduction Instructions

Commands, captions, descriptions, parameters, and speedup summaries for each
figure. See `README.md` for the top-level workflow.

---

## Fig. 8a-b, 9: Communication — AllGather, AllToAll, Failures

**Fig. 8 caption:** "Synthetic experiments: effective bandwidth is calculated
as the total communication size divided by finished time."

**Fig. 9 caption:** "Comparison of BALD and Baseline XY Routing Algorithm."

**Local:**
```bash
cd fig_communication/
make generate_exp    # generates ~1500 scripts in py/ and configs in cfg/
make run_scripts     # parallel execution, results in results/
make draw_pic        # produces pic/allgather_synthetic.pdf,
                     #          pic/alltoall_synthetic.pdf,
                     #          pic/failures.pdf (via fail_pic.py)
```

**SLURM:**
```bash
cd fig_communication/
make generate_exp
make run_slurm       # submits jobs via sbatch
# after jobs complete:
make draw_pic
```

**What it tests:** AllGather and AllToAll collective communication on a 5x5 mesh with
BALD, XY, and Ring routing across 13 message sizes (1KB-16GB). Failure experiments
test fault tolerance on a 6x6 mesh under various node/link failure scenarios.

**Speedup printed:**
- AllGather: BALD over XY bandwidth speedup (min/max/mean)
- AllToAll: BALD over XY speedup, normal and link-failure cases
- Failures: per-pattern throughput speedup across 145 failure scenarios

**Output files:**
- `pic/allgather_synthetic.pdf` → Fig. 8a
- `pic/alltoall_synthetic.pdf` → Fig. 8b
- `pic/failures.pdf` → Fig. 9

---

## Fig. 10a: Die Group Shape Sensitivity

**Paper caption:** "A Transformer block mapping on different die groups."

**Local:**
```bash
cd fig_intra_ch/
make generate_exp    # OPT-30B transformer block, 8 die-group shapes
make run_scripts     # BusyBarn vs Gemini across all configs
make draw_pic        # produces pic/intra_mapping_ch_pic.pdf
```

**SLURM:**
```bash
cd fig_intra_ch/
make generate_exp
make run_slurm
# after jobs complete:
make draw_pic
```

**What it tests:** How die-group topology (1x1, 1x2, ..., 3x3) affects transformer block
latency. X-axis: die-group shapes. Stacked bars: exposed compute / communication / overlap.

**Key parameters:** seq_len=2048, co_shape=4x4, tensorcore_grain=128x64, bw=256.

**Speedup printed:** BusyBarn over Gemini per channel shape, plus min/max/mean summary.

**Output:** `pic/intra_mapping_ch_pic.pdf`

---

## Fig. 10b: Core Shape Sensitivity

**Paper caption:** "A Transformer block mapping on one die with various core shapes."

**Local:**
```bash
cd fig_intra_coreshape/
make generate_exp    # 8 core shapes: 5x5, 4x8, 6x6, ..., 10x10
make run_scripts
make draw_pic        # produces pic/intra_mapping_coreshape_pic.pdf
```

**SLURM:**
```bash
cd fig_intra_coreshape/
make generate_exp
make run_slurm
# after jobs complete:
make draw_pic
```

**What it tests:** How core-array shape (square vs rectangular) affects intra-die mapping.
Single die (ch=1x1), tensorcore_grain=64x64, seq_len=512.

**Speedup printed:** BusyBarn over Gemini per core shape, plus min/max/mean summary.

**Output:** `pic/intra_mapping_coreshape_pic.pdf`

---

## Fig. 10c: Power/Fault Sensitivity

**Paper caption:** "A Transformer block mapping on one die with various core power and faults."

**Local:**
```bash
cd fig_intra_power/
make generate_exp    # 3 tensorcore grains × 2 failure patterns
make run_scripts
make draw_pic        # produces pic/intra_mapping_power_pic.pdf
```

**SLURM:**
```bash
cd fig_intra_power/
make generate_exp
make run_slurm
# after jobs complete:
make draw_pic
```

**What it tests:** Performance under different core power configurations (64x64, 128x64,
128x128 tensorcore grain) with and without 1 node failure. co_shape=4x4, seq_len=512.

**Speedup printed:** BusyBarn over Gemini per power/failure config, plus min/max/mean summary.

**Output:** `pic/intra_mapping_power_pic.pdf`

---

## Fig. 10d: Multi-Fault Scaling

**Paper caption:** "A Transformer Block mapping on one die at different defect rates."

**Local:**
```bash
cd fig_intra_multifaults/
make generate_exp    # 20x20 mesh, 6 failure patterns (10/15/20% defect)
make run_scripts
make draw_pic        # produces pic/multifaults_pic.pdf
```

**SLURM:**
```bash
cd fig_intra_multifaults/
make generate_exp
make run_slurm
# after jobs complete:
make draw_pic
```

**What it tests:** Fault tolerance scaling on a large 20x20 core mesh with increasing
defect rates (10%, 15%, 20%) using clustered and random failure patterns.

**Speedup printed:** BusyBarn over Gemini per fault rate/type, plus min/max/mean summary.

**Output:** `pic/multifaults_pic.pdf`

---

## Fig. 11: Workload Distribution Heatmap

**Paper caption:** "Workload distribution heatmap of Gemini and BusyBarn with link fault
between Core2-Die1 and Core0-Die3."

**Local:**
```bash
cd fig_heatmap/
bash ./run.sh 1000    # runs SA internally for each operator, --steps 1000
```

**SLURM:**
```bash
cd fig_heatmap/
bash ./run_slurm.sh 1000
```

**What it tests:** Per-core and per-link utilization after SA optimization. Shows how
BusyBarn balances workload compared to Gemini, especially with link faults.
Runs SA internally — no pre-computed results needed.

**Speedup printed:** Block latency, per-operator times, core utilization stats,
and BusyBarn vs Gemini speedup with latency reduction percentage.

**Output:** `results/hot_transformer_block_combined_step1000.pdf`

---

## Fig. 12: End-to-End Model Comparison

**Paper caption:** "4 Models End-to-End Latency Comparison."

**Local:**
```bash
cd fig_endtoend/
bash ./run.sh 16    # runs all 12 model experiments, then aggregates and plots
```

**SLURM:**
```bash
cd fig_endtoend/
bash ./run_slurm.sh       # generates locally, submits model runs via sbatch
# after all model jobs complete:
python end_to_end.py
python endtoend_pic.py
```

Or step by step:
```bash
# Step 1: Generate + run all 12 model experiments
for d in ../models/*/; do (cd "$d" && make generate_exp && make run_scripts); done

# Step 2: Aggregate results across models
python end_to_end.py

# Step 3: Plot
python endtoend_pic.py
```

**What it tests:** End-to-end transformer latency for 6 models (GPT-44L, OPT-48L,
Qwen3MoE-48L, Qwen-64L, Qwen2MoE-28L, Llama-80L) across 3 sequence lengths
(512, 2048, 8192) and 3 hardware configurations (Dojo, Cerebras, Manual).

**12 model directories** in `models/`:
- `intra_mapping_gpt_prefill`, `intra_mapping_gpt_decode`
- `intra_mapping_ch` (=OPT prefill), `intra_mapping_opt_decode`
- `intra_mapping_qwen3moe_prefill`, `intra_mapping_qwen3moe_decode`
- `intra_mapping_qwen_prefill`, `intra_mapping_qwen_decode`
- `intra_mapping_qwen2moe_prefill`, `intra_mapping_qwen2moe_decode`
- `intra_mapping_llama_prefill`, `intra_mapping_llama_decode`

**Parameters:** ch_shapes=[(1,1),(1,4),(2,2)], co_shape=4x4, grain=128x64,
sql_list=[512,2048,8192], no failures.

**Speedup printed:** Per-backend (Dojo/Cerebras/Manual) min/max/geomean,
plus overall geomean across all models, sequences, and backends.

**Output:** `endtoend.pdf`

**Note:** This is the most time-consuming experiment. Each model directory
generates hundreds of experiment scripts covering all parameter combinations.

---

## Fig. 13: SA Convergence

**Paper caption:** "Comparison of Convergence Behavior. The dashed red line represents
the near-optimal value obtained from one million search attempts."

**Local:**
```bash
cd fig_convergence/
bash ./run.sh    # 8 SA convergence traces + 4 brute-force baselines, then plots
```

**SLURM:**
```bash
cd fig_convergence/
bash ./run_slurm.sh       # submits 8 SA + 4 brute-force jobs
# after jobs complete:
python plot_convergence.py
```

**What it tests:** SA convergence speed for BusyBarn vs Gemini mapping. Tracks
transformer block latency every 100 SA steps up to 6000.
Brute-force baseline from 4000 random rounds provides near-optimal reference.

**Hardware:** Qwen2.5-7B on 2x2 mesh (wamis_hdc, default config).

**Speedup printed:** Initial/final latency per variant, reduction %, gap vs brute-force
bound, and BusyBarn over Gemini final speedup.

**Result files:**
- `results/{op}_distributed_{variant}_convergence.txt` (8 files, tab-separated: Step, Time_cost, Energy)
- `results/{op}_distributed_brute.txt` (4 files, contains "Best found: <int>")

**Output:** `results/sa_convergence.pdf`

---

## Fig. 14+15: Ablation Study + Runtime Breakdown

**Paper captions:**
- Fig. 14: "Ablation study of BusyBarn."
- Fig. 15: "Runtime Breakdown for BusyBarn. The gray arrow indicates the program execution order."

Both figures are produced from the **same experiment runs** in `fig_ablation/`.
Each run uses `--profile` to collect both simulation results (for the ablation bar chart)
and wall-clock timings (for the breakdown pie chart).

**Local:**
```bash
cd fig_ablation/
bash ./run.sh 4    # 32 runs (4 ops × 2 variants × 2 barrier × 2 reroute), max 4 parallel
make draw_pic # produces both figures
```

**SLURM:**
```bash
cd fig_ablation/
bash ./run_slurm.sh    # submits 32 jobs via sbatch
# after jobs complete:
make draw_pic
```

**What it tests:**
- *Ablation (Fig. 14):* Four mapping+routing combinations (G+XY, B+XY, G+BALD, B+BALD)
  with and without barrier synchronization. Stacked bars: Compute / Overlap / Communication,
  normalized by Gemini baseline.
- *Breakdown (Fig. 15):* Wall-clock time breakdown of the BusyBarn pipeline across 6 phases:
  Path Profiling, Building Events, Mapping (SA), BALD, Backend (event_driver), Others.

**Hardware:** Qwen2.5-32B on 1x1 chiplet with 16x16 cores (256 cores), bw=96,
seq_len=4096, 1000 SA steps.

**Speedup printed:** Per-operator and transformer-block-level speedup table for all 8
configurations (4 variants × 2 barrier modes), normalized to Gemini+XY baseline.

**Result files:** `sbatch_log/{op}_{variant}_ch1x1_bw96_16x16_sp8{barrier}{reroute}_{id}.txt`

**Output:**
- `results/transformer_block_16x16_bw96_1x1.pdf` (Fig. 14)
- `results/pie_time_busybarn_ch1x1_bw96_16x16_sp8_reroute.pdf` (Fig. 15)

**Note:** Memory intensive — each run simulates 256 cores. Run with max 4 parallel jobs
on machines with 16GB+ RAM. The breakdown pie chart measures wall-clock time of the
Python-level pipeline, not hardware simulation cycles — absolute times will differ
across platforms, but relative ordering of dominant phases should be consistent.
