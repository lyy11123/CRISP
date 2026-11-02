#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

SUMMARY="$OUTPUT_DIR/summary.txt"
: > "$SUMMARY"

collected=0
missing=0

copy_fig() {
  local src="$1" dst="$2" label="$3"
  if [ -f "$SCRIPT_DIR/$src" ]; then
    cp "$SCRIPT_DIR/$src" "$OUTPUT_DIR/$dst"
    echo "  [OK]      $label -> $dst"
    collected=$((collected + 1))
  else
    echo "  [MISSING] $label ($src)"
    missing=$((missing + 1))
  fi
}

capture_summary() {
  local dir="$1" cmd="$2" label="$3"
  local full_dir="$SCRIPT_DIR/$dir"
  echo "" >> "$SUMMARY"
  echo "============================================================" >> "$SUMMARY"
  echo " $label" >> "$SUMMARY"
  echo "============================================================" >> "$SUMMARY"
  if (cd "$full_dir" && python $cmd >> "$SUMMARY" 2>/dev/null); then
    echo "  [OK]      Summary: $label"
  else
    echo "  (summary not available for $label)" >> "$SUMMARY"
    echo "  [SKIP]    Summary: $label (results not ready)"
  fi
}

echo "============================================"
echo "  Collecting results to output/"
echo "============================================"

echo ""
echo "--- Copying figures ---"
copy_fig "fig_communication/pic/allgather_synthetic.pdf"                    "fig08a_allgather.pdf"      "Fig. 8a  AllGather"
copy_fig "fig_communication/pic/alltoall_synthetic.pdf"                     "fig08b_alltoall.pdf"       "Fig. 8b  AllToAll"
copy_fig "fig_communication/pic/failures.pdf"                               "fig09_failures.pdf"        "Fig. 9   Failures"
copy_fig "fig_intra_ch/pic/intra_mapping_ch_pic.pdf"                        "fig10a_diegroup.pdf"       "Fig. 10a Die Group"
copy_fig "fig_intra_coreshape/pic/intra_mapping_coreshape_pic.pdf"          "fig10b_coreshape.pdf"      "Fig. 10b Core Shape"
copy_fig "fig_intra_power/pic/intra_mapping_power_pic.pdf"                  "fig10c_power.pdf"          "Fig. 10c Power/Fault"
copy_fig "fig_intra_multifaults/pic/intra_mapping_multifaults_pic.pdf"      "fig10d_multifaults.pdf"    "Fig. 10d Multi-Fault"
copy_fig "fig_heatmap/results/hot_transformer_block_combined_step1000.pdf"  "fig11_heatmap.pdf"         "Fig. 11  Heatmap"
copy_fig "fig_endtoend/endtoend.pdf"                                        "fig12_endtoend.pdf"        "Fig. 12  End-to-End"
copy_fig "fig_endtoend/endtoend_gpt.pdf"                                    "fig12_endtoend_gpt.pdf"    "Fig. 12  End-to-End (GPT)"
copy_fig "fig_convergence/results/sa_convergence.pdf"                       "fig13_convergence.pdf"     "Fig. 13  Convergence"
copy_fig "fig_ablation/results/transformer_block_16x16_bw96_1x1.pdf"       "fig14_ablation.pdf"        "Fig. 14  Ablation"
copy_fig "fig_ablation/results/pie_time_busybarn_ch1x1_bw96_16x16_sp8_reroute.pdf"  "fig15_breakdown.pdf"  "Fig. 15  Breakdown"

echo ""
echo "--- Generating summary (re-running plot scripts) ---"
capture_summary "fig_communication"    "allgather_synthetic_pic.py"  "Fig. 8a: AllGather Speedup"
capture_summary "fig_communication"    "alltoall_synthetic_pic.py"   "Fig. 8b: AllToAll Speedup"
capture_summary "fig_communication"    "fail_pic.py"                 "Fig. 9: Failure Speedup"
capture_summary "fig_intra_ch"         "intra_mapping_ch_pic.py"     "Fig. 10a: Die Group Shape Speedup"
capture_summary "fig_intra_coreshape"  "coreshape_pic.py"            "Fig. 10b: Core Shape Speedup"
capture_summary "fig_intra_power"      "intra_decoder_pic.py"        "Fig. 10c: Power/Fault Speedup"
capture_summary "fig_intra_multifaults" "multifaults_pic.py"         "Fig. 10d: Multi-Fault Speedup"
capture_summary "fig_endtoend"         "endtoend_pic.py"             "Fig. 12: End-to-End Speedup"
capture_summary "fig_endtoend"         "endtoend_pic_gpt.py"         "Fig. 12: End-to-End Speedup (GPT)"
capture_summary "fig_convergence"      "plot_convergence.py"         "Fig. 13: SA Convergence"
capture_summary "fig_ablation"         "plot_16x16_1x1.py"           "Fig. 14: Ablation"
capture_summary "fig_ablation"         "plot_pie_time.py"            "Fig. 15: Breakdown"

echo ""
echo "============================================"
echo "  Collected: $collected figures"
echo "  Missing:   $missing figures"
echo "  Output:    output/"
echo "  Summary:   output/summary.txt"
echo "============================================"
