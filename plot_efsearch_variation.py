import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

ef_values = [1, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200]
recalls = [66.7, 66.7, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
latencies = [28, 33, 41, 46, 58, 63, 112, 175, 154, 121, 140]

# =====================================================================
# LEFT PLOT: Recall@3 vs efSearch (The Accuracy Curve)
# =====================================================================
ax1.plot(ef_values, recalls, 'o-', color='#27ae60', linewidth=2.5, markersize=7, label='Recall@3 (%)')
ax1.axvline(x=20, color='#f39c12', linestyle=':', linewidth=1.8, label='Minimum Reliable ef (ef = 20)')
ax1.axvline(x=50, color='#2980b9', linestyle='--', linewidth=2.2, label='★ Production Default (ef = 50)')

# Shaded regions
ax1.axvspan(0, 10, color='#e74c3c', alpha=0.12, label='Danger Zone: ef < 10 (Misses True Neighbors)')
ax1.axvspan(10, 50, color='#2ecc71', alpha=0.15, label='High Accuracy Zone (Recall = 100%)')
ax1.axvspan(50, 210, color='#95a5a6', alpha=0.12, label='Diminishing Returns Zone (ef > 50)')

ax1.annotate('ef = 1, 3:\nRecall drops to 66.7%\n(Missed 1 of top 3 chunks!)', 
             xy=(3, 66.7), xytext=(15, 75),
             arrowprops=dict(facecolor='#e74c3c', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', color='#c0392b', fontsize=9)

ax1.annotate('ef = 50 (Sweet Spot):\n100% Recall guaranteed\nwith sub-millisecond speed', 
             xy=(50, 100.0), xytext=(65, 88),
             arrowprops=dict(facecolor='#2980b9', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', color='#1f4788', fontsize=9)

ax1.set_title('A. Retrieval Accuracy: Recall@3 vs efSearch', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('efSearch Parameter (Candidate Pool Size on Layer 0)', fontsize=10)
ax1.set_ylabel('Recall@3 (%) vs Brute Force Ground Truth', fontsize=10)
ax1.set_xlim(0, 210)
ax1.set_ylim(50, 105)
ax1.legend(loc='lower right', fontsize=8, frameon=True)

# =====================================================================
# RIGHT PLOT: Latency (µs) vs efSearch (The Cost Curve)
# =====================================================================
ax2.plot(ef_values, latencies, 's-', color='#e74c3c', linewidth=2.5, markersize=7, label='Query Latency (µs)')
ax2.axvline(x=50, color='#2980b9', linestyle='--', linewidth=2.2, label='★ Production Default (ef = 50)')

ax2.annotate('ef = 20:\nFast (~58 µs)\nAdequate for small datasets', 
             xy=(20, 58), xytext=(25, 20),
             arrowprops=dict(facecolor='#f39c12', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', color='#d35400', fontsize=9)

ax2.annotate('ef = 100 - 200:\nLatency jumps to 140 - 175 µs\n(0% extra recall, wasted CPU!)', 
             xy=(100, 154), xytext=(80, 80),
             arrowprops=dict(facecolor='#c0392b', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', color='#c0392b', fontsize=9)

ax2.set_title('B. Computational Cost: Search Latency vs efSearch', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('efSearch Parameter (Candidate Pool Size on Layer 0)', fontsize=10)
ax2.set_ylabel('Execution Time (microseconds µs)', fontsize=10)
ax2.set_xlim(0, 210)
ax2.set_ylim(0, 200)
ax2.legend(loc='upper left', fontsize=8, frameon=True)

plt.tight_layout()
out_path_1 = r"C:\Users\BIT\Downloads\Projects\Your-own-gpt\VectorDB-Java\efsearch_variation_graph.png"
out_path_2 = r"C:\Users\BIT\.gemini\antigravity-ide\brain\66b22eea-7787-463f-8b8c-c0b4ebb8f084\efsearch_variation_graph.png"
plt.savefig(out_path_1, dpi=300, bbox_inches='tight')
plt.savefig(out_path_2, dpi=300, bbox_inches='tight')
print(f"efSearch plots saved to:\n  - {out_path_1}\n  - {out_path_2}")
