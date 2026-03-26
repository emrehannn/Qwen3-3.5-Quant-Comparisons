import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load the completed result
result_file = Path("results/completed/Qwen3-4B-Q8_perplexity.json")
with open(result_file) as f:
    data = json.load(f)

# Create a simple bar chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(["Qwen3-4B-Q8"], [data["perplexity"]], color="steelblue")
ax.set_ylabel("Perplexity")
ax.set_title("WikiText-103 Perplexity - First Benchmark Result")
ax.set_ylim(0, max(data["perplexity"] * 1.2, 25))

# Add value label on bar
ax.text(0, data["perplexity"] + 0.5, f'{data["perplexity"]:.2f}', 
        ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("results/verification_graph.png", dpi=150)
print(f"✓ Graph saved: results/verification_graph.png")
print(f"✓ Perplexity: {data['perplexity']:.2f}")
print(f"✓ Total tokens processed: {data['total_tokens']:,}")
