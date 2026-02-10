# compare_distributions_fixed.py
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("COMPARING HALLUCINATIONS TO TRAINING DATA")
print("=" * 60)

# Load training data heatmap
print("\nLoading COCO training heatmap...")
training_heatmap = np.load('coco_training_heatmap.npy')
print("✓ Loaded")

# Load hallucination detections
print("\nLoading hallucination detections...")
try:
    all_detections = np.load('hallucination_detections.npy', allow_pickle=True)
    print(f"✓ Loaded {len(all_detections)} hallucinations")
except FileNotFoundError:
    print("✗ No hallucinations found!")
    print("Run quick_detect.py first")
    exit()

if len(all_detections) == 0:
    print("\n⚠️  No hallucinations detected!")
    exit()

# Create hallucination heatmap
print("\nCreating hallucination heatmap...")
grid_size = 10
hallucination_heatmap = np.zeros((grid_size, grid_size))

for det in all_detections:
    x = int(det['center_x'] * (grid_size - 1))
    y = int(det['center_y'] * (grid_size - 1))
    x = max(0, min(grid_size-1, x))
    y = max(0, min(grid_size-1, y))
    hallucination_heatmap[y, x] += 1

# Normalize
if hallucination_heatmap.sum() > 0:
    hallucination_heatmap = hallucination_heatmap / hallucination_heatmap.sum()

print("✓ Heatmap created")

# Calculate correlation
correlation = np.corrcoef(
    training_heatmap.flatten(), 
    hallucination_heatmap.flatten()
)[0, 1]

print(f"\n✓ Correlation: {correlation:.3f}")

# Create visualization
print("\nCreating visualization...")
fig = plt.figure(figsize=(18, 5))

# Training data
ax1 = plt.subplot(1, 3, 1)
im1 = ax1.imshow(training_heatmap, cmap='hot', interpolation='nearest')
ax1.set_title('Training Data\n(Where Real Objects Are)', 
              fontweight='bold', fontsize=14)
ax1.set_xlabel('Horizontal Position')
ax1.set_ylabel('Vertical Position')
plt.colorbar(im1, ax=ax1, label='Probability')

# Hallucinations
ax2 = plt.subplot(1, 3, 2)
im2 = ax2.imshow(hallucination_heatmap, cmap='hot', interpolation='nearest')
ax2.set_title('Hallucinations\n(Where YOLO Invents Objects)', 
              fontweight='bold', fontsize=14)
ax2.set_xlabel('Horizontal Position')
ax2.set_ylabel('Vertical Position')
plt.colorbar(im2, ax=ax2, label='Probability')

# Difference
ax3 = plt.subplot(1, 3, 3)
difference = hallucination_heatmap - training_heatmap
im3 = ax3.imshow(difference, cmap='RdBu', interpolation='nearest', 
                vmin=-0.15, vmax=0.15)
ax3.set_title(f'Difference\nCorrelation = {correlation:.3f}', 
              fontweight='bold', fontsize=14)
ax3.set_xlabel('Horizontal Position')
ax3.set_ylabel('Vertical Position')
plt.colorbar(im3, ax=ax3, label='Difference')

plt.tight_layout()
plt.savefig('hallucination_vs_training_FINAL.png', dpi=150, bbox_inches='tight')
print("✓ Saved: hallucination_vs_training_FINAL.png")

# Print interpretation
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nCorrelation: {correlation:.3f}")
print("\nInterpretation:")

if correlation > 0.5:
    print("  🔴 STRONG EVIDENCE of dataset-induced hallucinations!")
    print("  Hallucinations closely follow training data patterns.")
    print("  This proves spatial bias causes hallucinations!")
    
elif correlation > 0.3:
    print("  🟡 MODERATE EVIDENCE of dataset bias influence.")
    print("  Hallucinations somewhat follow training patterns.")
    
elif correlation > 0.1:
    print("  🟢 WEAK correlation.")
    print("  Hallucinations may be more random.")
    
else:
    print("  ⚪ NO significant correlation.")

# Show where hallucinations cluster
flat_halluc = hallucination_heatmap.flatten()
sorted_indices = np.argsort(flat_halluc)[::-1]

print("\nTop 5 hallucination hotspots:")
for i in range(min(5, len(sorted_indices))):
    idx = sorted_indices[i]
    if flat_halluc[idx] == 0:
        break
    row, col = idx // grid_size, idx % grid_size
    print(f"  {i+1}. Grid({row},{col}) - Y:{row*10}-{(row+1)*10}%, X:{col*10}-{(col+1)*10}% - {flat_halluc[idx]:.1%}")

print("\n" + "=" * 60)
print("✓ Analysis complete!")
print("=" * 60)

plt.show()