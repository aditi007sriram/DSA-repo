# analyze_coco_fixed.py
import json
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("ANALYZING COCO SPATIAL DISTRIBUTION")
print("=" * 60)

# Check if COCO data exists
annotation_file = 'coco_data/annotations/instances_train2017.json'

if not os.path.exists(annotation_file):
    print("ERROR: COCO annotations not found!")
    print("Please run the download script first.")
    exit()

print("\nLoading COCO data...")
with open(annotation_file, 'r') as f:
    coco_data = json.load(f)

print(f"✓ Loaded {len(coco_data['images'])} images")
print(f"✓ Loaded {len(coco_data['annotations'])} annotations")

# Create spatial heatmap
print("\nAnalyzing spatial distribution...")
grid_size = 10
heatmap = np.zeros((grid_size, grid_size))

# Get image dimensions
image_dims = {}
for img in coco_data['images']:
    image_dims[img['id']] = (img['width'], img['height'])

print("✓ Image dimensions loaded")

# Process annotations
sample_size = min(100000, len(coco_data['annotations']))
print(f"\nProcessing {sample_size} annotations...")

processed = 0
for ann in coco_data['annotations'][:sample_size]:
    img_id = ann['image_id']
    
    if img_id not in image_dims:
        continue
    
    img_width, img_height = image_dims[img_id]
    bbox = ann['bbox']
    
    # Calculate center
    center_x = (bbox[0] + bbox[2]/2) / img_width
    center_y = (bbox[1] + bbox[3]/2) / img_height
    
    # Clamp to valid range
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    
    # Add to heatmap
    grid_x = int(center_x * (grid_size - 1))
    grid_y = int(center_y * (grid_size - 1))
    heatmap[grid_y, grid_x] += 1
    
    processed += 1
    if processed % 10000 == 0:
        print(f"  Progress: {processed} / {sample_size}")

print(f"✓ Processed {processed} annotations")

# Normalize
heatmap = heatmap / heatmap.sum()

# Save heatmap
np.save('coco_training_heatmap.npy', heatmap)
print("\n✓ Saved: coco_training_heatmap.npy")

# Create visualization
print("\nCreating visualization...")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
ax.set_title('Where Objects Appear in COCO Training Data\n(Hotter = More Objects)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Horizontal Position (Left → Right)', fontsize=12)
ax.set_ylabel('Vertical Position (Top → Bottom)', fontsize=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Probability Density', fontsize=12)

# Add grid labels
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.set_xticklabels([f'{i*10}%' for i in range(grid_size)])
ax.set_yticklabels([f'{i*10}%' for i in range(grid_size)])

plt.tight_layout()
plt.savefig('coco_spatial_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: coco_spatial_distribution.png")

# Print statistics
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# Find hottest regions
flat = heatmap.flatten()
top_indices = np.argsort(flat)[::-1][:5]

print("\nTop 5 hottest regions:")
for i, idx in enumerate(top_indices):
    row = idx // grid_size
    col = idx % grid_size
    prob = flat[idx]
    print(f"  {i+1}. Row {row}, Col {col} - {prob:.1%} of objects")

# Calculate biases
center_region = heatmap[3:7, 3:7].sum()
bottom_half = heatmap[5:, :].sum()

print(f"\n% in center 40%: {center_region:.1%}")
print(f"% in bottom half: {bottom_half:.1%}")

print("\n" + "=" * 60)
print("KEY FINDING: Objects cluster in predictable regions!")
print("=" * 60)

print("\nOpening visualization...")
plt.show()