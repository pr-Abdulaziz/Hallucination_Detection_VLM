from datasets import load_dataset
import os
import PIL.Image

save_dir = "vg/test_images"
os.makedirs(save_dir, exist_ok=True)

print("Loading visual_genome dataset (streaming)...")
ds = load_dataset("visual_genome", "images_v1.2.0", split="train", streaming=True, trust_remote_code=True)

print("Downloading first 5 images...")
count = 0
for item in ds:
    img = item["image"]
    img_id = item["image_id"]
    # Ensure it is a PIL image or convert
    if not isinstance(img, PIL.Image.Image):
        # Already PIL image in datasets
        pass
    img.save(os.path.join(save_dir, f"{img_id}.jpg"))
    print(f"Saved: {img_id}.jpg")
    count += 1
    if count >= 5:
        break

print(f"Success: Saved {count} images to {save_dir}")
