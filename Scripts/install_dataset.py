# install dataset
from datasets import load_dataset # type: ignore
import os

dataset = load_dataset("cj-mills/hagrid-classification-512p-no-gesture-150k")

sample = dataset['train'][0]
print(sample.keys())  # Should show 'image', 'label', etc.


output_dir = "exported_dataset"
os.makedirs(output_dir, exist_ok=True)

for i, sample in enumerate(dataset["train"]):
    label = sample['label']
    label_dir = os.path.join(output_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)

    sample['image'].save(os.path.join(label_dir, f"img_{i}.jpg"))

    if i % 1000 == 0:
        print(f"Exported {i} images...")