import torch
import pandas as pd
from collections import Counter
from tqdm import tqdm


def analyze_dataset(dataset, save_path="label_statistics.xlsx"):
    print("Thống kê nhãn trong tập huấn luyện...")
    label_counter = Counter()

    for i in tqdm(range(len(dataset)), desc="Đang duyệt dataset"):
        _, mask = dataset[i]
        unique, counts = torch.unique(mask.long(), return_counts=True)
        for u, c in zip(unique.tolist(), counts.tolist()):
            label_counter[u] += c

    classes = sorted(label_counter.keys())
    counts = [label_counter[c] for c in classes]

    total = sum(counts)
    weights = [1 - (c / total) for c in counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print(f"Các lớp có trong dataset: {classes}")
    print(f"Số pixel tương ứng: {counts}")
    print(f"Trọng số loss cho từng lớp: {weights_tensor}")

    df = pd.DataFrame({"class": classes, "count": counts})
    df.to_excel(save_path, index=False)
    print(f"Đã lưu thống kê nhãn vào {save_path}")

    return weights_tensor
