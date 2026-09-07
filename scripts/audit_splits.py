import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import pandas as pd
from pathlib import Path
import json
import sys

def main():
    print("=== SPLIT INTEGRITY AUDIT ===\n")
    
    splits_dir = Path("data/isic2019/splits")
    labels_path = Path("data/isic2019/labels/cleaned.csv")
    
    if not labels_path.exists():
        print(f"Error: {labels_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(labels_path)
    total = len(df)
    
    split_ids = {}
    for name, filename in [("train", "train_indices.csv"), ("val", "val_indices.csv"), 
                            ("cal", "cal_indices.csv"), ("test", "test_indices.csv")]:
        path = splits_dir / filename
        if path.exists():
            split_ids[name] = set(pd.read_csv(path, header=None)[0].tolist())
        else:
            split_ids[name] = set()
            
    n_train = len(split_ids["train"])
    n_val = len(split_ids["val"])
    n_cal = len(split_ids["cal"])
    n_test = len(split_ids["test"])
    
    sum_check = "PASS" if (n_train + n_val + n_cal + n_test) == total else "FAIL"
    
    print("Images:")
    print(f"  Total: {total}")
    print(f"  Train: {n_train}")
    print(f"  Val: {n_val}")
    print(f"  Calibration: {n_cal}")
    print(f"  Test: {n_test}")
    print(f"  Sum check: {sum_check}\n")
    
    overlap_pairs = [("Train", "Val", "train", "val"), 
                     ("Train", "Cal", "train", "cal"), 
                     ("Train", "Test", "train", "test"), 
                     ("Val", "Cal", "val", "cal"), 
                     ("Val", "Test", "val", "test"), 
                     ("Cal", "Test", "cal", "test")]
    
    print("Image overlap:")
    img_overlaps = {}
    for label1, label2, k1, k2 in overlap_pairs:
        overlap = len(split_ids[k1] & split_ids[k2])
        img_overlaps[f"{label1}-{label2}"] = overlap
        print(f"  {label1}-{label2}: {overlap}")
    print("")
        
    id_to_group = {}
    for _, row in df.iterrows():
        lid = row.get("lesion_id", "")
        if pd.isna(lid) or lid == "":
            lid = row["image_id"]
        id_to_group[row["image_id"]] = str(lid)
        
    split_groups = {}
    for name, ids in split_ids.items():
        split_groups[name] = set(id_to_group.get(img_id, img_id) for img_id in ids)
        
    print("Group overlap (lesion_id):")
    grp_overlaps = {}
    for label1, label2, k1, k2 in overlap_pairs:
        overlap = len(split_groups[k1] & split_groups[k2])
        grp_overlaps[f"{label1}-{label2}"] = overlap
        print(f"  {label1}-{label2}: {overlap}")
    print("")
        
    print("Class distribution:")
    class_dist = {}
    for name in split_ids.keys():
        subset = df[df["image_id"].isin(split_ids[name])]
        counts = subset["diagnosis"].value_counts().to_dict() if "diagnosis" in subset else {}
        class_dist[name] = counts
        print(f"  {name.capitalize()}: {counts}")
    print("")
    
    all_img_overlap = sum(img_overlaps.values())
    all_grp_overlap = sum(grp_overlaps.values())
    
    status = "PASS" if sum_check == "PASS" and all_img_overlap == 0 and all_grp_overlap == 0 else "FAIL"
    print(f"STATUS: {status}")
    
    out_dir = Path("research/baseline_v2/provenance")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "images": {
            "total": total,
            "train": n_train,
            "val": n_val,
            "cal": n_cal,
            "test": n_test,
            "sum_check": sum_check
        },
        "image_overlap": img_overlaps,
        "group_overlap": grp_overlaps,
        "class_distribution": class_dist,
        "status": status
    }
    
    with open(out_dir / "split_audit.json", "w") as f:
        json.dump(results, f, indent=2)
        
    sys.exit(0 if status == "PASS" else 1)

if __name__ == "__main__":
    main()
