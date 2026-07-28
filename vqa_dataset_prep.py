"""
Sinh du lieu VQA (Hoi-Dap Y Khoa) tu dataset ISIC 2019 cho mo hinh Qwen2-VL.
Se doc ground truth CSV, ghep voi huong dan y khoa tieng Viet, va tao ra file JSONL
dung de train LoRA (SFTTrainer).
"""
import pandas as pd
import json
import os
import random

# ==========================================================
# 1. Tu dien kien thuc (Clinical Knowledge Base) bang Tieng Viet
# ==========================================================
CLINICAL_KNOWLEDGE = {
    "MEL": "Day la dau hieu cua benh Ung thu hac to (Melanoma) - dang ung thu da nguy hiem nhat. Ton thuong thuong co dac diem ABCDE: vien khong deu, bat doi xung, va co mau sac loang lo. Benh nhan can den gap bac si da lieu ngay lap tuc de lam sinh thiet vi nguy co di can rat cao.",
    "NV": "Day la Not ruoi lanh tinh (Melanocytic Nevi). Ton thuong co hinh dang doi xung, vien ro rang va mau sac dong nhat. Day la truong hop an toan, khong can can thiep y te, tuy nhien nen tiep tuc theo doi neu co su thay doi ve kich thuoc.",
    "BCC": "Hinh anh cho thay Ung thu bieu mo te bao day (Basal Cell Carcinoma). Day la loai ung thu da pho bien nhat do tiep xuc voi tia UV. Mac du hiem khi di can, nhung no co the pha huy mo xung quanh. Benh nhan can dat lich kham trong 1-2 tuan toi de loai bo khoi u.",
    "AK": "Day la Day sung quang hoa / Benh Bowen (Actinic Keratosis). Day la mot dang ton thuong tien ung thu do tich luy anh nang mat troi nhieu nam. Benh nhan can can thiep y te som de tranh tien trien thanh ung thu bieu mo te bao vay (SCC).",
    "BKL": "Day la ton thuong Day sung lanh tinh (Benign Keratosis). Day chi la tinh trang lao hoa da hoac day sung tiet ba, thuong gaps o nguoi lon tuoi. Truong hop nay hoan toan lanh tinh va khong dang lo ngai.",
    "DF": "Day la U xo bi (Dermatofibroma), mot khoi u lanh tinh rat pho bien thuong xuat hien o chi duoi. Ton thuong thuong cung va co mau nau. Khong can dieu tri tru khi gay mat tham my hoac kho chiu.",
    "VASC": "Day la Ton thuong mach mau (Vascular Lesion), co the la u mach mau hoac gian mach. Phan lon la lanh tinh, tuy nhien benh nhan nen tham van bac si neu khoi u nay chay mau hoac phat trien nhanh."
}

# Cac kieu hoi đa dang de AI khong bi hoc vet cau hoi
QUESTIONS = [
    "Bac si hay phan tich hinh anh ton thuong da nay giup toi.",
    "Day la benh gi? Co nguy hiem khong?",
    "Cho toi xin danh gia lam sang ve not ruoi nay.",
    "Hinh anh nay the hien loai benh da lieu nao?",
    "Xin hay chuan doan buc anh y khoa sau day.",
]

def generate_vqa_dataset(csv_path, img_dir, output_jsonl):
    if not os.path.exists(csv_path):
        print(f"❌ Khong tim thay {csv_path}. Vui long kiem tra lai duong dan.")
        return

    df = pd.read_csv(csv_path)
    print(f"🔄 Dang xu ly {len(df)} anh tu ISIC 2019...")

    dataset = []
    
    # ISIC 2019 classes: MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]

    for _, row in df.iterrows():
        img_name = row['image']
        img_path = os.path.join(img_dir, f"{img_name}.jpg")
        
        # Xac dinh ground truth label
        label = None
        for cls in classes:
            if cls in row and row[cls] == 1.0:
                label = cls
                break
        
        if not label: continue # Bo qua neu thuoc nhom UNK hoac SCC (do khong co trong kien thuc)

        answer = CLINICAL_KNOWLEDGE[label]
        question = random.choice(QUESTIONS)

        # Format chuan cua Qwen-VL / chat templates
        entry = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
        }
        dataset.append(entry)

    # Chia train/val
    random.shuffle(dataset)
    train_split = int(0.9 * len(dataset))
    train_data = dataset[:train_split]
    val_data = dataset[train_split:]

    # Luu file Train
    with open(output_jsonl.replace(".jsonl", "_train.jsonl"), "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Luu file Val
    with open(output_jsonl.replace(".jsonl", "_val.jsonl"), "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Hoan thanh! Da luu {len(train_data)} cau hoi train va {len(val_data)} cau hoi val.")

if __name__ == "__main__":
    csv_file = "data/isic_2019/ISIC_2019_Training_GroundTruth.csv"
    img_folder = "data/isic_2019/images"
    out_file = "derma_vqa.jsonl"
    
    # Tao dataset Ao neu chua tai ISIC (de code khong bao loi khi ban chay test)
    if not os.path.exists(csv_file):
        print("⚠️ Tao file CSV ao de test code (vi tren Windows chua co ISIC 2019)...")
        os.makedirs(img_folder, exist_ok=True)
        # Tao dummy img
        from PIL import Image
        Image.new('RGB', (224, 224), color='red').save(os.path.join(img_folder, "ISIC_001.jpg"))
        # Tao dummy csv
        with open(csv_file, "w") as f:
            f.write("image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK\n")
            f.write("ISIC_001,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n")

    generate_vqa_dataset(csv_file, img_folder, out_file)
