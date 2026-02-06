import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
import torch
import tifffile  # Thư viện đọc TIF chuyên dụng

# [QUAN TRỌNG] Import GRAUpp
from graupp import GRAUpp

app = Flask(__name__)

# --- CONFIGURATION ---
NUM_CLASSES = 5
PIXEL_SIZE_M2 = 25.0 
# English Class Names
CLASS_NAMES = ["Background", "Level 1 (Very Low)", "Level 2 (Low)", "Level 3 (Medium)", "Level 4 (High)"]

COLOR_MAP = np.array([
    [0, 0, 0],      # Black
    [139, 0, 0],    # Dark Red
    [255, 165, 0],  # Orange
    [255, 255, 0],  # Yellow
    [50, 205, 50]   # Green
], dtype=np.uint8)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Device: {device}")

# [QUAN TRỌNG] Khởi tạo model GRAUpp
# Input 10 kênh, Output 5 lớp
model = GRAUpp(num_classes=NUM_CLASSES, input_channels=10).to(device).float()

if os.path.exists("best_model.pth"):
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.eval()
        print("✅ Model GRAUpp loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Gợi ý: Kiểm tra lại file best_model.pth có đúng là của GRAUpp không?")
else:
    print("⚠️ Warning: 'best_model.pth' not found. Using random weights.")

def image_to_base64(img_np):
    """Convert numpy (0-1 or 0-255) to base64 for web display"""
    # Nếu ảnh có nhiều hơn 3 kênh, chỉ lấy 3 kênh đầu để hiển thị RGB
    if img_np.ndim == 3 and img_np.shape[2] > 3:
        img_np = img_np[:, :, :3]

    if img_np.max() <= 1.0: 
        img_np = (img_np * 255).astype(np.uint8)
    else: 
        img_np = img_np.astype(np.uint8)
        
    img_pil = Image.fromarray(img_np)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def normalize_min_max(img):
    """
    [TRAIN LOGIC] Matches training preprocessing exactly.
    (img - min) / (max - min)
    """
    img = img.astype(np.float32)
    min_v, max_v = img.min(), img.max()
    # Tránh chia cho 0 nếu ảnh chỉ có 1 màu
    if max_v > min_v:
        img = (img - min_v) / (max_v - min_v)
    else:
        img = img * 0 # Nếu max == min thì trả về 0
    return img

def read_image_file(file_bytes, is_mask=False):
    """Smart image reader (supports standard images & GeoTIFF)"""
    file_stream = io.BytesIO(file_bytes)
    img_np = None

    # 1. Try Pillow first (JPG, PNG)
    try:
        img = Image.open(file_stream)
        if not is_mask and img.mode != 'RGB': 
            img = img.convert("RGB")
        img_np = np.array(img)
    except:
        # 2. Fallback to Tifffile (GeoTIFF / Satellite TIF)
        try:
            file_stream.seek(0)
            img_np = tifffile.imread(file_stream)
            
            # Handle channel-first format (C, H, W) -> (H, W, C)
            if img_np.ndim == 3 and img_np.shape[0] < 20: 
                img_np = np.transpose(img_np, (1, 2, 0))
                
        except Exception as e:
            raise ValueError(f"Cannot read image file. Error: {e}")

    return img_np

def calculate_stats(pred_mask, gt_mask=None):
    stats = []
    for i in range(NUM_CLASSES):
        count_pred = np.sum(pred_mask == i)
        area_pred = (count_pred * PIXEL_SIZE_M2) / 1_000_000 # km2
        
        row = {
            "class_name": CLASS_NAMES[i],
            "color": f"rgb{tuple(COLOR_MAP[i])}",
            "pred_area": round(area_pred, 5),
            "gt_area": "-",
            "diff": "-"
        }
        if gt_mask is not None:
            count_gt = np.sum(gt_mask == i)
            area_gt = (count_gt * PIXEL_SIZE_M2) / 1_000_000
            diff = area_pred - area_gt
            row["gt_area"] = round(area_gt, 5)
            row["diff"] = round(diff, 5)
        stats.append(row)
    return stats

def colorize_mask(mask):
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(NUM_CLASSES):
        color_img[mask == i] = COLOR_MAP[i]
    return color_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image_file' not in request.files:
        return jsonify({'error': 'Missing satellite image file!'})
    
    # 1. Read Satellite Image
    img_file = request.files['image_file']
    img_bytes = img_file.read()
    
    try:
        # Read to numpy
        raw_img = read_image_file(img_bytes, is_mask=False)
        
        # ========================================================
        # [DEBUGGING LOGS]
        # ========================================================
        print("\n" + "="*50)
        print(f"📂 File Uploaded: {img_file.filename}")
        print(f"📊 Original Shape: {raw_img.shape}")
        
        if raw_img.ndim == 3:
            channels = raw_img.shape[2]
            if channels == 10:
                print("✅ STATUS: PERFECT! (10 Channels Detected)")
            elif channels == 3:
                print("⚠️ STATUS: WARNING! (3 Channels Detected - RGB only)")
                print("   -> Simulating missing 7 channels. Accuracy may drop.")
            else:
                print(f"⚠️ STATUS: UNUSUAL ({channels} Channels Detected)")
        print("="*50 + "\n")
        # ========================================================

        # [CRITICAL] Normalize exactly like Training
        norm_img = normalize_min_max(raw_img)
        
        # Handle Tensor Input (10 Channels simulation)
        if norm_img.ndim == 3 and norm_img.shape[2] == 10:
            tensor_input = torch.from_numpy(norm_img).permute(2, 0, 1) # (10, H, W)
        else:
            # Simulate 10 channels from 3 channels
            if norm_img.ndim == 3 and norm_img.shape[2] > 3:
                rgb_part = norm_img[:, :, :3]
            else:
                rgb_part = norm_img
                
            tensor_rgb = torch.from_numpy(rgb_part).permute(2, 0, 1) # (3, H, W)
            # Stacking [RGB, RGB, RGB, R] -> 3 + 3 + 3 + 1 = 10
            tensor_input = torch.cat([tensor_rgb, tensor_rgb, tensor_rgb, tensor_rgb[0:1]], dim=0)

        # Add Batch Dimension & Float32
        tensor_input = tensor_input.unsqueeze(0).to(device).float()

        # 2. Read Ground Truth Mask (If exists)
        gt_mask_np = None
        gt_base64 = None
        if 'mask_file' in request.files and request.files['mask_file'].filename != '':
            mask_file = request.files['mask_file']
            gt_mask_np = read_image_file(mask_file.read(), is_mask=True)
            if gt_mask_np.ndim == 3: gt_mask_np = gt_mask_np[:,:,0] 
            gt_mask_np = gt_mask_np.astype(np.uint8)
            gt_base64 = image_to_base64(colorize_mask(gt_mask_np))

        # 3. Inference
        with torch.no_grad():
            output = model(tensor_input)
            # GRAUpp thường trả về list (Deep Supervision), lấy cái cuối cùng (final output)
            if isinstance(output, list): output = output[-1]
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
        # 4. Return Results
        display_img = raw_img
        # Ensure display image is normalized for viewing
        if display_img.max() > 1.0:
            display_img = normalize_min_max(display_img)

        return jsonify({
            'original_image': image_to_base64(display_img),
            'pred_image': image_to_base64(colorize_mask(pred_mask)),
            'gt_image': gt_base64,
            'stats': calculate_stats(pred_mask, gt_mask_np)
        })

    except Exception as e:
        print(f"Server Error: {e}")
        import traceback
        traceback.print_exc() # In chi tiết lỗi nếu có
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)