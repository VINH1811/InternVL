import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# 1. Cấu hình đường dẫn model (HuggingFace)
# Bạn có thể đổi thành 'OpenGVLab/InternVL2_5-1B' hoặc 'OpenGVLab/InternVL2_5-4B' nếu VRAM thấp
path = 'OpenGVLab/InternVL2_5-8B' 

# 2. Hàm tiền xử lý ảnh (Cần thiết cho InternVL)
def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    # ... (Code xử lý ảnh động chi tiết có trong README, rút gọn để demo)
    # Để đơn giản cho lần chạy đầu, ta dùng resize cơ bản nếu code trên quá dài
    transform = build_transform(input_size=image_size)
    return torch.stack([transform(image)]) 

# 3. Load Model và Tokenizer
print("Đang tải model...")
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# 4. Chuẩn bị ảnh và câu hỏi
image_path = './examples/image1.jpg' # Thay bằng đường dẫn ảnh của bạn
# Tạo ảnh mẫu nếu chưa có
try:
    image = Image.open(image_path).convert('RGB')
except:
    print(f"Không tìm thấy {image_path}, đang tải ảnh mẫu từ mạng...")
    import requests
    url = "https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

# Xử lý ảnh đơn giản cho demo
pixel_values = build_transform(input_size=448)(image).unsqueeze(0).to(torch.bfloat16).cuda()

# 5. Thực hiện Chat
question = '<image>\nHãy mô tả chi tiết bức ảnh này.'
generation_config = dict(max_new_tokens=1024, do_sample=False)

print("\n--- Bắt đầu suy luận ---")
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}')
print(f'Assistant: {response}')