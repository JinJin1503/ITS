import torch
import pathlib
import os

print("Đang fix patch_pt file...")

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

try:
    checkpoint = torch.load('best.pt', map_location='cpu', weights_only=False)
    print(f"Loaded checkpoint thành công")
    
    pathlib.PosixPath = temp
    
    torch.save(checkpoint, 'best_fixed.pt')
    print(f"Đã lưu file mới: best_fixed.pt")
    
    original_size = os.path.getsize('best.pt') / (1024*1024)
    fixed_size = os.path.getsize('best_fixed.pt') / (1024*1024)
    print(f"Kích thước: {original_size:.2f} MB → {fixed_size:.2f} MB")
    
    if 'model' in checkpoint:
        print(f" Model info:")
        if hasattr(checkpoint['model'], 'names'):
            print(f"   Classes: {checkpoint['model'].names}")
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
    
    print("\n Hoàn tất! Chạy lệnh sau để detect:")
    print("   python detect.py --weights best_fixed.pt --source camdoxe.jpg")
    
except Exception as e:
    print(f"❌ Lỗi: {e}")
    pathlib.PosixPath = temp