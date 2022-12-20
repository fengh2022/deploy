import torch
from model import get_model
from utils.params import arg
import os

torch_ckpt = os.path.join(arg.export_dir, 'model.pth')
onnx_file = os.path.join(arg.export_dir, 'model.onnx')

model = get_model(model_name=arg.model_name, num_classes=2)
model.load_state_dict(torch.load(torch_ckpt))

dummy_input = torch.randn(1, 3, arg.img_h, arg.img_w)


torch.onnx.export(
    model,
    dummy_input,
    onnx_file,
    input_names=['input'],
    output_names=['output']
)

print(onnx_file)