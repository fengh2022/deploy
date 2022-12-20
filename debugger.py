import onnx
import torch.autograd
from model import VGG
from types import MethodType
import onnxruntime
import numpy as np
from utils.params import arg
import os

class DebugOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, name):
        return g.op(
            'my::DebugOp',
            x,
            name_s=name
        )
    @staticmethod
    def forward(ctx, x, name):
        return x

class Debugger():
    def __init__(self):
        self.debugger_apply = DebugOp.apply
        self.torch_value = {}
        self.onnx_value = {}
        self.output_debug_name = []

    def debug(self, x, name):
        self.torch_value[name] = x.detach().cpu().numpy()
        return self.debugger_apply(x, name)

    def extract_debug_model(self, input_model_path, output_model_path):
        model = onnx.load(input_model_path)
        inputs = [input.name for input in model.graph.input]
        outputs = []
        delete_nodes = []

        for node in model.graph.node:
            if node.op_type == 'DebugOp':
                debug_name = node.attribute[0].s.decode('ASCII')
                self.output_debug_name.append(debug_name)

                output_name = node.output[0]
                outputs.append(output_name)

                node.op_type = 'Identity'   #替换为恒等映射算子，原始算子推理引擎无实现
                node.domain = ''            #域清空，代表库算子
                del node.attribute[:]       # 按照onnx格式要求清空attribute
                delete_nodes.append(node)
        e = onnx.utils.Extractor(model)
        extracted = e.extract_model(inputs, outputs)

        onnx.save(extracted, output_model_path)

    def run_debug_model(self, input, debug_model):
        sess = onnxruntime.InferenceSession(
            debug_model,
            # providers=['CPUExecutionProvider']
        )
        outputs = sess.run(None, input)

        for name, value in zip(self.output_debug_name, outputs):
            self.onnx_value[name] = value

    def print_debug_result(self):
        for name in self.torch_value.keys():
            if name in self.onnx_value:
                mse = np.mean( np.sqrt((self.torch_value[name]-self.onnx_value[name])**2) )
                print(f'{name}--MSE: {mse}')



debugger = Debugger()


def new_forward(self, x):
    x = self.stage1(x)
    x = debugger.debug(x, 'x_0')

    x = self.stage2(x)
    x = debugger.debug(x, 'x_1')

    x = self.stage3(x)
    x = debugger.debug(x, 'x_2')

    x = self.stage4(x)
    x = debugger.debug(x, 'x_3')

    x = self.pooling(x)
    x = debugger.debug(x, 'x_4')

    x = x.view(x.shape[0], -1)
    x = debugger.debug(x, 'x_5')

    x = self.fc1(x)
    x = self.out(x)

    return x

torch_model_path = os.path.join(arg.export_dir, 'model.pth')
before_debug_model_path = os.path.join(arg.export_dir, 'before_debug_model.onnx')
after_debug_model_path = os.path.join(arg.export_dir, 'after_debug_model.onnx')


torch_model = VGG()
torch_model.load_state_dict(torch.load(torch_model_path))
torch_model.eval()  # 必须处于eval模式，因为有bn层


# 利用当前封装好的new_forward方法替换之前的forward方法
torch_model.forward = MethodType(new_forward, torch_model)

dummy_input = torch.randn(1, 3, arg.img_h, arg.img_w)
torch.onnx.export(
    torch_model,
    dummy_input,
    before_debug_model_path,
    input_names=['input'],
    output_names=['output']
)

# 提取debug节点输出为new onnx模型
debugger.extract_debug_model(before_debug_model_path, after_debug_model_path)

input = torch.randn(1, 3, arg.img_h,arg.img_h)

debugger.run_debug_model({'input': input.numpy()}, after_debug_model_path)
torch_model(input)
# 比较debug节点输出的mse
debugger.print_debug_result()

