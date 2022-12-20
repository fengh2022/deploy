from typing import Union, Optional, Sequence, Dict, Any

import torch
from torch.utils.data import DataLoader
import tensorrt as trt
from time import time
from utils.params import arg
import os
from glob import glob
from dataset import CatDogDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class TRTWrapper(torch.nn.Module):
    def __init__(self, engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [_ for _ in self.engine]

        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name)
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

            # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return outputs


if __name__=='__main__':
    engine_model = os.path.join(arg.export_dir, 'model.engine')
    model = TRTWrapper(engine_model, ['output'])

    evaldata = CatDogDataset(rootdir=arg.root_dir, is_training=False)
    evalloader = DataLoader(evaldata, batch_size=1, num_workers=arg.num_workers, shuffle=True)

    y_true = []
    y_pred = []

    time_total = 0.0
    valid_cnt = 0

    for img, lab in tqdm(evalloader):
        img = img.float().cuda()
        lab = lab[0].item()

        tap = time()

        torch.cuda.synchronize()  # 同步
        output = model(dict(input=img))['output']
        torch.cuda.synchronize()

        gap = time()-tap

        time_total += gap

        if gap>0:
            valid_cnt += 1

        output = output[0].argmax().item()
        y_true.append(lab)
        y_pred.append(output)

    acc = accuracy_score(y_true, y_pred)


    print(f'acc: {acc}, fps: {valid_cnt/time_total}')



