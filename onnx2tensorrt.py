import torch
import onnx
import tensorrt as trt
from utils.params import arg
import os

onnx_file_path = os.path.join(arg.export_dir, 'model.onnx')
engine_file_path = os.path.join(arg.export_dir, 'model.engine')


def build_engine(onnx_file_path, engine_file_path, arg):
    print(f'{onnx_file_path}----->{engine_file_path}, quantify_mode: {arg.quantify_mode}')
    device = torch.device(arg.device)
    onnx_model = onnx.load(onnx_file_path)
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 20
    profile = builder.create_optimization_profile()
    if arg.quantify_mode == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif arg.quantify_mode == 'int8':
        from calibration.calibrator import CatDogCalibrator
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = CatDogCalibrator()
        config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
        config.int8_calibrator = calibrator

    profile.set_shape('input', [1, 3, arg.img_h, arg.img_w], [1, 3, arg.img_h, arg.img_w], [1, 3, arg.img_h, arg.img_w])
    config.add_optimization_profile(profile)
    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    with open(engine_file_path, mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!", engine_file_path)


if __name__=='__main__':
    onnx_file_path = os.path.join(arg.export_dir, 'model.onnx')
    engine_file_path = os.path.join(arg.export_dir, 'model.engine')

    build_engine(onnx_file_path, engine_file_path, arg)