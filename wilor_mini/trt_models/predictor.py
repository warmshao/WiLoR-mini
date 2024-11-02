import numpy as np

import torch
from torch.cuda import nvtx
from collections import OrderedDict

try:
    import tensorrt as trt
    import ctypes
except ModuleNotFoundError:
    print("No TensorRT Found")

numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool


class TensorRTPredictor:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, **kwargs):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        engine_path = kwargs.get("model_path", None)
        self.debug = kwargs.get("debug", False)
        assert engine_path, f"model:{engine_path} must exist!"
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        # self.context = self.engine.create_execution_context()
        # assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.tensors = OrderedDict()

        # TODO: 支持动态shape输入
        for idx in range(self.engine.num_io_tensors):
            name = self.engine[idx]
            is_input = self.engine.get_tensor_mode(name).name == "INPUT"
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            binding = {
                "index": idx,
                "name": name,
                "dtype": dtype,
                "shape": list(shape)
            }
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        self.allocate_max_buffers(**kwargs)
        self.activate()

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
        else:
            self.context = self.engine.create_execution_context()

    def deactivate(self):
        if self.context:
            del self.context
        self.context = None

    def allocate_max_buffers(self, device="cuda", **kwargs):
        nvtx.range_push("allocate_max_buffers")
        # 目前仅支持 batch 维度的动态处理, 如果是其他维度的动态，请传入 output_max_shapes
        output_max_shapes = kwargs.get("output_max_shapes", {})

        batch_size = 1
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            shape = self.engine.get_tensor_shape(binding)
            is_input = self.engine.get_tensor_mode(binding).name == "INPUT"
            if binding in output_max_shapes:
                shape = output_max_shapes[binding]
            else:
                if -1 in shape:
                    if is_input:
                        shape = self.engine.get_tensor_profile_shape(binding, 0)[-1]
                        batch_size = shape[0]
                    else:
                        shape[0] = batch_size
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            tensor = torch.empty(
                np.prod(list(shape)), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor
        nvtx.range_pop()

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        specs = []
        for i, o in enumerate(self.inputs):
            specs.append((o["name"], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt input {i} -> {o['name']} -> {o['shape']}")
        return specs

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for i, o in enumerate(self.outputs):
            specs.append((o["name"], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt output {i} -> {o['name']} -> {o['shape']}")
        return specs

    def adjust_buffer(self, feed_dict):
        nvtx.range_push("adjust_buffer")
        for name, buf in feed_dict.items():
            input_tensor = self.tensors[name]
            current_shape = list(buf.shape)
            if len(current_shape) == 0:
                current_shape = (1,)
            tensor_len = np.prod(current_shape)
            input_tensor[:tensor_len].copy_(buf.reshape(-1))
            self.context.set_input_shape(name, current_shape)
        nvtx.range_pop()

    def predict(self, feed_dict, stream):
        """
        Execute inference on a batch of images.
        :param data: A list of inputs as numpy arrays.
        :return A list of outputs as numpy arrays.
        """
        nvtx.range_push("set_tensors")
        self.adjust_buffer(feed_dict)
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        nvtx.range_pop()
        nvtx.range_push("execute")
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")
        nvtx.range_pop()
        return self.tensors

    def __del__(self):
        if self.engine is not None:
            del self.engine
            self.engine = None
        if self.context is not None:
            del self.context
            self.context = None
        del self.inputs
        del self.outputs
        del self.tensors


def get_predictor(**kwargs):
    predict_type = kwargs.get("predict_type", "trt")
    if predict_type == "trt":
        return TensorRTPredictor(**kwargs)
    else:
        raise NotImplementedError
