import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically manages CUDA context initialization


class TensorRTEngine:
    """
    TensorRT Engine wrapper for loading ONNX-converted .engine files
    and running high-speed hardware-accelerated inference on Jetson Orin Nano.
    """

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found at: {engine_path}")

        # 1. Load serialized TensorRT engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # 2. Allocate CUDA host and device buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host (CPU) and device (GPU CUDA) memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Runs async inference on input numpy array and returns host predictions.
        
        Args:
            input_data (np.ndarray): Preprocessed image tensor (e.g., shape [1, 3, 64, 64])
            
        Returns:
            np.ndarray: Model output logits / class probabilities
        """
        # Ensure input data is contiguous and flattened to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())

        # Transfer input memory to GPU asynchronously
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Execute TensorRT Inference kernel
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Transfer output predictions back to CPU asynchronously
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)

        # Synchronize CUDA stream
        self.stream.synchronize()

        # Reshape output predictions to match engine output dimensions
        output_shape = self.engine.get_binding_shape(1)
        return self.outputs[0]['host'].reshape(output_shape)


if __name__ == "__main__":
    print("TensorRTEngine helper module ready.")