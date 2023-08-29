from typing import Any, List, Callable
import cv2 
import threading
import numpy as np
import onnxruntime
import onnx
import roop.globals

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path


# THREAD_LOCK = threading.Lock()


class Enhance_CodeFormer():

    model_codeformer = None

    processorname = 'codeformer'

    def Initialize(self):
        if self.model_codeformer is None:
            model_path = resolve_relative_path('../models/CodeFormer/CodeFormerv0.1.onnx')
            self.model_codeformer = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)


    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        input_shape = temp_frame.shape
        # preprocess
        temp_frame = cv2.resize(temp_frame, (512, 512))
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)


        model_inputs = self.model_codeformer.get_inputs()
        model_outputs = self.model_codeformer.get_outputs()
        io_binding = self.model_codeformer.io_binding()           
        io_binding.bind_cpu_input(model_inputs[0].name, temp_frame.astype(np.float32))
        io_binding.bind_cpu_input(model_inputs[1].name, np.array([0.5]))
        io_binding.bind_output(model_outputs[0].name, "cpu")
        self.model_codeformer.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]

        # post-process
        result = result.transpose((1, 2, 0))

        un_min = -1.0
        un_max = 1.0
        result = np.clip(result, un_min, un_max)
        result = (result - un_min) / (un_max - un_min)

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result = (result * 255.0).round()
        # result = cv2.resize(result, (input_shape[1], input_shape[0]))
        
        return result.astype(np.uint8)


    def Release(self):
        self.model_codeformer = None









