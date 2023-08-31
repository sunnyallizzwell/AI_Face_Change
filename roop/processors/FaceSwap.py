from typing import Any, List, Callable
import cv2 
import threading
import numpy as np
import onnxruntime
import onnx
import roop.globals

from insightface.utils.face_align import norm_crop2
from numpy.linalg import norm as l2norm
from roop.processors.frame.face_swapper import get_face_swapper

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path

# THREAD_LOCK = threading.Lock()



class FaceSwapInsightFace():
    model_swap_insightface = None

    input_size = []                # size of the inswapper.onnx inputs
    emap = []                      # comes from loading the inswapper model. not sure of data

    processorname = 'faceswap'
    type = 'swap'


    def Initialize(self):
        if self.model_swap_insightface is not None:
            return
        
        # Load Swapper model and get graph param
        model_path = resolve_relative_path('../models/inswapper_128.onnx')
        model = onnx.load(model_path)
        graph = model.graph
        self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])

        # Create Swapper model session
        opts = onnxruntime.SessionOptions()
        # opts.enable_profiling = True 
        self.model_swap_insightface, self.emap = onnxruntime.InferenceSession( model_path, opts, providers=roop.globals.execution_providers), self.emap

        # Get in/out size and create some data
        inputs =  self.model_swap_insightface.get_inputs()
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_size = tuple(input_shape[2:4][::-1])
        

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        img_fake, M = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=False)
        return img_fake, M 
        # aimg, _ = norm_crop2(temp_frame, target_face.kps, self.input_size[0])
        # blob = cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, self.input_size, (0.0, 0.0, 0.0), swapRB=True)

        # #Select source embedding
        # n_e = source_face.normed_embedding / l2norm(source_face.normed_embedding)
        # latent = n_e.reshape((1,-1))
        # latent = np.dot(latent, self.emap)
        # latent /= np.linalg.norm(latent)

                            
        # io_binding = self.model_swap_insightface.io_binding()
        # model_inputs = self.model_swap_insightface.get_inputs()
        # model_outputs = self.model_swap_insightface.get_outputs()            
        # io_binding.bind_cpu_input(model_inputs[0].name, blob)
        # io_binding.bind_cpu_input(model_inputs[1].name, latent)
        # io_binding.bind_output(model_outputs[0].name, "cuda")
        # self.model_swap_insightface.run_with_iobinding(io_binding)
        # ort_outs = io_binding.copy_outputs_to_cpu()
        # pred = ort_outs[0]

        # img_fake = pred.transpose((0,2,3,1))[0]
        # return np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]


    def Release(self):
        self.model_swap_insightface = None


                



