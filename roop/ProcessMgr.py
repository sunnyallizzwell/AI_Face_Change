import cv2 
import numpy as np
from math import floor, ceil
from skimage import transform as trans

from roop.ProcessEntry import ProcessEntry
from roop.processors.FaceSwap import FaceSwapInsightFace
from roop.processors.Enhance_GFPGAN import Enhance_GFPGAN
from roop.processors.Enhance_Codeformer import Enhance_CodeFormer
from roop.ProcessOptions import ProcessOptions

from roop.face_util import get_first_face, get_all_faces
from roop.utilities import compute_cosine_distance


class ProcessMgr():
    input_face_datas = []
    target_face_datas = []

    processors = []
    options : ProcessOptions = None

    # 5-point template constant for face alignment - don't ask
    insight_arcface_dst = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32)  


    def __init__(self):
        pass

    def initialize(self, input_faces, target_faces, options):
        self.input_face_datas = input_faces
        self.target_face_datas = target_faces
        self.options = options

        processornames = options.processors.split(",")
        prolist = []

        if len(self.processors) < 1:
            for pn in processornames:
                if pn == "faceswap":
                    p = FaceSwapInsightFace()
                    p.Initialize()
                    self.processors.append(p)
                elif pn == "gfpgan":
                    p = Enhance_GFPGAN()
                    p.Initialize()
                    self.processors.append(p)
                elif pn == "codeformer":
                    p = Enhance_CodeFormer()
                    p.Initialize()
                    self.processors.append(p)
        else:
            for i in range(len(self.processors) -1, -1, -1):
                if not self.processors[i].processorname in processornames:
                    self.processors[i].Release()
                    del self.processors[i]

            for i,pn in enumerate(processornames):
                if i >= len(self.processors) or self.processors[i].processorname != pn:
                    p = None
                    if pn == "faceswap":
                        p = FaceSwapInsightFace()
                        p.Initialize()
                    elif pn == "gfpgan":
                        p = Enhance_GFPGAN()
                        p.Initialize()
                    elif pn == "codeformer":
                        p = Enhance_CodeFormer()
                        p.Initialize()
                    if p is not None:
                        self.processors.insert(i, p)
                    


                    








    def process(entry: ProcessEntry=None):
        pass

    def process_frame(self, frame):
        if len(self.input_face_datas) < 1:
            return frame
    
        temp_frame = frame.copy()

        if self.options.swap_mode == "first":
            face = get_first_face(frame)
            if face is None:
                return frame
            return self.process_face(self.options.selected_index, face, temp_frame, self.options.mask_top)

        else:
            faces = get_all_faces(frame)
            if faces is None:
                return frame
            
            if self.swap_mode == "all":
                for face in faces:
                    temp_frame = self.process_face(self.options.selected_index, face, temp_frame, self.options.mask_top)
                return temp_frame
            
            elif self.swap_mode == "selected":
                for i,tf in enumerate(self.target_face_datas):
                    for face in faces:
                        if compute_cosine_distance(tf.embedding, face.embedding) <= self.options.face_distance_threshold:
                            if i < len(self.input_face_datas):
                                temp_frame = self.process_face(i, face, temp_frame)
                            break

            elif self.swap_mode == "all_female" or self.swap_mode == "all_male":
                gender = 'F' if self.swap_mode == "all_female" else 'M'
                for face in faces:
                    if face.sex == gender:
                        temp_frame = self.process_face(self.options.selected_index, face, temp_frame,self.options.mask_top)

        return temp_frame


    def process_face(self,face_index, target_face, frame, mask_top=0):
        target_img = frame
        temp_frame = frame.copy()
        for p in self.processors:
            temp_frame = p.Run(self.input_face_datas[face_index], target_face, temp_frame)

        temp_frame = cv2.resize(temp_frame, (512, 512))

        ratio = 4.0
        diff_x = 8.0*ratio
        dst = self.insight_arcface_dst * ratio
        dst[:,0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(target_face.kps, dst)
        M1 = tform.params[0:2, :]
        IM = cv2.invertAffineTransform(M1)

        img_mask = np.full((temp_frame.shape[0],temp_frame.shape[1]), 0, dtype=np.float32)
        mask_border = 5
        # img_mask = cv2.rectangle(img_mask, (mask_border+int(self.mask_left), mask_border+int(self.mask_top)), 
        #                         (512 - mask_border-int(self.mask_right), 512-mask_border-int(self.mask_bottom)), (255, 255, 255), -1)    
        img_mask = cv2.rectangle(img_mask, (mask_border, mask_border+int(mask_top)), 
                                (512 - mask_border, 512-mask_border), (255, 255, 255), -1)    
        img_mask = cv2.GaussianBlur(img_mask, (self.options.mask_blur_amount*2+1,self.options.mask_blur_amount*2+1), 0)    
        img_mask /= 255


        img_mask_0 = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
        img_mask = cv2.warpAffine(img_mask, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)

        fake_merged = img_mask_0* temp_frame
        
        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])    
        fake_merged = cv2.warpAffine(fake_merged, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0) 
       
        bbox = target_face.bbox

        padding_size = 0.3
        padding_hor = (bbox[2]-bbox[0]) * padding_size
        padding_vert = (bbox[3]-bbox[1]) * padding_size
        
        left = floor(bbox[0]-padding_hor)
        top = floor(bbox[1]-padding_vert)
        right = ceil(bbox[2]+padding_hor)
        bottom = ceil(bbox[3]+padding_vert)
        if left<0:
            left=0
        if top<0: 
            top=0
        if right>target_img.shape[1]:
            right=target_img.shape[1]
        if bottom>target_img.shape[0]:
            bottom=target_img.shape[0]
        
        fake_merged = fake_merged[top:bottom, left:right, 0:3]
        target_img_a = target_img[top:bottom, left:right, 0:3]
        img_mask = img_mask[top:bottom, left:right, 0:1]

        fake_merged = fake_merged + (1-img_mask) * target_img_a.astype(np.float32)
        
        target_img[top:bottom, left:right, 0:3] = fake_merged
        
        fake_merged = target_img.astype(np.uint8)   
        return fake_merged    #BGR


    def unload_models():
        pass

    def release_resources():
        pass
