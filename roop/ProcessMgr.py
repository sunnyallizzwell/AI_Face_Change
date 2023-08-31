import os
import cv2 
import numpy as np
import psutil
from math import floor, ceil
from skimage import transform as trans

from roop.ProcessEntry import ProcessEntry
from roop.processors.FaceSwap import FaceSwapInsightFace
from roop.processors.Enhance_GFPGAN import Enhance_GFPGAN
from roop.processors.Enhance_Codeformer import Enhance_CodeFormer
from roop.processors.Enhance_DMDNet import Enhance_DMDNet
from roop.ProcessOptions import ProcessOptions

from roop.face_util import get_first_face, get_all_faces
from roop.utilities import compute_cosine_distance

from typing import Any, List, Callable
from roop.typing import Frame
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from threading import Thread, Lock
from queue import Queue, PriorityQueue
from tqdm import tqdm
from chain_img_processor.ffmpeg_writer import FFMPEG_VideoWriter # ffmpeg install needed
import roop.globals


class ProcessMgr():
    input_face_datas = []
    target_face_datas = []

    processors = []
    options : ProcessOptions = None
    
    num_threads = 1
    current_index = 0
    processing_threads = 1
    buffer_wait_time = 0.1
    loadbuffersize = 0 

    lock = Lock()

    frames_queue = None
    processed_queue = None


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
                    elif pn == "dmdnet":
                        p = Enhance_DMDNet()
                        p.Initialize()
                    if p is not None:
                        self.processors.insert(i, p)
                    

    def read_frames_thread(self, cap, frame_start, frame_end, num_threads):
        num_frame = 0
        total_num = frame_end - frame_start
        if frame_start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_start)

        while True and roop.globals.processing:
            ret, frame = cap.read()
            if not ret:
                for _ in range(num_threads):
                    self.frames_queue.put(None)
                break
                
            num_frame += 1
            self.frames_queue.put((num_frame - 1,frame), block=True)
            if num_frame == total_num:
                break

        for _ in range(num_threads):
            self.frames_queue.put(None)



    def process_frames(self, progress) -> None:
        while True:
            frametuple = self.frames_queue.get()
            if frametuple is None:
                self.processing_threads -= 1
                self.processed_queue.put(None)
                return
            else:
                index,frame = frametuple
                resimg = self.process_frame(frame)
                self.processed_queue.put((index,resimg))
                progress()


    def write_frames_thread(self, target_video, width, height, fps, total):
        with FFMPEG_VideoWriter(target_video, (width, height), fps, codec=roop.globals.video_encoder, crf=roop.globals.video_quality, audiofile=None) as output_video_ff:
            nextindex = 0
            num_producers = self.num_threads
            order_buffer = PriorityQueue()
            
            while True and roop.globals.processing:
                while not order_buffer.empty():
                    frametuple = order_buffer.get_nowait()
                    index, frame = frametuple
                    if index == nextindex:
                        output_video_ff.write_frame(frame)
                        del frame
                        del frametuple
                        nextindex += 1
                    else:
                        order_buffer.put(frametuple)
                        break

                frametuple = self.processed_queue.get()
                if frametuple is not None:
                    index, frame = frametuple
                    if index != nextindex:
                        order_buffer.put(frametuple)
                    else:
                        output_video_ff.write_frame(frame)
                        nextindex += 1
                else:
                    num_producers -= 1
                    if num_producers < 1:
                        return
            


    def run_batch_chain(self, source_video, target_video, frame_start, frame_end, fps, threads:int = 1, buffersize=32):
        cap = cv2.VideoCapture(source_video)
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = (frame_end - frame_start) + 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        total = frame_count
        self.num_threads = threads

        if buffersize < 1:
            buffersize = 1
        self.loadbuffersize = buffersize
        self.processing_threads = self.num_threads
        self.frames_queue = Queue(buffersize * self.num_threads)
        self.processed_queue = Queue()

        readthread = Thread(target=self.read_frames_thread, args=(cap, frame_start, frame_end, threads))
        readthread.start()

        writethread = Thread(target=self.write_frames_thread, args=(target_video, width, height, fps, total))
        writethread.start()

        with tqdm(total=total, desc='Processing', unit='frames', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
            with ThreadPoolExecutor(thread_name_prefix='swap_proc', max_workers=self.num_threads) as executor:
                futures = []
                
                for threadindex in range(threads):
                    future = executor.submit(self.process_frames, lambda: self.update_progress(progress))
                    futures.append(future)
                
                for future in as_completed(futures):
                    future.result()
        # wait for the task to complete
        readthread.join()
        writethread.join()
        cap.release()



    def update_progress(self, progress: Any = None) -> None:
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
        msg = 'memory_usage: ' + '{:.2f}'.format(memory_usage).zfill(5) + f' GB execution_threads {self.num_threads}'
        progress.set_postfix({
            'memory_usage': '{:.2f}'.format(memory_usage).zfill(5) + 'GB',
            'execution_threads': self.num_threads
        })
        progress.update(1)


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
            
            if self.options.swap_mode == "all":
                for face in faces:
                    temp_frame = self.process_face(self.options.selected_index, face, temp_frame, self.options.mask_top)
                    del face
                return temp_frame
            
            elif self.options.swap_mode == "selected":
                for i,tf in enumerate(self.target_face_datas):
                    for face in faces:
                        if compute_cosine_distance(tf.embedding, face.embedding) <= self.options.face_distance_threshold:
                            if i < len(self.input_face_datas):
                                temp_frame = self.process_face(i, face, temp_frame, self.options.mask_top)
                            break
                        del face

            elif self.options.swap_mode == "all_female" or self.options.swap_mode == "all_male":
                gender = 'F' if self.swap_mode == "all_female" else 'M'
                for face in faces:
                    if face.sex == gender:
                        temp_frame = self.process_face(self.options.selected_index, face, temp_frame,self.options.mask_top)
                    del face

        return temp_frame


    def process_face(self,face_index, target_face, frame, mask_top=0):
        for p in self.processors:
            if p.type == 'swap':
                fake_frame, M = p.Run(self.input_face_datas[face_index], target_face, frame)
                scale_factor = 0.0
            else:
                if p.processorname == 'dmdnet':
                    fake_frame, scale_factor = p.Run(self.input_face_datas[face_index], target_face, fake_frame, M)
                else:
                    fake_frame, scale_factor = p.Run(self.input_face_datas[face_index], target_face, fake_frame)

        if scale_factor == 0.0:
            upscale = 512
            scale_factor = int(upscale / fake_frame.shape[1])
            fake_frame = cv2.resize(fake_frame, (upscale, upscale), cv2.INTER_CUBIC)

        return self.paste_upscale(fake_frame, M, frame, scale_factor, mask_top)

    # Paste back adapted from here
    # https://github.com/fAIseh00d/refacer/blob/main/refacer.py
    # which is revised insightface paste back code

    def paste_upscale(self, upsk_face, M, target_img, scale_factor, mask_top):
            M_scale = M * scale_factor
            IM = cv2.invertAffineTransform(M_scale)

            face_matte = np.full((target_img.shape[0],target_img.shape[1]), 255, dtype=np.uint8)
            ##Generate white square sized as a upsk_face
            img_matte = np.full((upsk_face.shape[0],upsk_face.shape[1]), 255, dtype=np.uint8)
            if mask_top > 0:
                img_matte[:mask_top,:] = 0
 
            ##Transform white square back to target_img
            img_matte = cv2.warpAffine(img_matte, IM, (target_img.shape[1], target_img.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0.0) 
            ##Blacken the edges of face_matte by 1 pixels (so the mask in not expanded on the image edges)
            img_matte[:1,:] = img_matte[-1:,:] = img_matte[:,:1] = img_matte[:,-1:] = 0

            #Detect the affine transformed white area
            mask_h_inds, mask_w_inds = np.where(img_matte==255) 
            #Calculate the size (and diagonal size) of transformed white area width and height boundaries
            mask_h = np.max(mask_h_inds) - np.min(mask_h_inds) 
            mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
            mask_size = int(np.sqrt(mask_h*mask_w))
            #Calculate the kernel size for eroding img_matte by kernel (insightface empirical guess for best size was max(mask_size//10,10))
            # k = max(mask_size//12, 8)
            k = max(mask_size//10, 10)
            kernel = np.ones((k,k),np.uint8)
            img_matte = cv2.erode(img_matte,kernel,iterations = 1)
            #Calculate the kernel size for blurring img_matte by blur_size (insightface empirical guess for best size was max(mask_size//20, 5))
            # k = max(mask_size//24, 4) 
            k = max(mask_size//20, 5) 
            kernel_size = (k, k)
            blur_size = tuple(2*i+1 for i in kernel_size)
            img_matte = cv2.GaussianBlur(img_matte, blur_size, 0)
            
            #Normalize images to float values and reshape
            img_matte = img_matte.astype(np.float32)/255
            face_matte = face_matte.astype(np.float32)/255
            img_matte = np.minimum(face_matte, img_matte)
            img_matte = np.reshape(img_matte, [img_matte.shape[0],img_matte.shape[1],1]) 
            ##Transform upcaled face back to target_img
            paste_face = cv2.warpAffine(upsk_face, IM, (target_img.shape[1], target_img.shape[0]), borderMode=cv2.BORDER_REPLICATE) 
            ##Re-assemble image
            paste_face = img_matte * paste_face
            paste_face = paste_face + (1-img_matte) * target_img.astype(np.float32) 
            return paste_face.astype(np.uint8)



        # temp_frame = cv2.resize(temp_frame, (512, 512))

        # ratio = 4.0
        # diff_x = 8.0*ratio
        # dst = self.insight_arcface_dst * ratio
        # dst[:,0] += diff_x
        # tform = trans.SimilarityTransform()
        # tform.estimate(target_face.kps, dst)
        # M1 = tform.params[0:2, :]
        # IM = cv2.invertAffineTransform(M1)

        # img_mask = np.full((temp_frame.shape[0],temp_frame.shape[1]), 0, dtype=np.float32)
        # mask_border = 5
        # # img_mask = cv2.rectangle(img_mask, (mask_border+int(self.mask_left), mask_border+int(self.mask_top)), 
        # #                         (512 - mask_border-int(self.mask_right), 512-mask_border-int(self.mask_bottom)), (255, 255, 255), -1)    
        # img_mask = cv2.rectangle(img_mask, (mask_border, mask_border+int(mask_top)), 
        #                         (512 - mask_border, 512-mask_border), (255, 255, 255), -1)    
        # img_mask = cv2.GaussianBlur(img_mask, (self.options.mask_blur_amount*2+1,self.options.mask_blur_amount*2+1), 0)    
        # img_mask /= 255


        # img_mask_0 = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
        # img_mask = cv2.warpAffine(img_mask, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)

        # fake_merged = img_mask_0* temp_frame
        
        # img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])    
        # fake_merged = cv2.warpAffine(fake_merged, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0) 
       
        # bbox = target_face.bbox

        # padding_size = 0.3
        # padding_hor = (bbox[2]-bbox[0]) * padding_size
        # padding_vert = (bbox[3]-bbox[1]) * padding_size
        
        # left = floor(bbox[0]-padding_hor)
        # top = floor(bbox[1]-padding_vert)
        # right = ceil(bbox[2]+padding_hor)
        # bottom = ceil(bbox[3]+padding_vert)
        # if left<0:
        #     left=0
        # if top<0: 
        #     top=0
        # if right>target_img.shape[1]:
        #     right=target_img.shape[1]
        # if bottom>target_img.shape[0]:
        #     bottom=target_img.shape[0]
        
        # fake_merged = fake_merged[top:bottom, left:right, 0:3]
        # target_img_a = target_img[top:bottom, left:right, 0:3]
        # img_mask = img_mask[top:bottom, left:right, 0:1]

        # fake_merged = fake_merged + (1-img_mask) * target_img_a.astype(np.float32)
        

        # smallest = min(fake_merged.shape[0], fake_merged.shape[1])
        # mask_border = smallest // 12
        # if mask_border > 4:
        #     img_white = np.full((fake_merged.shape[0], fake_merged.shape[1]), 0, dtype=float)
        #     img_white = cv2.rectangle(img_white, (mask_border, mask_border), 
        #                             (img_white.shape[1] - mask_border, img_white.shape[0]-mask_border), (255, 255, 255), -1)    
        #     img_mask = img_white
        #     t1 = mask_border * 2
        #     kernel = np.ones((t1, t1), np.uint8)
        #     img_mask = cv2.erode(img_mask, kernel, iterations=2)
        #     t1 = mask_border
        #     kernel_size = (t1, t1)
        #     blur_size = tuple(2 * j + 1 for j in kernel_size)
        #     img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        #     img_mask /= 255
        #     img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
        #     fake_merged = img_mask * fake_merged + (1 - img_mask) * target_img_a
        
        # target_img[top:bottom, left:right, 0:3] = fake_merged
        # return target_img.astype(np.uint8)    #BGR


    def unload_models():
        pass

    def release_resources():
        pass
