import numpy as np
import cv2
from jaa import JaaCore
from roop.utilities import get_device


from typing import Any

version = "4.0.0"

class ChainImgProcessor(JaaCore):

    def __init__(self):
        JaaCore.__init__(self)

        self.processors:dict = {
        }

        self.processors_objects:dict[str,list[ChainImgPlugin]] = {}

        self.default_chain = ""
        self.init_on_start = ""

        self.inited_processors = []

        self.is_demo_row_render = False

    def process_plugin_manifest(self, modname, manifest):
        # adding processors from plugin manifest
        if "img_processor" in manifest:  # process commands
            for cmd in manifest["img_processor"].keys():
                self.processors[cmd] = manifest["img_processor"][cmd]

        return manifest

    def init_with_plugins(self):
        self.init_plugins(["core"])
        self.display_init_info()

        #self.init_translator_engine(self.default_translator)
        init_on_start_arr = self.init_on_start.split(",")
        for proc_id in init_on_start_arr:
            self.init_processor(proc_id)

    def run_chain(self, img, params:dict[str,Any] = None, chain:str = None, thread_index:int = 0):
        if chain is None:
            chain = self.default_chain
        if params is None:
            params = {}
        params["_thread_index"] = thread_index
        chain_ar = chain.split(",")
        # init all not inited processors first
        for proc_id in chain_ar:
            if proc_id != "":
                if not proc_id in self.inited_processors:
                    self.init_processor(proc_id)



        # run processing
        if self.is_demo_row_render:
            import cv2
            import numpy as np
            height, width, channels = img.shape
            img_blank = np.zeros((height+30, width*(1+len(chain_ar)), 3), dtype=np.uint8)
            img_blank.fill(255)

            y = 30
            x = 0
            img_blank[y:y + height, x:x + width] = img

            # Set the font scale and thickness
            font_scale = 1
            thickness = 2

            # Set the font face to a monospace font
            font_face = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(img_blank, "original", (x+4, y-7), font_face, font_scale, (0, 0, 0), thickness)


        i = 0
        for proc_id in chain_ar:
            i += 1
            if proc_id != "":
                #img = self.processors[proc_id][1](self, img, params) # params can be modified inside
                y = 30
                img = self.processors_objects[proc_id][thread_index].process(img,params)
                if self.is_demo_row_render:
                    x = width*i
                    img_blank[y:y + height, x:x + width] = img
                    cv2.putText(img_blank, proc_id, (x + 4, y - 7), font_face, font_scale, (0, 0, 0), thickness)

        if self.is_demo_row_render:
            return img_blank, params

        return img, params

    # ---------------- init translation stuff ----------------
    def fill_processors_for_thread_chains(self, threads:int = 1, chain:str = None):
        if chain is None:
            chain = self.default_chain

        chain_ar = chain.split(",")
        # init all not initialized processors first
        for processor_id in chain_ar:
            if processor_id != "":
                if self.processors_objects.get(processor_id) is None:
                    self.processors_objects[processor_id] = []
                while len(self.processors_objects[processor_id]) < threads:
                    self.add_processor_to_list(processor_id)

    def add_processor_to_list(self, processor_id: str):
        obj = self.processors[processor_id](self)
        obj.init_plugin()
        if self.processors_objects.get(processor_id) is None:
            self.processors_objects[processor_id] = []
        self.processors_objects[processor_id].append(obj)
    def init_processor(self, processor_id: str):
        if processor_id == "": # blank line case
            return

        if processor_id in self.inited_processors:
            return

        try:
            if self.verbose:
                self.print_blue("TRY: init processor plugin '{0}'...".format(processor_id))
            self.add_processor_to_list(processor_id)
            self.inited_processors.append(processor_id)
            if self.verbose:
                self.print_blue("SUCCESS: '{0}' initialized!".format(processor_id))

        except Exception as e:
            self.print_error("Error init processor plugin {0}...".format(processor_id), e)

    # ------------ formatting stuff -------------------
    def display_init_info(self):
        if self.verbose:
            print("ChainImgProcessor v{0}:".format(version))
            self.format_print_key_list("processors:", self.processors.keys())

    def format_print_key_list(self, key:str, value:list):
        print(key+": ".join(value))

    def print_error(self,err_txt,e:Exception = None):
        print(err_txt,"red")
        # if e != None:
        #     cprint(e,"red")
        import traceback
        traceback.print_exc()

    def print_red(self,txt):
        print(txt)

    def print_blue(self, txt):
        print(txt)

class ChainImgPlugin:

    device = 'cpu'

    def __init__(self, core: ChainImgProcessor):
        self.core = core
        self.device = get_device()

    def init_plugin(self): # here you can init something. Called once
        pass
    def process(self, img, params:dict): # process img. Called multiple
        return img
    
    def unload(self):
        pass


    def cutout(self, frame, start_x, start_y, end_x, end_y, padding_factor):
        padding_x = int((end_x - start_x) * padding_factor)
        padding_y = int((end_y - start_y) * padding_factor)

        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = min(frame.shape[1], end_x + padding_x)
        end_y = min(frame.shape[0], end_y + padding_y)
        return frame[start_y:end_y, start_x:end_x], start_x, start_y, end_x, end_y
    
    def paste_into(self, clip, frame, start_x, start_y, end_x, end_y, smooth):
        if smooth:
            smallest = min(clip.shape[0], clip.shape[1])
            mask_border = smallest // 12
            if mask_border > 4:
                img_white = np.full((clip.shape[0], clip.shape[1]), 0, dtype=float)
                # img_white = cv2.warpAffine(img_white, mat_rev, img_shape)
                # img_white[img_white > 20] = 255
                img_white = cv2.rectangle(img_white, (mask_border, mask_border), 
                                        (img_white.shape[1] - mask_border, img_white.shape[0]-mask_border), (255, 255, 255), -1)    
                img_mask = img_white
                t1 = mask_border * 2
                kernel = np.ones((t1, t1), np.uint8)
                img_mask = cv2.erode(img_mask, kernel, iterations=2)
                t1 = mask_border
                kernel_size = (t1, t1)
                blur_size = tuple(2 * j + 1 for j in kernel_size)
                img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
                img_mask /= 255
                img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
                frame_clip = frame[start_y:end_y, start_x:end_x]
                clip = img_mask * clip + (1 - img_mask) * frame_clip
                
        frame[start_y:end_y, start_x:end_x] = clip
        return frame

    


_img_processor:ChainImgProcessor = None
def get_single_image_processor() -> ChainImgProcessor:
    global _img_processor
    if _img_processor is None:
        _img_processor = ChainImgProcessor()
        _img_processor.init_with_plugins()
    return _img_processor

