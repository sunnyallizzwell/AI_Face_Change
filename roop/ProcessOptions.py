class ProcessOptions:

    def __init__(self,processors, face_distance, blur, swap_mode, selected_index, mask_top):
        self.processors = processors
        self.face_distance_threshold = face_distance
        self.mask_blur_amount = blur
        self.swap_mode = swap_mode
        self.selected_index = selected_index
        self.mask_top = mask_top
