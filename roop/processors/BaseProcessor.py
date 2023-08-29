from roop.typing import Face, Frame

class BaseProcessor():

    def Initialize(self):
        pass

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame):
        pass

    def Unload(self):
        pass
        
    def Release(self):
        pass