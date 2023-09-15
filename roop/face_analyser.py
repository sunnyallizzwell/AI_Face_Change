import threading
from typing import Any
import insightface

import roop.globals
from roop.typing import Frame
import cv2
from PIL import Image
from roop.capturer import get_video_frame

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        faces = get_face_analyser().get(frame)
        return sorted(faces, key = lambda x : x.bbox[0])
    except IndexError:
        return None

def extract_face_images(source_filename, video_info,face_box_extra_ratio=0):
    face_data = []
    source_image = None
    
    if video_info[0]:
        frame = get_video_frame(source_filename, video_info[1])
        if frame is not None:
            source_image = frame
        else:
            return face_data
    else:
        source_image = cv2.imread(source_filename)


    faces = get_many_faces(source_image)

    dimensions = source_image.shape
    hh = dimensions[0]
    ww = dimensions[1]
    #channels = dimensions[2]

    i = 0
    for face in faces:
        (startX, startY, endX, endY) = face['bbox'].astype("int")
        if face_box_extra_ratio!=0:
            w1=endX-startX
            h1=endY-startY
            startX=int(startX - w1*face_box_extra_ratio);
            endX  =int(endX   + w1*face_box_extra_ratio);
            startY=int(startY - h1*face_box_extra_ratio);
            endY  =int(endY   + h1*face_box_extra_ratio);
            if startX<0: startX=0
            if endX>ww: endX=ww
            if startY<0: startY=0
            if endY>hh: endY=hh

        face_temp = source_image[startY:endY, startX:endX]
        if face_temp.size < 1:
            continue
        i += 1
        face_data.append([face, face_temp])
    return face_data