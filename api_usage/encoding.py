#  Get ClassName and Encoding Vector of Training_images

import sys
sys.path.append('.')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')
from scipy.spatial.distance import cosine

import yaml
import cv2
import numpy as np
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler
import os
with open('config/model_conf.yaml') as f:
    model_conf = yaml.full_load(f)

def featureKnow():
    model_path = 'models'

    # face detection model setting.
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face detection model...')
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        logger.error('Falied to load face detection Model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # face landmark model setting.
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face landmark model...')
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        logger.error('Failed to load face landmark model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # face recognition model setting.
    model_category = 'face_recognition'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face recognition model...')
    try:
        faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
        model, cfg = faceRecModelLoader.load_model()
        faceRecModelHandler = FaceRecModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        logger.error('Failed to load face recognition model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')
    face_cropper = FaceRecImageCropper()

    path = 'api_usage/Training_images'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}', cv2.IMREAD_COLOR)
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    # print(classNames)
    featureListKnown = []
    print(len(images))
    for image in images:
        try:
            dets = faceDetModelHandler.inference_on_image(image)
            face_nums = dets.shape[0]
            if face_nums != 1:
                logger.info('Input image should contain one faces to compute similarity!')
            landmarks = faceAlignModelHandler.inference_on_image(image, dets[0])
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))
            cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
            feature = faceRecModelHandler.inference_on_image(cropped_image)
            featureListKnown.append(feature)
        except Exception as e:
            logger.error('Pipeline failed!')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')
    # print(featureListKnown)
    np.savez_compressed('api_usage/encoding.npz', classNames, featureListKnown)

featureKnow()