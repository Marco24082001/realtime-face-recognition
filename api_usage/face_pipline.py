"""
@author: JiXuan Xu, Jun Wang
@date: 20201024
@contact: jun21wangustc@gmail.com 
"""
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

def compare_faces(feature, featureListKnowed, threadshold):
    scores = [lambda:cosine(feature,ft) < threadshold for ft in featureListKnowed]
    return scores

def face_distance(feature, featureListKnowed):
    distances = [cosine(feature, ft) for ft in featureListKnowed]
    return distances
if __name__ == '__main__':
    # common setting for all models, need not modify.
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

    data = np.load('api_usage/encoding.npz')
    classNames, featureListKnown = data['arr_0'], data['arr_1']
    cap = cv2.VideoCapture(0)
    count = 0
    stop = False
    while stop != True:
        success, img = cap.read()
        image = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            dets = faceDetModelHandler.inference_on_image(image)
            bboxs = dets
            face_nums = dets.shape[0]
            feature_list = []
            for i in range(face_nums):
                landmarks = faceAlignModelHandler.inference_on_image(image, dets[i])
                landmarks_list = []
                for (x, y) in landmarks.astype(np.int32):
                    landmarks_list.extend((x, y))
                cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
                feature = faceRecModelHandler.inference_on_image(cropped_image)
                feature_list.append(feature)
            print(len(feature_list))
            if len(feature_list) != 0:
                similarity = compare_faces(feature_list[0], featureListKnown, 0.5)
                distances = face_distance(feature_list[0], featureListKnown)
                matchIndex = np.argmin(distances)
                # print(classNames)
                # print(matchIndex)
                if similarity[matchIndex]:
                    name = classNames[matchIndex].upper()
                    print(name)
                    for box in bboxs:
                        box = np.array(box).astype(int)
                        box *=4
                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                        cv2.putText(img, name, (box[0] + 6, box[3] - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        # y1, x2, y2, x1 = faceLoc
        #             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #             cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        #             cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                pass
        except Exception as e:
            logger.error('Pipeline failed!')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        # facesCurFrame = face_recognition.face_locations(imgS)
        # encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        #
        # if (facesCurFrame != [] and encodesCurFrame != []):
        #     for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        #         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        #         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #         matchIndex = np.argmin(faceDis)
        #         if matches[matchIndex]:
        #             name = classNames[matchIndex].upper()
        #             y1, x2, y2, x1 = faceLoc
        #             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #             cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        #             cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        #     count += 1
        #     if count != 0:
        #         stop = True
        cv2.imshow('Webcam', img)
        if (cv2.waitKey(1) == ord('q')):
            break
        # markAttendance(name)

    # read image and get face features.
    # image_path = 'api_usage/test_images/test1.jpg'
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # try:
    #     dets = faceDetModelHandler.inference_on_image(image)
    #     face_nums = dets.shape[0]
    #     if face_nums != 2:
    #         logger.info('Input image should contain two faces to compute similarity!')
    #     feature_list = []
    #     for i in range(face_nums):
    #         landmarks = faceAlignModelHandler.inference_on_image(image, dets[i])
    #         landmarks_list = []
    #         for (x, y) in landmarks.astype(np.int32):
    #             landmarks_list.extend((x, y))
    #         cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
    #         feature = faceRecModelHandler.inference_on_image(cropped_image)
    #         feature_list.append(feature)
    #     # score = np.dot(feature_list[0], feature_list[1])
    #     score = cosine(feature_list[0], feature_list[1])
    #     logger.info('The similarity score of two faces: %f' % score)
    # except Exception as e:
    #     logger.error('Pipeline failed!')
    #     logger.error(e)
    #     sys.exit(-1)
    # else:
    #     logger.info('Success!')