# import cv2
# import math
# import argparse

# def highlightFace(net, frame, conf_threshold=0.7):
#     frameOpencvDnn=frame.copy()
#     frameHeight=frameOpencvDnn.shape[0]
#     frameWidth=frameOpencvDnn.shape[1]
#     blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

#     net.setInput(blob)
#     detections=net.forward()
#     faceBoxes=[]
#     for i in range(detections.shape[2]):
#         confidence=detections[0,0,i,2]
#         if confidence>conf_threshold:
#             x1=int(detections[0,0,i,3]*frameWidth)
#             y1=int(detections[0,0,i,4]*frameHeight)
#             x2=int(detections[0,0,i,5]*frameWidth)
#             y2=int(detections[0,0,i,6]*frameHeight)
#             faceBoxes.append([x1,y1,x2,y2])
#             cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
#     return frameOpencvDnn,faceBoxes


# parser=argparse.ArgumentParser()
# parser.add_argument('--image')

# args=parser.parse_args()

# faceProto="opencv_face_detector.pbtxt"
# faceModel="opencv_face_detector_uint8.pb"
# ageProto="age_deploy.prototxt"
# ageModel="age_net.caffemodel"
# genderProto="gender_deploy.prototxt"
# genderModel="gender_net.caffemodel"

# MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList=['Male','Female']

# faceNet=cv2.dnn.readNet(faceModel,faceProto)
# ageNet=cv2.dnn.readNet(ageModel,ageProto)
# genderNet=cv2.dnn.readNet(genderModel,genderProto)

# video=cv2.VideoCapture(args.image if args.image else 0)
# padding=20
# while cv2.waitKey(1)<0 :
#     hasFrame,frame=video.read()
#     if not hasFrame:
#         cv2.waitKey()
#         break
    
#     resultImg,faceBoxes=highlightFace(faceNet,frame)
#     if not faceBoxes:
#         print("No face detected")

#     for faceBox in faceBoxes:
#         face=frame[max(0,faceBox[1]-padding):
#                    min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
#                    :min(faceBox[2]+padding, frame.shape[1]-1)]

#         blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
#         genderNet.setInput(blob)
#         genderPreds=genderNet.forward()
#         gender=genderList[genderPreds[0].argmax()]
#         print(f'Gender: {gender}')

#         ageNet.setInput(blob)
#         agePreds=ageNet.forward()
#         age=ageList[agePreds[0].argmax()]
#         print(f'Age: {age[1:-1]} years')

#         cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
#         cv2.imshow("Detecting age and gender", resultImg)


# File: age_gender_detect.py

import cv2
import math
import argparse

# Load the age/gender Caffe and DNN models once when the module is imported
def load_age_gend_models():
    faceProto = "../Models/computer_vision_models/age_gender/opencv_face_detector.pbtxt"
    faceModel = "../Models/computer_vision_models/age_gender/opencv_face_detector_uint8.pb"
    ageProto = "../Models/computer_vision_models/age_gender/age_deploy.prototxt"
    ageModel = "../Models/computer_vision_models/age_gender/age_net.caffemodel"
    genderProto = "../Models/computer_vision_models/age_gender/gender_deploy.prototxt"
    genderModel = "../Models/computer_vision_models/age_gender/gender_net.caffemodel"
    
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    try:
        return faceNet, ageNet, genderNet
    except cv2.error as e:
        print(f"Error loading age/gender models: {e}")
        print("Please ensure the model files are in the same directory as this script.")
        # Exit the program if models can't be loaded, as the module is unusable without them.
        exit()


def highlightFace(net, frame, conf_threshold=0.7):
    """(Original function from the provided code, kept for visualization)"""
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def detect_age_gender(frame, faceNet, ageNet, genderNet, padding=20):
    """
    Detects age and gender for all faces in an input image frame.

    Args:
        frame: The input image frame (a NumPy array).
        faceNet: The pre-trained face detection model.
        ageNet: The pre-trained age classification model.
        genderNet: The pre-trained gender classification model.
        padding: The amount of padding to add around the detected face for better classification.

    Returns:
        A list of dictionaries. Each dictionary contains the detected 'gender', 'age',
        and 'bbox' for a face found in the frame.
    """
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faceBoxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])

    detected_faces_data = []
    
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # detected_faces_data.append({'gender': gender, 'age': age[1:-1], 'bbox': faceBox}) #age in years
        detected_faces_data.append({'gender': gender, 'age': age[1:-1]})
        
    return detected_faces_data

if __name__ == '__main__':
    # This block will only execute when age_gender_detect.py is run directly.
    # It will be ignored when the file is imported as a module.
    # First cd to src\Code
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to the image file. If not provided, webcam will be used.')
    args = parser.parse_args()
    
    faceNet, ageNet, genderNet = load_age_gend_models()
    video = cv2.VideoCapture(args.image if args.image else 0)

    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break
        
        detections = detect_age_gender(frame, faceNet, ageNet, genderNet)
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        
        if not detections:
            print("No face detected")
        
        # Display results on the frame
        for i, faceBox in enumerate(faceBoxes):
            if i < len(detections):
                detection = detections[i]
                text = f'{detection["gender"]}, {detection["age"]}'
                cv2.putText(resultImg, text, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Detecting age and gender", resultImg)