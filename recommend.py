import cv2
import os
from keras.applications.vgg16 import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image
from keras.models import model_from_json
def read_image(MyImage):
    path = './static/{}'.format(MyImage)
    img = cv2.imread(path)
    recommend(img,MyImage)

def recommend(img,nam):
    feature_list = np.array(pickle.load(open('embedding1.pkl', 'rb')))
    filenames = pickle.load(open('filenames1.pkl', 'rb'))
    json_file = open("mod.json", 'r')
    loded_model_json = json_file.read()
    model = model_from_json(loded_model_json)
    model.load_weights("mod.h5")
    print(model)
    detector = MTCNN()
    # load image -> face detection
    sample_img = img

    results = detector.detect_faces(sample_img)
    x, y, width, height = results[0]['box']
    face = sample_img[y:y + height, x:x + width]

    # cv2.imshow('output',face)
    # cv2.waitKey(0)

    # extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocess_img = preprocess_input(expanded_img)
    result = model.predict(preprocess_img).flatten()
    # print(result)
    # print(result.shape)

    # find the cosine distance of current image with all the 8566 features
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]

    temp_img = cv2.imread(filenames[index_pos])
    # print(filenames[index_pos])
    #cv2.imshow('output', temp_img)
    name = 'output' + nam
    path = './static/{}'.format(name)
    cv2.imwrite(path,temp_img)
    print(name)
    return name