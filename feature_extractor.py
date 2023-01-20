import os
import pickle

'''fetching mame of all the actors
actors = os.listdir('Data')
print(actors)

#fetching all images of all actors in a list
filenames=[]
for actor in actors:
    c =0
    for file in os.listdir(os.path.join('Data',actor)):
        if c==50:
            break
        c+=1
        filenames.append(os.path.join('data',actor,file))

print(len(filenames))

#creating pickle for this final image list
pickle.dump(filenames,open('filenames1.pkl','wb'))'''

#==========================================================================


#from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filenames1.pkl','rb'))
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
print(model.summary())

def feature_extractor(img_path,model):
    img = load_img(img_path,target_size=(224,224))
    img_array = img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocess_img = preprocess_input(expanded_img)

    result = model.predict(preprocess_img).flatten()
    return result

features = []

for file in tqdm(filenames):
    result = features.append(feature_extractor(file,model))

pickle.dump(features,open('embedding1.pkl','wb'))