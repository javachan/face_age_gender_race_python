import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib

from tqdm import tqdm
from imutils import face_utils,resize
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential,load_model
from keras.layers import Conv2D,Flatten,MaxPooling2D,Dense,Dropout,Activation,BatchNormalization
from keras.optimizers import Adam

class data_reader:
    def __init__(self,PATH,ages):
        self.PATH = PATH
        self.ages = ages

        if type(self.ages) == np.ndarray:
            self.ages = self.ages.tolist()

        self.ages.sort()

    def read_data_age(self,x_name = "XNPY.npy",y_name = "YNPY.npy",read_from_npy_file = True,make_gray = True,save = True,test_size = 0.1,tocateg = True):#1 means female, 0 means male
        if read_from_npy_file:
            try:
                x = np.load(x_name)
                y = np.load(y_name)
            except:
                raise FileNotFoundError("XNPY or YNPY file doesn't exists. If you don't have files, set 'read_from_npy_file' parameter as False")

        if not read_from_npy_file:
            x = []
            y = []

            for image in tqdm(os.listdir(self.PATH)):
                age = int(image.split("_")[0])
                if self.ages.count(age):
                    img = cv2.imread(os.path.join(self.PATH,image))

                    if make_gray:
                        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                    img = np.array(img)

                    x.append(img)
                    y.append(age)

            x = np.array(x)
            y = np.array(y)

            if tocateg:
                y = to_categorical(y,num_classes=int(self.ages[-1]) + 1)

            if save:
                np.save(x_name,x)
                np.save(y_name,y)

        if make_gray == True and x.shape[-1] != 1:
            x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)

        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size)

        return x_train, x_test, y_train, y_test,int(self.ages[-1]) + 1

    def read_data_gender(self,x_name = "XNPY.npy",y_name = "YNPY.npy",read_from_npy_file = True,make_gray = True,save = True,test_size = 0.1,tocateg = True):#1 means female, 0 means male
        if read_from_npy_file:
            try:
                x = np.load(x_name)
                y = np.load(y_name)
            except:
                raise FileNotFoundError("XNPY or YNPY file doesn't exists. If you don't have files, set 'read_from_npy_file' parameter as False")

        if not read_from_npy_file:
            x = []
            y = []

            for image in tqdm(os.listdir(self.PATH)):
                gender = int(image.split("_")[1])

                img = cv2.imread(os.path.join(self.PATH,image))

                if make_gray:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                img = np.array(img)

                x.append(img)
                y.append(gender)

            x = np.array(x)
            y = np.array(y)

            if tocateg:
                y = to_categorical(y,num_classes=2)

            if save:
                np.save(x_name,x)
                np.save(y_name,y)

        if make_gray == True and x.shape[-1] != 1:
            x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)

        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size)

        return x_train, x_test, y_train, y_test,2

    def read_data_race(self,x_name = "XNPY.npy",y_name = "YNPY.npy",read_from_npy_file = True,make_gray = True,save = True,test_size = 0.1,tocateg = True):#1 means female, 0 means male
        if read_from_npy_file:
            try:
                x = np.load(x_name)
                y = np.load(y_name)
            except:
                raise FileNotFoundError("XNPY or YNPY file doesn't exists. If you don't have files, set 'read_from_npy_file' parameter as False")

        if not read_from_npy_file:
            x = []
            y = []

            for image in tqdm(os.listdir(self.PATH)):
                race = int(image.split("_")[2])

                img = cv2.imread(os.path.join(self.PATH,image))

                if make_gray:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                img = np.array(img)

                x.append(img)
                y.append(race)

            x = np.array(x)
            y = np.array(y)

            if tocateg:
                y = to_categorical(y,num_classes=6)

            if save:
                np.save(x_name,x)
                np.save(y_name,y)

        if make_gray == True and x.shape[-1] != 1:
            x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)

        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size)

        return x_train, x_test, y_train, y_test,6

class modeler:
    NAME = "25epoch_0.0005lr_ccloss_softmaxlasta_relutypa_0.5drop_truebatchn"
    def __init__(self,x_train, x_test, y_train, y_test,lr = 0.0005,dropout_level = 0.4,ishape = (200,200,1),num_classes = 61,
                 epoch = 100,batch_size = 64,loss_f = "categorical_crossentropy",last_activation = "softmax",typical_activation = "relu",
                 batchn = False):

        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test
        self.lr = lr
        self.dropout_level = dropout_level
        self.ishape = ishape
        self.epoch = epoch
        self.batch_size = batch_size
        self.loss = loss_f
        self.last_activation = last_activation
        self.typical_activation = typical_activation
        self.num_classes = num_classes
        self.batchn = batchn

        model = self.make_model()

        model.fit(x_train,y_train,batch_size=self.batch_size,epochs=self.epoch,validation_data=[x_test,y_test])
        model.save(f"{modeler.NAME}.model")

    def make_model(self):
        model = Sequential()

        model.add(Conv2D(16, (2,2),input_shape=self.ishape))
        model.add(Activation(self.typical_activation))
        model.add(MaxPooling2D((2,2)))
        if self.batchn:
            model.add(BatchNormalization())
        model.add(Dropout(self.dropout_level))

        model.add(Conv2D(32, (2,2)))
        model.add(Activation(self.typical_activation))
        model.add(MaxPooling2D((2,2)))
        if self.batchn:
            model.add(BatchNormalization())
        model.add(Dropout(self.dropout_level))

        model.add(Conv2D(64, (2,2)))
        model.add(Activation(self.typical_activation))
        model.add(MaxPooling2D((2,2)))
        if self.batchn:
            model.add(BatchNormalization())
        model.add(Dropout(self.dropout_level))

        model.add(Conv2D(128, (2,2)))
        model.add(Activation(self.typical_activation))
        model.add(MaxPooling2D((2,2)))
        if self.batchn:
            model.add(BatchNormalization())
        model.add(Dropout(self.dropout_level))

        model.add(Conv2D(256, (2,2)))
        model.add(Activation(self.typical_activation))
        model.add(MaxPooling2D((2,2)))
        if self.batchn:
            model.add(BatchNormalization())
        model.add(Dropout(self.dropout_level))

        model.add(Conv2D(512, (2,2)))
        model.add(Activation(self.typical_activation))
        model.add(MaxPooling2D((2,2)))
        if self.batchn:
            model.add(BatchNormalization())
        model.add(Dropout(self.dropout_level))

        model.add(Conv2D(1024, (2,2)))
        model.add(Activation(self.typical_activation))
        if self.batchn:
            model.add(BatchNormalization())
        model.add(Dropout(self.dropout_level))

        model.add(Flatten())

        model.add(Dense(512))
        model.add(Activation(self.typical_activation))
        model.add(Dropout(self.dropout_level))

        model.add(Dense(256))
        model.add(Activation(self.typical_activation))
        model.add(Dropout(self.dropout_level))

        model.add(Dense(128))
        model.add(Activation(self.typical_activation))
        model.add(Dropout(self.dropout_level))

        model.add(Dense(self.num_classes, activation=self.last_activation))

        model.summary()

        model.compile(loss = self.loss, optimizer=Adam(), metrics=["accuracy"])

        return model

class predicter:
    def predicter(AGE_MODEL,GENDER_MODEL,RACE_MODEL,img,detector,q = 10,map_age = None,map_gender = None,map_race = None):
        image = img.copy()
        faces = predicter.give_face_on_image(image, detector,q)

        for face in faces:
            try:
                x,y,w,h = face[0],face[1],face[2],face[3]
                cv2.rectangle(image,(x,y),(w,h),(31,31,31),2)

                w1,h1 = 200,200
                faci = img[y:h,x:w]
                faci = cv2.resize(faci,(w1,h1))
                faci = cv2.cvtColor(faci,cv2.COLOR_BGR2GRAY)

                faci = np.array(faci)
                faci = faci.reshape(1,w1,h1,1)

                agep, genp, racp = predicter.make_predicts(AGE_MODEL,GENDER_MODEL,RACE_MODEL,faci)

                if map_age != None:
                    agep = map_age.get(agep)

                if map_gender != None:
                    genp = map_gender.get(genp)

                if map_race != None:
                    racp = map_race.get(racp)

                print(agep, genp, racp)

                cv2.putText(image,f"{agep}-{genp}-{racp}",(x,y-int(h/20)),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,255),3)
            except Exception as e:
                print(e)
                pass

        image = resize(image, width=600)
        return image

    def make_predicts(AGE_MODEL,GENDER_MODEL,RACE_MODEL,face):
        agep = np.argmax(AGE_MODEL.predict(face))
        genp = np.argmax(GENDER_MODEL.predict(face))
        racp = np.argmax(RACE_MODEL.predict(face))

        return agep,genp,racp

    def give_face_on_image(image,detector,q):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        faces = []

        for (index, rect) in enumerate(rects):
            faces.append((rect.left(),rect.top(),rect.right(),rect.bottom()))

        return faces

if __name__ == '__main__':
    TRAIN = False
    TRAIN_FOR = "age" # age/gender/race
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if TRAIN:
        age_list = [1,2,3,5,8,10,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,100]

        reader = data_reader("UTKFace",ages=age_list)

        if TRAIN_FOR == "age":
            x_train, x_test, y_train, y_test,nmc = reader.read_data_age(test_size = 0.1,read_from_npy_file=False,save=True,tocateg=True)
        elif TRAIN_FOR == "gender":
            x_train, x_test, y_train, y_test,nmc = reader.read_data_gender(test_size = 0.1,read_from_npy_file=False,save=True,tocateg=True)
        elif TRAIN_FOR == "race":
            x_train, x_test, y_train, y_test,nmc = reader.read_data_race(test_size = 0.1,read_from_npy_file=False,save=True,tocateg=True)
        else:
            raise Warning("Wrong 'TRAIN_FOR' variable. Has to be age, gender or race.")

        modeler.NAME = modeler.NAME + "_" + TRAIN_FOR
        modeler(x_train, x_test, y_train, y_test,epoch=25,batch_size=128,batchn=True,dropout_level=0.5,loss_f="categorical_crossentropy",num_classes=nmc)

    if not TRAIN:
        map_gender = {0: "male", 1: "female"}
        map_race = {0: "white", 1: "black",2: "asian",3: "indian",4: "other"}
        detector = dlib.get_frontal_face_detector()

        AGE_PATH = r"models\age_models\25epoch_0.0005lr_ccloss_softmaxlasta_relutypa_0.5drop_truebatchn_AGE_to50all.model"
        GENDER_PATH = r"models\gender_models\100epoch_0.0005lr_ccloss_softmaxlasta_relutypa_0.5drop_truebatchn_GENDER.model"
        RACE_PATH = r"models\race_models\200epoch_0.0005lr_ccloss_softmaxlasta_relutypa_0.5drop_truebatchn_RACE.model"

        AGE_MODEL = load_model(AGE_PATH)
        GENDER_MODEL = load_model(GENDER_PATH)
        RACE_MODEL = load_model(RACE_PATH)

        def on_cam(AGE_MODEL,GENDER_MODEL,RACE_MODEL,detector, map_gender,map_race,cama = 0,flip = True):
            cap = cv2.VideoCapture(cama)
            while True:
                _,frame = cap.read()

                if flip:
                    frame = cv2.flip(frame,1)

                frame = predicter.predicter(AGE_MODEL,GENDER_MODEL,RACE_MODEL,frame,detector, map_gender=map_gender,map_race=map_race)

                cv2.imshow("frame",frame)

                if cv2.waitKey(10) == 27:
                    cv2.destroyAllWindows()
                    cap.release()
                    break

        def on_img(AGE_MODEL,GENDER_MODEL,RACE_MODEL,path,detector, map_gender,map_race):
            frame = cv2.imread(path)
            frame = predicter.predicter(AGE_MODEL,GENDER_MODEL,RACE_MODEL, frame, detector, map_gender=map_gender,map_race=map_race)
            cv2.imshow("frame", frame)
            cv2.waitKey(0)

        def on_list(path,on_img):
            for img in os.listdir(path):
                if img.endswith(".jpg") or img.endswith(".png"):
                    on_img(AGE_MODEL,GENDER_MODEL,RACE_MODEL,os.path.join(path,img),detector, map_gender=map_gender,map_race=map_race)


        on_list(r"images",on_img)
        #on_cam(AGE_MODEL, GENDER_MODEL, RACE_MODEL, detector, map_gender=map_gender, map_race=map_race)
        #on_img(AGE_MODEL, GENDER_MODEL, RACE_MODEL,r"C:\Users\burak\Desktop\python_jobs\pindown\pins\_\_1.jpg", detector, map_gender=map_gender, map_race=map_race)
















































