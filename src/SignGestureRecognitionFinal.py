from tkinter import *
from keras.models import load_model 
import os
from keras.models import Model
from keras.preprocessing import image as image_utils
import numpy as np
import cv2
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
model = load_model('/Users/vignesh-pt3291/Downloads/Projects/gestmodelLS.h5')

gesture_names = {0: 'Caged',
                 1: 'Like',
                 2: 'Nice',
                 3: 'Blessings',
                 4: 'Peace',
                 5: 'Z',
                 6: 'Circles',
                 7: 'Released',
                 8: 'Four',
                 9: 'You',
                 10: 'I',
                 11: 'Birds',
                 12: 'Calls',
                 13: 'Super',
                 14: 'Little',
                 15: 'Joined'}

def get_gesture_details(X_test, y_test):
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    ac_ges = ""
    pred_ges = ""
    for i in range(len(pred)):
      ac_ges += gesture_names[y_true[i]] + " ; "
      pred_ges += gesture_names[pred[i]] + " ; "
    actisglabel.delete(1.0, END)
    actisglabel.insert(END, ac_ges)
    predisglabel.delete(1.0, END)
    predisglabel.insert(END, pred_ges)

def get_continuous_gesture_details(X_test, y_test):
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    ac_ges = ""
    pred_ges = ""
    for i in range(len(pred)):
      ac_ges += gesture_names[y_true[i]] + " "
      pred_ges += gesture_names[pred[i]] + " "
    actcsglabel.delete(1.0, END)
    actcsglabel.insert(END, ac_ges.replace("Z",";"))
    predcsglabel.delete(1.0, END)
    predcsglabel.insert(END, pred_ges.replace("Z",";"))

gestures = {'L': 'Like',
           'fist': 'Caged',
           'C': 'C',
           'okay': 'Nice',
           'peace': 'Peace',
           'palm': 'Blessings',
            '0': 'Circle',
            '1': 'Released',
            '4': 'Four',
            'g': 'You',
            'i': 'I',
            'x': 'Birds',
            'y': 'Calls',
            '7': 'Super',
            'q': 'Little',
            'r': 'Joined'
            }

gestures_map = {'Caged' : 0,
                'Like': 1,
                'Nice': 2,
                'Blessings': 3,
                'Peace': 4,
                'C': 5 ,
                'Circle': 6,
                'Released': 7,
                'Four': 8,
                'You': 9,
                'I': 10,
                'Birds':11,
                'Calls':12,
                'Super':13,
                'Little':14,
                'Joined':15
                }

def process_image(path):
    img = cv2.imread(path)
    img =  cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img)
    return img

def process_data(X_data, y_data):
    X_data = np.array(X_data, dtype = 'float32')
    X_data = np.stack((X_data,)*3, axis=-1)
    X_data /= 255
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return X_data, y_data

def walk_file_tree(relative_path):
    X_data = []
    y_data = [] 
    for directory, subdirectories, files in os.walk(relative_path):
        for file in files:
            if not file.startswith('.'):
                path = os.path.join(directory, file)
                cl = file.split('_')[0]
                gesture_name = gestures[cl]
                y_data.append(gestures_map[gesture_name])
                X_data.append(process_image(path))   

            else:
                continue

    X_data, y_data = process_data(X_data, y_data)
    return X_data, y_data

def get_confusion_matrix(X_test, y_test):
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    ConfusionMatrix.delete(1.0, END)
    ConfusionMatrix.insert(END, confusion_matrix(y_true, pred))

def get_performance_metrics(X_test, y_test):
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    PerformanceMetrics.delete(1.0, END)
    PerformanceMetrics.insert(END, classification_report(y_true, pred))

def RecognizeISG():
    try:
        if("All" in image_field.get("1.0", "end-1c")):
            path = "/Users/vignesh-pt3291/Downloads/Projects/Testing"
            X_test,y_test = walk_file_tree(path)
        else:
            images = image_field.get("1.0", "end-1c").split(',')
            X_test = []
            y_test = []
            for image in images:
                path = "/Users/vignesh-pt3291/Downloads/Projects/Testing"
                path = os.path.join(path,image)
                if(path != "/Users/vignesh-pt3291/Downloads/Projects/Testing/"): 
                    X_test.append(process_image(path))
                    y_test.append(gestures_map[gestures[image.split('_')[0]]])
            X_test,y_test = process_data(X_test, y_test)
        get_gesture_details(X_test,y_test)
    except Exception as e:
        actisglabel.delete(1.0, END)
        actisglabel.insert(END, str(e))
        predisglabel.delete(1.0, END)
        predisglabel.insert(END, str(e))

def ClearISG():
    actisglabel.delete(1.0, END)
    predisglabel.delete(1.0, END)
    image_field.delete(1.0, END)

def RecognizeCSG(): 
    try:
        if("All" in images_field.get("1.0", "end-1c")):
            f = open("/Users/vignesh-pt3291/Downloads/Projects/Input.txt","r")
            Lines = f.readlines() 
            f.close()
            X_test = []
            y_test = []
            for line in Lines: 
                images=line.strip().split(',')
                for image in images:
                    path = "/Users/vignesh-pt3291/Downloads/Projects/Testing"
                    path = os.path.join(path,image)
                    if(path != "/Users/vignesh-pt3291/Downloads/Projects/Testing/"): 
                        X_test.append(process_image(path))
                        y_test.append(gestures_map[gestures[image.split('_')[0]]])
        else:
            images = images_field.get("1.0", "end-1c").split(',')
            X_test = []
            y_test = []
            for image in images:
                path = "/Users/vignesh-pt3291/Downloads/Projects/Testing"
                path = os.path.join(path,image)
                if(path != "/Users/vignesh-pt3291/Downloads/Projects/Testing/"): 
                    X_test.append(process_image(path))
                    y_test.append(gestures_map[gestures[image.split('_')[0]]])
        X_test,y_test = process_data(X_test, y_test)
        get_continuous_gesture_details(X_test,y_test)
    except Exception as e:
        actcsglabel.delete(1.0, END)
        actcsglabel.insert(END, str(e))
        predcsglabel.delete(1.0, END)
        predcsglabel.insert(END, str(e))

def ClearCSG():
    actcsglabel.delete(1.0, END)
    predcsglabel.delete(1.0, END)
    images_field.delete(1.0, END)

def getPMCM():
    relative_path = '/Users/vignesh-pt3291/Downloads/Projects/Testing'
    X_test, y_test = walk_file_tree(relative_path)
    get_performance_metrics(X_test,y_test)
    get_confusion_matrix(X_test,y_test)
    
def ClearPMCM():
    PerformanceMetrics.delete(1.0, END)
    ConfusionMatrix.delete(1.0, END)

if __name__ == "__main__": 
    gui = Tk() 
    gui.configure(background="#ffe") 
    gui.title("Continuous Sign Language Recognition System") 
    gui.minsize(1800,1200) 
    image_label = Label(gui, text = "Isolated Sign Gesture Sample Images", bg='orange')
    image_field = Text(gui, height=1, width=191, bg='orange')
    aisl = Label(gui, text = "Actual Isolated Sign Gesture", bg='green')
    pisl = Label(gui, text = "Predicted Isolated Sign Gesture", bg='green')
    actisglabel = Text(gui, height=1, width=191, bg='green')
    predisglabel = Text(gui, height=1, width=191, bg='green')
    photo = PhotoImage(file = '/Users/vignesh-pt3291/Downloads/animatedashokchakra.gif')
    photo = photo.subsample(5,5)
    isrb = Button(gui, text="ISGRecognition", command=RecognizeISG, fg='black', bg='orange', height=67, width=300, compound=TOP, image = photo)
    clisb = Button(gui, text="Clear", command=ClearISG, fg='black', bg='orange', height=67, width=300, image = photo, compound=TOP)
    images_label = Label(gui, text = "Continuous Sign Gesture Sample Images", bg='orange')
    images_field = Text(gui, height=1, width=191, bg='orange')
    acol = Label(gui, text = "Actual Continuous Sign Gesture", bg='green')
    pcol = Label(gui, text = "Predicted Continuous Sign Gesture", bg='green')
    actcsglabel = Text(gui, height=1, width=191, bg='green')
    predcsglabel = Text(gui, height=1, width=191, bg='green')
    csrb = Button(gui, text="CSGRecognition", command=RecognizeCSG, fg="black", bg='white', height=67, width=300, image = photo, compound=TOP)
    clcob = Button(gui, text="Clear", command=ClearCSG, fg="black", bg='white', height=67, width=300, image = photo, compound=TOP)
    #cmb = Button(gui, text="GetConfusionMatrix", command=getCM, fg="black", bg='green', height=2, width=16)
    #clcmb = Button(gui, text="Clear", command=ClearCM, fg="black", bg='green', height=2, width=16)
    ConfusionMatrix = Text(gui, height=16, width=68, bg='orange')
    pmb = Button(gui, text="GetConfusionMatrix&PerformanceMetrics", command=getPMCM, fg="black", bg='green', height=67, width=300, image = photo, compound=TOP)
    clpmb = Button(gui, text="Clear", command=ClearPMCM, fg="black", bg='green', height=67, width=300, image = photo, compound=TOP)
    conf = Label(gui, text="ConfusionMatrix", bg='orange')
    PerformanceMetrics = Text(gui, height=22, width=67, bg='green')
    perf = Label(gui, text="PerformanceMetrics", bg='green')
    isglabel = Label(gui, text="Isolated Sign Gesture Recognition", bg='orange',font=('Helvetica', 9, 'bold'))
    csglabel = Label(gui, text="Continuous Sign Gesture Recognition", bg='orange', font=('Helvetica', 9, 'bold'))
    cmpmlabel = Label(gui, text="Get Confusion Matrix & Performance Metrics", bg='orange', font=('Helvetica', 9, 'bold'))
    cslslabel = Label(gui, text="Continuous Sign Language Recognition System", bg='#ffd', font=('Helvetica', 12, 'bold'))
    cslslabel.grid(row=0, column=1)
    isglabel.grid(row=1, column=1)
    image_label.grid(row=2, column=0)
    image_field.grid(row=2, column=1)
    clisb.grid(row=3, column=0)
    isrb.grid(row=3, column=1)
    aisl.grid(row=4, column=0)
    actisglabel.grid(row=4, column=1)
    pisl.grid(row=5, column=0)
    predisglabel.grid(row=5, column=1)
    csglabel.grid(row=6, column=1)
    images_label.grid(row=7, column=0)
    images_field.grid(row=7, column=1)
    clcob.grid(row=8, column=0)
    csrb.grid(row=8, column=1)
    acol.grid(row=9, column=0)
    actcsglabel.grid(row=9, column=1)
    pcol.grid(row=10, column=0)
    predcsglabel.grid(row=10, column=1)
    cmpmlabel.grid(row=11,column=1)
    #clcmb.grid(row=12, column=0)
    #cmb.grid(row=12, column=1)
    conf.grid(row=12, column=0)
    ConfusionMatrix.grid(row=12, column=1)
    clpmb.grid(row=13, column=0)
    pmb.grid(row=13, column=1)
    perf.grid(row=14, column=0)
    PerformanceMetrics.grid(row=14, column=1)
    gui.mainloop() 
