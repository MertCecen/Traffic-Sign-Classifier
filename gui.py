import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
#load the trained model to classify sign
from tensorflow.keras.models import load_model

sign_labels = ["Speed limit (20 km/h)",
               "Speed limit (30 km/h)",
               "Speed limit (50 km/h)",
               "Speed limit (60 km/h)",
               "Speed limit (70 km/h)",
               "Speed limit (80 km/h)",
               "End of speed limit (80 km/h)",
               "Speed limit (100 km/h)",
               "Speed limit (120 km/h)",
               "No passing",
               "No passing vehicle over 3.5 tons",
               "Right of way at intersection",
               "Priority road",
               "Yield",
               "Stop",
               "No vehicles",
               "Vehicle > 3.5 tons prohibited",
               "No entry",
               "General caution",
               "Dangerous curve left",
               "Dangerous curve right",
               "Double curve",
               "Bumpy road",
               "Slippery road",
               "Road narrows on the right",
               "Road work",
               "Traffic signals",
               "Pedestrians",
               "Children crossing",
               "Bicycles crossing",
               "Beware of ice/snow",
               "Wild animal crossing",
               "End speed + passing limits",
               "Turn right ahead",
               "Turn left ahead",
               "Ahead only",
               "Go straight or right",
               "Go straight or left",
               "Keep right",
               "Keep left",
               "Roundabout mandatory",
               "End of no passing",
               "End of no passing vehicles > 3.5 tons"]

classes = {}

for i in range(len(sign_labels)):
    classes[i+1] = sign_labels[i]
#print(classes)


model = load_model("traffic_classifier.h5")

root = tk.Tk()
root.geometry("700x600")
root.title("Traffic Sign Classifier")
root.configure(background = "#cc8b97")

my_label = Label(root, background = "#cc8b97", foreground = "white", font = ("arial", 16, "bold"))
sign_image = Label(root)

def classify_image(path):
    image = Image.open(path)
    image = image.resize((30,30))
    image = np.expand_dims(image, axis = 0)
    prediction = model.predict_classes([image])[0]
    sign = classes[prediction + 1]
    print("Sign label: ", sign)
    my_label.configure(foreground = "#1c0110", text = sign)
    
def show_classify_button(path):
    classify_button = Button(root, text = "Classify Image", command = lambda : classify_image(path), padx = 10, pady = 5)
    classify_button.configure(background = "#1c0110", foreground = "white", font = ("arial", 12, "bold"))
    classify_button.place(relx = 0.75, rely = 0.45)
    
    
def upload_image():
    try:
        path = filedialog.askopenfilename()
        uploaded = Image.open(path)
        my_img = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image = my_img)
        sign_image.image = my_img
        my_label.configure(text = "")
        show_classify_button(path)
    except:
        print("Error occured while uploading image.")
        
def show_upload_button():
    upload_button = Button(root, text = "Upload Image", command = upload_image, padx = 10, pady = 5)
    upload_button.configure(background = "#1c0110", foreground = "white", font = ("arial", 12, "bold"))
    upload_button.pack(side = BOTTOM, pady = 60)

def main():
    show_upload_button()
    sign_image.pack(side = BOTTOM, expand = True)
    my_label.pack(side = BOTTOM, expand = True)
    header = Label(root, text = "Your traffic sign", pady = 15, padx = 10, font  = ("arial",20 , "bold"))
    header.configure(background = "#1c0110", foreground = "white")
    header.pack(pady = 15)

    root.mainloop()
    
main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




















