from tkinter import *
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np


def predict_digit(img, method_obj, mean, std, normalize_fn):
    #convert rgb to grayscale
    img = img.convert('1')
    img.show()
    #rescale
    basewidth = 32
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
    #normalize
    img = np.asarray(img)
    img = img.astype(int)
    img = img.astype(float)
    img = normalize_fn(img, mean, std)
    #reshape into vector
    tensor = img.reshape(1, 32*32)
    #predict 
    res = method_obj.predict(tensor)[0]
    return res
class App(tk.Tk):
    def __init__(self, method_obj, mean, std, normalize_fn):
        tk.Tk.__init__(self)
        self.mean = mean
        self.std = std
        self.normalize_fn = normalize_fn
        self.method_obj=method_obj
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command =         self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        #bindings
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.dict = {0: "int", 1 :"sum",2 :"infinity",3 :"alpha",4 :"xi",5 :"equiv",6 :"partial",7 :"mathds R",8 :"in",9 :"square"
                     ,10 :"forall",11 :"approx",12 :"sim",13 :"Rightarrow",14 :"subseteq",15 :"pi",16 :"pm",17 :"neq",18 :"varphi",19 :"times" }
    def clear_all(self):
        self.canvas.delete("all")
    def classify_handwriting(self):
        #capturing the canvas
        im =ImageGrab.grab(bbox=(
        self.canvas.winfo_rootx()+30,
        self.canvas.winfo_rooty()+30,
        self.canvas.winfo_rootx() + self.canvas.winfo_width()+50,
        self.canvas.winfo_rooty() + self.canvas.winfo_height()+50
            ))
        #predict label with given image
        digit= predict_digit(im, self.method_obj, self.mean, self.std, self.normalize_fn)
        self.label.configure(text= self.dict[digit])
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=12
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
def test(method_obj, mean, std, normalize_fn):
    app = App(method_obj, mean, std, normalize_fn)
    mainloop()