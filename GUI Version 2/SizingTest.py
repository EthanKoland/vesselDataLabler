import tkinter as tk
from math import floor, ceil
from PIL import ImageTk, Image

class SizingTest(tk.Tk):
    def __init__(self, controller):
        super().__init__()
        
        self.geometry("1440x1080")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        
        self.imageGrid = tk.Frame(self, bg="red", padx=10, pady=10)
        self.imageGrid.grid(row=0, column=0, sticky="nsew")
        
        self.imageGrid.grid_columnconfigure(0, weight=1)
        
        self.imageGrid.grid_rowconfigure(0, weight=1)
        self.imageGrid.grid_rowconfigure(1, weight=8)
        self.imageGrid.grid_rowconfigure(2, weight=3)
        
        
        self.brushControll = tk.Frame(self.imageGrid, bg="yellow", padx=10, pady=10)
        self.brushControll.grid(row=0, column=0, sticky="nsew")
        
        self.brushControll.grid_columnconfigure(0, weight=1)
        self.brushControll.grid_columnconfigure(0, weight=1)
        self.brushControll.grid_columnconfigure(0, weight=2)
        
        self.brushControll.grid_rowconfigure(0, weight=1)
        
        
        
        self.image = tk.Frame(self.imageGrid, bg="purple")
        self.image.grid(row=1, column=0, sticky="nsew")
        self.image.bind('<Configure>', self._resize)
        
        self.image.grid_columnconfigure(0, weight=1)
        self.image.grid_rowconfigure(0, weight=1)
        
        # self.subImage = tk.Frame(self.image, bg="pink")
        self.img = ImageTk.PhotoImage(Image.open("CSAngioImages/0000.jpg"), width=100, height=100)
        self.subImage = tk.Label(self.image, image=self.img)
        self.subImage.grid(row=0, column=0, sticky="nsew", padx=10, pady=50)
        
        self.surronding = tk.Frame(self.imageGrid, bg="orange", padx=10, pady=10)
        self.surronding.grid(row=2, column=0, sticky="nsew")
        self.surronding.bind('<Configure>', self._resize2)
        
        self.surronding.rowconfigure(0, weight=1)
        
        self.surrongingImage = [tk.Frame(self.surronding, bg="black") for i in range(4)]
        
        for i in range(4):
            self.surronding.grid_columnconfigure(i, weight=1)
            self.surrongingImage[i].grid(row=0, column=i, sticky="nsew")
            
        
        
        # self.imageGrid.grid_columnconfigure(0, weight=1)
        # self.imageGrid.grid_columnconfigure(0, weight=1)
        # self.imageGrid.grid_columnconfigure(0, weight=2)
        
        
        
        
        
        # self.testButton = tk.Button(self.imageGrid, text="Test Button")
        # self.testButton.pack()
        
        # self.imageGrid.
        
        self.parameters = tk.Frame(self, bg="blue", padx=10, pady=10)
        self.parameters.grid(row=0, column=1, sticky="nsew")
        
        self.parameters.grid_columnconfigure(0, weight=1)
        
        self.parameters.grid_rowconfigure(0, weight=1)
        self.parameters.grid_rowconfigure(1, weight=1)
        self.parameters.grid_rowconfigure(2, weight=1)
       
        self.controllParameters = tk.LabelFrame(self.parameters, text="Control Parameters", bg="green")
        self.controllParameters.grid(row=0, column=0, sticky="nsew")
        
        self.filterParameters = tk.LabelFrame(self.parameters, text="Filter Parameters", bg="green")
        self.filterParameters.grid(row=1, column=0, sticky="nsew")
        
        self.brushFilters = tk.LabelFrame(self.parameters, text="brush Parameters", bg="green")
        self.brushFilters.grid(row=2, column=0, sticky="nsew")
        
        
        
        
        
        
        
        self.mainloop()
        
    def _resize(self, event):
        '''Modify padding when window is resized.'''
        w, h = event.width, event.height
        # w1, h1 = self.content.winfo_width(), self.content.winfo_height()
        # print(w1, h1)  # should be equal
        
        minSize = min(w, h)
        direction = "w" if w < h else "h"
        
        dimsX  = [0] * 3
        dimsY  = [0] * 3
        
        if(w < h):
            offset = (h - w)/2
            dimsX = [0, w, 0]
            dimsY = [offset, w, ceil(offset)]
        elif(w > h):
            offset = (w - h)/2
            dimsX = [offset, h, ceil(offset)]
            dimsY = [0, h, 0]
        else:
            dimsX = [0, w, 0]
            dimsY = [0, h, 0]
            
        self.subImage.grid(padx=dimsX[0], pady=dimsY[0]//2)
        #Print the dimensions of the subImage
        w, h = self.subImage.winfo_width(), self.subImage.winfo_height()
        # print(w, h) 
            
        
        # print(w, h)
        
    def _resize2(self, event):
        # w, h = event.width//4, event.height
        # print(w, h)
        
        # subImageX, subImageX = self.surrongingImage[0].winfo_width(), self.surrongingImage[0].winfo_height()
        
        # offsetX = min(10, 0.1*w)
        # offsetY = min(10, 0.1*h)
        
        # if(w < h):
        #     offsetY = (h - w)/2
        # elif(w > h):
        #     offsetX = (w - h)/2
            
        # for i in range(4):
        #     self.surrongingImage[i].grid(padx=offsetX, pady=offsetY)
        
        #Event gives the dims of the surronding frame
        #We want to have N images in the frame, currently 4
        # Each of the frames should be square, and take up 1/4 of the width of the surronding frame
        
        w, h = event.width, event.height
        w1, h2 = event.width, event.height
        
        #20 becase 2 x padding of 10
        w = max(w//4,0)
        # print(w, h)
        
        offsetX = 10
        offsetY = 10
        
        # if(w < h):
        #     offsetY += (h - w)/2
        #     # offsetX = min(10, 0.1*w)
        # elif(w > h):
        #     offsetX += (w - h)/2
        
        w1 = min(w, h)
        print(w, h)
        print(offsetX, offsetY)
            
        for i in range(4):
            self.surrongingImage[i].config(width=offsetX, height=offsetY)
        
        


class imageSampleFrame(tk.Frame):
    def __init__(self, controller):
        super().__init__()
        self["bg"] = "red"
        
        
        
if(__name__ == "__main__"):#
    SizingTest(None)