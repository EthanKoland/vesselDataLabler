import tkinter as tk
from tkinter import Widget, ttk
import numpy as np

from PIL import Image, ImageTk

class vesselEditor(tk.Tk):
    def __init__(self, controller):
        super().__init__()
        self.title("Vessel Editior")
        # self.geometry("800x600")
        
        # self.pool = pool
        # self.controller = controller
        # self.queue = []
        
        # self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        
        
        
        
        self.canvas = canvasEditior(self)
        self.canvas.grid(column=0, row=0, sticky=(tk.N,tk.W,tk.E,tk.S))
        
        self.labelFrame = tk.LabelFrame(self, text="Image Info")
        self.labelFrame.grid(column=1, row=0, sticky=(tk.N,tk.W,tk.E,tk.S))
        
        
        self.loadFolder_button = tk.Button(self.labelFrame, text="Load Folder", command=self.loadFolder)
        self.loadFolder_button.pack()
        
        
        self.loadImage_button = tk.Button(self.labelFrame, text="Load Image", command=self.loadImage)
        self.loadImage_button.pack()
        
        self.saveImage_button = tk.Button(self.labelFrame, text="Save Image", command=self.saveImage)
        self.saveImage_button.pack()
        
        self.nextImage_button = tk.Button(self.labelFrame, text="Next Image", command=self.nextImage)
        self.nextImage_button.pack()
        
        self.prevoiusImage_button = tk.Button(self.labelFrame, text="Previous Image", command=self.prevoiusImage)
        self.prevoiusImage_button.pack()
        
        self.parameters_button = tk.Button(self.labelFrame, text="Parameters", command=self.parameters)
        self.parameters_button.pack()
        
        
        
        self.mainloop()
        
    def loadFolder(self, event):
        print("load folder")
        pass
        
    def loadImage(self, event):
        print("load image")
        pass

    def saveImage(self, event):
        print("save image")
        pass
    
    def nextImage(self, event):
        print("next image")
        pass
    
    def prevoiusImage(self, event):
        print("previous image")
        pass
    
    def parameters(self, event):
        print("parameters")
        pass


class canvasEditior(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.image = Image.open('FrangiFilter/0002.jpg')
        self.imageObject = ImageTk.PhotoImage(self.image)
        self.erase = False
        
        self.canvas = tk.Canvas(parent, width = self.imageObject.width(), height = self.imageObject.height())
        self.canvasImage = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imageObject)
    

        self.mask = np.zeros((self.imageObject.width(), self.imageObject.height()))

        self.canvas.grid(column=0, row=0, sticky=(tk.N,tk.W,tk.E,tk.S))
        self.canvas.focus_set()
        # h.grid(column=0, row=1, sticky=(W,E))
        # v.grid(column=1, row=0, sticky=(N,S))
        # parent.grid_columnconfigure(0, weight=1)
        # parent.grid_rowconfigure(0, weight=1)

        self.lastx, self.lasty = 0, 0

        self.lineWidth = 1
        
        self.canvas.bind("<Button-1>", self.xy)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<q>", self.undo)
        self.canvas.bind("<w>", self.redo)
        self.canvas.bind("<B1-ButtonRelease>", self.doneStroke)
        self.canvas.bind("<z>", self.massUndo)
        self.canvas.bind("<x>", self.massRedo)

        self.draw = self.canvas.create_rectangle((10, 10, 30, 30), fill="white", tags=('palettePen', 'palettewhite'))
        self.canvas.tag_bind(self.draw , "<Button-1>", lambda x: self.setColor("white"))
        self.eraseButton = self.canvas.create_rectangle((10, 35, 30, 55), fill="black", tags=('palettePen', 'paletteErase', 'paletteSelected'))
        self.canvas.tag_bind(self.eraseButton, "<Button-1>", lambda x: self.eraseFunction(x))
        # self.erase = self.canvas.create_rectangle((10, 60, 30, 80), fill="black", tags=('palettePen', 'paletteblack', 'paletteSelected'))
        # self.canvas.tag_bind(self.erase, "<Button-1>", lambda x: self.eraseButton(x))

        id = self.canvas.create_rectangle((10, 85, 30, 105), fill="black", tags=('paletteSize', 'palette1', 'paletteSizeSelected'))
        self.canvas.tag_bind(id, "<Button-1>", lambda x:self.changeLineWidth(1))
        id = self.canvas.create_rectangle((10, 110, 30, 130), fill="black", tags=('paletteSize', 'palette5'))
        self.canvas.tag_bind(id, "<Button-1>", lambda x: self.changeLineWidth(5))
        id = self.canvas.create_rectangle((10, 135, 30, 155), fill="black", tags=('paletteSize', 'palette10'))
        self.canvas.tag_bind(id, "<Button-1>", lambda x: self.changeLineWidth(10))
        
        self.setColor('white')
        self.canvas.itemconfigure('palette', width=self.lineWidth)
        self.canvas.itemconfigure('paletteSizeSelected', outline='blue')
        self.canvas.itemconfigure('paletteSelected', outline='red')
        
        
        #(action, size, color, x1, y1, x2, y2)
        self.drawActions = []
        self.massAction = []
        self.currentAction = []
        self.redoAction = []
        self.redoLargeAction = []
 
    def xy(self, event):
        self.currentAction = []
        self.lastx, self.lasty = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def setColor(self, newcolor):
        self.erase = False
        self.color = newcolor
        self.canvas.dtag('palettePen', 'paletteSelected')
        self.canvas.itemconfigure('palettePen', outline='white')
        self.canvas.addtag('paletteSelected', 'withtag', 'palette%s' % self.color)
        self.canvas.itemconfigure('paletteSelected', outline='red')
        
    def draw(self,event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        self.redoAction = []
        self.redoLargeAction = []

        if(self.erase == True):
            self.eraseLine(x,y)
        else:
            self.drawLine(self.lastx,self.lasty,x,y, self.lineWidth, self.color)
       
    def drawLine(self, x1,y1,x2,y2, lineWidth, color, append=True):
        line = self.canvas.create_line((x1,y1,x2,y2), fill=color, width=lineWidth, tags='currentline')
        self.mask[int(x1):int(x2), int(y1):int(y2)] = 1
        if(append):
            self.drawActions.append(('draw', lineWidth, color, x1,y1,x2,y2))
            self.currentAction.append(('draw',lineWidth, color, x1,y1,x2,y2, line))
        self.lastx, self.lasty = x2, y2   
        
    def eraseLine(self,x,y, append=True):
        x1, y1 = x-self.lineWidth/2, y-self.lineWidth/2
        x2, y2 = x+self.lineWidth/2, y+self.lineWidth/2
        # find all items under cursor
        items = self.canvas.find_overlapping(x1, y1, x2, y2)
        # erase items except the background image
        for item in items:
            if item not in [self.canvasImage, self.draw, self.eraseButton]:
                
                dx1, dy1, dx2, dy2 = self.canvas.coords(item)
                if(append):
                    self.drawActions.append(('erase', self.lineWidth, self.color, dx1, dy1, dx2, dy2))
                    self.currentAction.append(("",self.lineWidth, self.color, dx1, dy1, dx2, dy2))
                self.canvas.delete(item)
                #print the cords of the deleted item
          
    def undo(self, event):
        print("undo")
        if(len(self.drawActions) == 0):
            return
        
        lastAction = self.drawActions.pop()
        self.redoAction.append(lastAction)
        
        if(lastAction[0] == 'draw'):
            self.eraseLine(lastAction[5], lastAction[6], False)
            # self.drawActions.pop()
            # self.currentAction.pop()
        elif(lastAction[0] == 'erase'):
            self.drawLine(lastAction[3], lastAction[4], lastAction[5], lastAction[6], lastAction[1], lastAction[2], False)
            # self.drawActions.pop()
            # self.currentAction.pop()
    
    def massUndo(self, event):
        print("mass undo")
        print(len(self.massAction))
        if(len(self.massAction) == 0):
            return
        lastAction = self.massAction.pop()
        self.redoLargeAction.append(lastAction)
        self.currentAction = []
        
        
        if(lastAction[0] == 'draw'):
            for item in lastAction[1]:
                self.eraseLine(item[5], item[6])
        elif(lastAction[0] == 'erase'):
            for item in lastAction[1]:
                self.drawLine(item[3], item[4], item[5], item[6], item[1], item[2])
        
    def redo(self, event):
        print("redo")
        if(len(self.redoAction) == 0):
            return
        
        lastAction = self.redoAction.pop()
        
        if(lastAction[0] == 'erase'):
            self.eraseLine(lastAction[5], lastAction[6])
        elif(lastAction[0] == 'draw'):
            self.drawLine(lastAction[3], lastAction[4], lastAction[5], lastAction[6], lastAction[1], lastAction[2])

            
    def massRedo(self, event):
        print("mass redo")
        if(len(self.redoLargeAction) == 0):
            return
        lastAction = self.redoLargeAction.pop()
        self.massAction.append(lastAction)
        self.currentAction = []
        
        if(lastAction[0] == 'draw'):
            for item in lastAction[1]:
                self.drawLine(item[3], item[4], item[5], item[6], item[1], item[2])
        elif(lastAction[0] == 'erase'):
            for item in lastAction[1]:
                self.eraseLine(item[5], item[6])
            
    def addLine(self, event):
        self.currentx, self.currenty = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        
    def eraseFunction(self,event):
        self.erase = True
        self.color = 'white'
        self.canvas.dtag('palettePen', 'paletteSelected')
        self.canvas.itemconfigure('palettePen', outline='white')
        self.canvas.addtag('paletteSelected', 'withtag', 'palette%s' % self.color)
        self.canvas.itemconfigure('paletteSelected', outline='red')
        
    def changeLineWidth(self,size):
        self.lineWidth = size
        # canvas.itemconfigure('currentline', width=lineWidth)
        self.canvas.dtag('paletteSize', 'paletteSizeSelected')
        self.canvas.itemconfigure('paletteSize', outline='white')
        self.canvas.addtag('paletteSizeSelected', 'withtag', 'palette%s' % self.lineWidth)
        self.canvas.itemconfigure('paletteSizeSelected', outline='blue')  

    def doneStroke(self,event):
        action = "erase" if self.erase else "draw"
        self.massAction.append((action, self.currentAction))
        self.currentAction = []

        
        
if __name__ == "__main__":
    a = vesselEditor("")