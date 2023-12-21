import tkinter as tk
import numpy as np
import tkinter.filedialog
from os.path import isfile, exists, join
from os import listdir
from glob import glob
from popupYamlConfig import congMenu
from frangiFilter2D import FrangiFilter2D
from skimage.filters import threshold_otsu
import cv2
import multiprocessing
from functools import partial


from PIL import Image, ImageTk

class vesselEditor(tk.Tk):
    def __init__(self, controller):
        super().__init__()
        self.title("Vessel Editior")
        
        #FrangiScaleRange=np.array([1, 10]), FrangiScaleRatio=2,
        #FrangiBetaOne=0.5, FrangiBetaTwo=15, verbose=False, BlackWhite=True
        self.data = {
            "FrangiScaleLowerBound": {
                "Value" : 1,
                "Type" : 'Int',
                "Description" : 'Lower bound of the scale range used in the Frangi Filter',
            },
            "FrangiScaleUpperBound": {
                "Value" : 10,
                "Type" : 'Int',
                "Description" : 'Upper bound of the scale range used in the Frangi Filter',
            },
            "FrangiScaleRatio": {
                "Value" : 2,
                "Type" : 'Int',
                "Description" : 'Step size between sigmas, default 2',
            },
            "FrangiBetaOne": {
                "Value" : 0.5,
                "Type" : 'Float',
                "Description" : 'Frangi correction constant, default 0.5',
            },
            "FrangiBetaTwo": {
                "Value" : 15,
                "Type" : 'Int',
                "Description" : 'Frangi correction constant, default 15',
            },
            "Verbose": {
                "Value" : False,
                "Type" : 'Bool',
                "Description" : 'Print progress to command window',
            },
            "BlackWhite": {
                "Value" : True,
                "Type" : 'Bool',
                "Description" : 'Detect black ridges (default) set to true, for white ridges set to false.',
            },
            "outputLocation": {
                "Value" : "finalImages",
                "Type" : 'Folder',
                "Description" : 'The realitive path to the output folder',
            },
            #Do I want to include this: 
            #Allow storing of the filtter,s could be used as a checkpoint
            #More complicated logic because i need to check another file location to see if it is valid
            #What to do in the case it doesn't exist
            
            "FilterLocation": {
                "Value" : "filterMasks",
                "Type" : 'Folder',
                "Description" : 'Location to the folder where to save the raw masks',
            },
            "openingKeral": {
                "Value" : 3,
                "Type" : 'int',
                "Description" : 'soze of opening kernal',
            },
            "openingiterations": {
                "Value" : 1,
                "Type" : 'Int',
                "Description" : 'Location to the folder where to save the raw masks',
            },
            "closingKeral": {
                "Value" : 3,
                "Type" : 'int',
                "Description" : 'soze of opening kernal',
            },
            "closingiterations": {
                "Value" : 1,
                "Type" : 'Int',
                "Description" : 'Location to the folder where to save the raw masks',
            },
        }
        
        self.currentImage = ""
        self.imageQueue = []
        self.previousImages = []
        

        
        
        self.canvas = canvasEditior(self, imagePath = self.currentImage)
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
        
        self.filterImage_button = tk.Button(self.labelFrame, text="Filter Image", command=self.filterImage)
        self.filterImage_button.pack()
        
        self.filterQueue_button = tk.Button(self.labelFrame, text = "Filter All Images", command = self.filterQueue)
        self.filterQueue_button.pack()
        #
        self.opening_button = tk.Button(self.labelFrame, text="Opening", command=self.opening)
        self.opening_button.pack()
        
        self.closing_button = tk.Button(self.labelFrame, text="Closing", command=self.closing)
        self.closing_button.pack()
        
        
        
        
        self.mainloop()
        
    def loadFolder(self):
        print("load folder")
        folderPath = tk.filedialog.askdirectory(initialdir = "/",
                                            title = "Select Folder containing images")
        
        for file in sorted(glob(f'{folderPath}/*.jpg')):
            self.imageQueue.append(file)
            
        print(self.imageQueue)
        pass
        
    def loadImage(self):
        print("load image")
        
        
        filename = tk.filedialog.askopenfilename(initialdir = "/",
                                            title = "Select a File",
                                            filetypes = (("JPG Files",
                                                        "*.jpg"),
                                                        ("PNG File",
                                                        "*.png")))
        
        self.currentImage = filename
        self.canvas.changeImage(filename)
        pass

    def saveImage(self):
        print("save image")
        mask = self.canvas.returnMask()
        filterMask = self.canvas.filterMask
        #Where do i want to check if the diretory exists, could do it here or when a new folder is selected. Is there any fowenside to doing both besides 
        #run time perfromace hit
        
        imageName = self.currentImage.split("/")[-1]
        imageName = imageName.split(".")[0]
        
        #Check if the directory exists
        if(exists(self.data["outputLocation"]["Value"])):
            filePath = join(self.data["outputLocation"]["Value"], imageName)
            m2 = mask*255
            m2 = np.round(m2)
            cv2.imwrite(filePath + "_mask.jpg", m2)
        else:
            tk.messagebox.showerror('Folder Not found', f'Unable to save image, output folder,{self.data["outputLocation"]["Value"]}, not found')
        
        if(exists(self.data["FilterLocation"]["Value"])):
            filePath = join(self.data["FilterLocation"]["Value"], imageName)
            m2 = filterMask*255
            m2 = np.round(m2)
            cv2.imwrite(filePath + "_filtermask.jpg", m2)
        else:
            tk.messagebox.showerror('Folder Not found', f'Unable to save image, output folder,{self.data["outputLocation"]["Value"]}, not found')
    
    def nextImage(self):
        print("next image")
        
        if(len(self.imageQueue) == 0):
            return
        # self.saveImage()
        self.previousImages.append(self.currentImage)
        self.currentImage = self.imageQueue.pop(0)
        self.canvas.changeImage(self.currentImage)
        
        imageName = self.currentImage.split("/")[-1]
        imageName = imageName.split(".")[0]
        self.canvas.changeImage(self.currentImage)
        
        print(join(self.data["outputLocation"]["Value"], imageName + "_mask.jpg"))
        print(exists(join(self.data["outputLocation"]["Value"], imageName + "_mask.jpg")))
        
        print(join(self.data["FilterLocation"]["Value"], imageName + "_filtermask.jpg"))
        print(exists(join(self.data["FilterLocation"]["Value"], imageName + "_filtermask.jpg")))
        
        if(exists(join(self.data["FilterLocation"]["Value"], imageName + "_filtermask.jpg"))):
            print("filter mask found")
            filterMask = cv2.imread(join(self.data["FilterLocation"]["Value"], imageName + "_filtermask.jpg"))
            filterMask = cv2.cvtColor(filterMask, cv2.COLOR_BGR2GRAY)/255.
            filterMask = np.round(filterMask)
            self.canvas.drawMask(filterMask)
        
        if(exists(join(self.data["outputLocation"]["Value"], imageName + "_mask.jpg"))):
            print("mask found")
            mask = cv2.imread(join(self.data["outputLocation"]["Value"], imageName + "_mask.jpg"))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)/255.
            mask = np.round(mask)
            self.canvas.drawMask(mask)
    
    def prevoiusImage(self):
        print("previous image")
        # self.saveImage()
        self.imageQueue.insert(0, self.currentImage)
        self.currentImage = self.previousImages.pop()
        
        #Check if there is an existing mask for the image
        imageName = self.currentImage.split("/")[-1]
        imageName = imageName.split(".")[0]
        self.canvas.changeImage(self.currentImage)
        
        print(join(self.data["outputLocation"]["Value"], imageName + "_mask.jpg"))
        print(exists(join(self.data["outputLocation"]["Value"], imageName + "_mask.jpg")))
        
        print(join(self.data["FilterLocation"]["Value"], imageName + "_filtermask.jpg"))
        print(exists(join(self.data["FilterLocation"]["Value"], imageName + "_filtermask.jpg")))
        
        if(exists(join(self.data["FilterLocation"]["Value"], imageName + "_filtermask.jpg"))):
            print("filter mask found")
            filterMask = cv2.imread(join(self.data["FilterLocation"]["Value"], imageName + "_filtermask.jpg"))
            filterMask = cv2.cvtColor(filterMask, cv2.COLOR_BGR2GRAY)/255
            filterMask = np.round(filterMask)
            self.canvas.drawMask(filterMask)
        
        if(exists(join(self.data["outputLocation"]["Value"], imageName + "_mask.jpg"))):
            print("mask found")
            mask = cv2.imread(join(self.data["outputLocation"]["Value"], imageName + "_mask.jpg"))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)/255
            mask = np.round(mask)
            self.canvas.drawMask(mask)
        
        
        pass
    
    def parameters(self):
        print("parameters")
        congMenu(self, self.data)
        
    def filterImage(self):
        print("filter image")
        image = cv2.imread(self.currentImage)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.
        
        FrangiScaleRange = (self.data["FrangiScaleLowerBound"].get("Value", 1), self.data["FrangiScaleUpperBound"].get("Value", 1))
        FrangiScaleRatio = self.data["FrangiScaleRatio"].get("Value", 2)
        FrangiBetaOne = self.data["FrangiBetaOne"].get("Value", 0.5)
        FrangiBetaTwo = self.data["FrangiBetaTwo"].get("Value", 15)
        verbose = self.data["Verbose"].get("Value", False)
        BlackWhite = self.data["BlackWhite"].get("Value", True)
        
        
        mask = FrangiFilter2D(gray,FrangiScaleRange=FrangiScaleRange, FrangiScaleRatio=FrangiScaleRatio,
                              FrangiBetaOne=FrangiBetaOne, FrangiBetaTwo=FrangiBetaTwo,
                              verbose=verbose, BlackWhite=BlackWhite)
        
        #Get the base name of the image file
        imageName = self.currentImage.split("/")[-1]
        imageName = imageName.split(".")[0]
        
        grayout = cv2.normalize(mask[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

        # Otsu's thresholding

        ret2,th2 = cv2.threshold(grayout,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # manual threshold

        ret2,th2 = cv2.threshold(grayout,8,255,cv2.THRESH_BINARY)
     
        m2 = th2 == 255
        
        
        if(exists(self.data["FilterLocation"]["Value"])):
            #Output the mask
            filePath = join(self.data["FilterLocation"]["Value"], imageName)
            cv2.imwrite(filePath + "_filtermask.jpg", m2*255)
        
        self.canvas.drawMask(m2)
        print("success")

    def filterQueue(self):
        print("filter queue")
        
        filterLocation = self.data["FilterLocation"].get("Value", "filterMasks")
        FrangiScaleRange = (self.data["FrangiScaleLowerBound"].get("Value", 1), self.data["FrangiScaleUpperBound"].get("Value", 1))
        FrangiScaleRatio = self.data["FrangiScaleRatio"].get("Value", 2)
        FrangiBetaOne = self.data["FrangiBetaOne"].get("Value", 0.5)
        FrangiBetaTwo = self.data["FrangiBetaTwo"].get("Value", 15)
        verbose = self.data["Verbose"].get("Value", False)
        BlackWhite = self.data["BlackWhite"].get("Value", True)
        
        pool = multiprocessing.Pool()
        #mappedResults = pool.map(partial(mergeController, debug=debug, **kwargs), data.items())
        mappedResults = pool.map(partial(filter,filterloaction = filterLocation, FrangiScaleRange=FrangiScaleRange,
                                         FrangiScaleRatio=FrangiScaleRatio, FrangiBetaOne=FrangiBetaOne, 
                                         FrangiBetaTwo=FrangiBetaTwo, verbose=verbose, BlackWhite=BlackWhite), self.imageQueue)
        
        
            
        pass
    
    def opening(self):
        mask = self.canvas.returnMask()
        
        kernalSize = self.data["openingKeral"].get("Value", 3)
        iterations = self.data["openingiterations"].get("Value", 1)
        
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernalSize, kernalSize))
        opening = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernal, iterations=iterations)
        
        self.canvas.drawMask(opening, createFilterMask=False)
        
    def closing(self):
        mask = self.canvas.returnMask()
        
        kernalSize = self.data["closingKeral"].get("Value", 3)
        iterations = self.data["closingiterations"].get("Value", 1)
        
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernalSize, kernalSize))
        closing = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernal, iterations=iterations)
        
        self.canvas.drawMask(closing, createFilterMask=False)
    
def filter(imagePath, **kwargs):
    

    FrangiScaleRange = kwargs.get("FrangiScaleRange", (1,10))
    FrangiScaleRatio = kwargs.get("FrangiScaleRatio", 2)
    FrangiBetaOne = kwargs.get("FrangiBetaOne", 0.5)
    FrangiBetaTwo = kwargs.get("FrangiBetaTwo", 15)
    verbose = kwargs.get("verbose", False)
    BlackWhite = kwargs.get("BlackWhite", True)
    filterlocation = kwargs.get("filterloaction", "filterMasks")

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.

    mask = FrangiFilter2D(gray,FrangiScaleRange=FrangiScaleRange, FrangiScaleRatio=FrangiScaleRatio,
                            FrangiBetaOne=FrangiBetaOne, FrangiBetaTwo=FrangiBetaTwo,
                            verbose=verbose, BlackWhite=BlackWhite)

    grayout = cv2.normalize(mask[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    ret2,th2 = cv2.threshold(grayout,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # manual threshold

    # ret2,th2 = cv2.threshold(grayout,8,255,cv2.THRESH_BINARY)

    m2 = th2 == 255

    imageName = imagePath.split("/")[-1]
    imageName = imageName.split(".")[0]

    if(exists(filterlocation)):
        #Output the mask
        filePath = join(filterlocation, imageName)
        cv2.imwrite(filePath + "_filtermask.jpg", m2*255)
        
    return m2
        


class canvasEditior(tk.Frame):
    def __init__(self, parent, imagePath = "FrangiFilter/0002.jpg"):
        super().__init__(parent)
        
        
        self.erase = False
            
        if(isfile(imagePath)):
            self.image = Image.open(imagePath)
            self.imageObject = ImageTk.PhotoImage(self.image)
            
            width = self.imageObject.width()
            height = self.imageObject.height()
        else:
            self.image = Image.new("RGB", (512,512), "black")
            self.imageObject = ImageTk.PhotoImage(self.image)
            
            width = 512
            height = 512
            
        self.canvas = tk.Canvas(parent, width = width, height = width)
        self.canvasImage = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imageObject)
            
    

        self.mask = np.zeros((self.imageObject.width(), self.imageObject.height()))
        self.filterMask = np.zeros((self.imageObject.width(), self.imageObject.height()))

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

        self.smallBrush = self.canvas.create_rectangle((10, 85, 30, 105), fill="black", tags=('paletteSize', 'palette1', 'paletteSizeSelected'))
        self.canvas.tag_bind(self.smallBrush , "<Button-1>", lambda x:self.changeLineWidth(1))
        self.medBrush  = self.canvas.create_rectangle((10, 110, 30, 130), fill="black", tags=('paletteSize', 'palette5'))
        self.canvas.tag_bind(self.medBrush, "<Button-1>", lambda x: self.changeLineWidth(5))
        self.largeBrush = self.canvas.create_rectangle((10, 135, 30, 155), fill="black", tags=('paletteSize', 'palette10'))
        self.canvas.tag_bind(self.largeBrush, "<Button-1>", lambda x: self.changeLineWidth(10))
        
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

        prevX, prevY = int(x1), int(y1)
        endX, endY = int(x2), int(y2)
        
        minX = min(prevX, endX)
        maxX = max(prevX, endX)
        minY = min(prevY, endY)
        maxY = max(prevY, endY)
        print(prevX, prevY, endX, endY)
        lineRange = lineWidth//2
        
        if(prevX == endX):
            for i in range(minY, maxY):
                dx1, dy1, dx2, dy2 = endX - lineRange, i-lineRange , endX+lineRange, i+lineRange
                #item = self.canvas.create_line(endX, i , endX+1, i+1, fill=color, width=lineWidth)
                item = self.canvas.create_rectangle(dx1, dy1, dx2, dy2, fill=color, outline=color, width=0)
                self.mask[int(dy1):int(dy2), int(dx1):int(dx2)] = 1
                if(append):
                    self.drawActions.append(('draw', lineWidth, color,dx1, dy1, dx2, dy2))
                    self.currentAction.append(('draw',lineWidth, color, dx1, dy1, dx2, dy2, item))
            pass
        elif(prevY == endY):
            for i in range(minX, maxX):
                dx1, dy1, dx2, dy2 = i-lineRange, endY - lineRange, i+lineRange, endY+lineRange
                # item = self.canvas.create_line(i, endY, i+1, endY+1,fill=color, width=lineWidth)
                item = self.canvas.create_rectangle(dx1, dy1, dx2, dy2,fill=color, outline=color, width=0)
                self.mask[int(dy1):int(dy2), int(dx1):int(dx2)] = 1
                if(append):
                    self.drawActions.append(('draw', lineWidth, color, dx1, dy1, dx2, dy2))
                    self.currentAction.append(('draw',lineWidth, color, dx1, dy1, dx2, dy2, item))
            pass
        else:
            slope = (endY - prevY) / (endX - prevX)
            b = prevY - slope * prevX
            
            if(abs(slope) > 1):
                points = [(round((newY - b)/slope), newY) for newY in range(minY, maxY+1)]
            else:
                points = [(newX,round(newX * slope + b)) for newX in range(minX, maxX+1)]
                
            for newX, newY in points:
                dx1, dy1, dx2, dy2 = newX- lineRange, newY- lineRange, newX + lineRange , newY + lineRange
                #item = self.canvas.create_line(prevX - lineRange, prevY - lineRange, newX + lineRange, newY + lineRange,fill=color, width=lineWidth)
                item = self.canvas.create_rectangle(dx1, dy1, dx2, dy2,fill=color, width=0, outline=color)
                self.mask[ int(dy1):int(dy2), int(dx1):int(dx2)] = 1
                if(append):
                    self.drawActions.append(('draw', lineWidth, color,dx1, dy1, dx2, dy2))
                    self.currentAction.append(('draw',lineWidth, color, dx1, dy1, dx2, dy2, item))
                    
                prevX = newX
                prevY = newY
                self.lastx, self.lasty = prevX, prevY
            
            
        self.lastx, self.lasty = endX, endY   
        
    def eraseLine(self,x,y, append=True):
        x1, y1 = x-self.lineWidth/2, y-self.lineWidth/2
        x2, y2 = x+self.lineWidth/2, y+self.lineWidth/2
        # find all items under cursor
        items = self.canvas.find_overlapping(x1, y1, x2, y2)
        # erase items except the background image
        for item in items:
            if item not in [self.canvasImage, self.draw, self.eraseButton, self.smallBrush, self.medBrush, self.largeBrush]:
                
                dx1, dy1, dx2, dy2 = self.canvas.coords(item)
                self.mask[int(dy1):int(dy2), int(dx1):int(dx2)] = 0
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
        
        #Edge case of mass undoing when there is nothing to undo
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
        
        #Edge case of mass redoing when there is nothing to undo
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
        
    def drawMask(self, mask, createFilterMask = True):
        self.mask = mask.copy()
        if(createFilterMask):
            self.filterMask = mask.copy()
        #Clear the canvas
        for item in self.canvas.find_all():
            if item not in [self.canvasImage, self.draw, self.eraseButton, self.smallBrush, self.medBrush, self.largeBrush]:
                self.canvas.delete(item)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if(mask[i,j] == 1):
                    x, y = self.canvas.canvasx(j), self.canvas.canvasy(i)
                    self.canvas.create_rectangle(x, y, x+1, y+1, fill="yellow", outline="yellow", width=0)
        
    def eraseFunction(self,event):
        self.erase = True
        self.color = 'white'
        self.canvas.dtag('palettePen', 'paletteSelected')
        self.canvas.itemconfigure('palettePen', outline='gray')
        self.canvas.addtag('paletteSelected', 'withtag', 'paletteErase')
        self.canvas.itemconfigure('paletteSelected', outline='red')
        
    def changeLineWidth(self,size):
        self.lineWidth = size
        # canvas.itemconfigure('currentline', width=lineWidth)
        self.canvas.dtag('paletteSize', 'paletteSizeSelected')
        self.canvas.itemconfigure('paletteSize', outline='gray')
        self.canvas.addtag('paletteSizeSelected', 'withtag', 'palette%s' % self.lineWidth)
        self.canvas.itemconfigure('paletteSizeSelected', outline='blue')  

    def doneStroke(self,event):
        action = "erase" if self.erase else "draw"
        self.massAction.append((action, self.currentAction))
        self.currentAction = []
        
    def changeImage(self, imagePath):
        self.image = Image.open(imagePath)
        self.imageObject = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfigure(self.canvasImage, image=self.imageObject)
        
    def returnMask(self):
        return self.mask

        
        
if __name__ == "__main__":
    a = vesselEditor("")