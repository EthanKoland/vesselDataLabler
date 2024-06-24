from tkinter import *
from tkinter import ttk
import numpy as np

from PIL import Image, ImageTk


root = Tk()

# h = ttk.Scrollbar(root, orient=HORIZONTAL)
# v = ttk.Scrollbar(root, orient=VERTICAL)
image = Image.open('FrangiFilter/0002.jpg')
python_image = ImageTk.PhotoImage(image)
# img = PhotoImage(file="FrangiFilter/0002.jpg")

mask = np.zeros((python_image.width(), python_image.height()))

print(python_image.width(), python_image.height())
canvas = Canvas(root, width = python_image.width(), height = python_image.height())
canvas.create_image(0, 0, anchor=NW, image=python_image)
# h['command'] = canvas.xview
# v['command'] = canvas.yview



canvas.grid(column=0, row=0, sticky=(N,W,E,S))
# h.grid(column=0, row=1, sticky=(W,E))
# v.grid(column=1, row=0, sticky=(N,S))
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

lastx, lasty = 0, 0

lineWidth = 5

def xy(event):
    global lastx, lasty
    lastx, lasty = canvas.canvasx(event.x), canvas.canvasy(event.y)

def setColor(newcolor):
    global color
    color = newcolor
    canvas.dtag('palettePen', 'paletteSelected')
    canvas.itemconfigure('palettePen', outline='white')
    canvas.addtag('paletteSelected', 'withtag', 'palette%s' % color)
    canvas.itemconfigure('paletteSelected', outline='#999999')
    

def addLine(event):
    global lastx, lasty, mask
    x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
    canvas.create_line((lastx, lasty, x, y), fill=color, width=lineWidth, tags='currentline')
    mask[int(lastx):int(x), int(lasty):int(y)] = 1
    lastx, lasty = x, y
    
def eraseLine(event):
    global lastx, lasty
    x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
    
    x1, y1 = x-lineWidth/2, y-lineWidth/2
    x2, y2 = x+lineWidth/2, y+lineWidth/2
    # find all items under cursor
    items = canvas.find_overlapping(x1, y1, x2, y2)
    # erase items except the background image
    for item in items:
        if item != python_image:
            canvas.delete(item)
    
def changeLineWidth(size):
    global lineWidth
    lineWidth = size
    # canvas.itemconfigure('currentline', width=lineWidth)
    canvas.dtag('paletteSize', 'paletteSelected')
    canvas.itemconfigure('paletteSize', outline='white')
    canvas.addtag('paletteSelected', 'withtag', 'palette%s' % lineWidth)
    canvas.itemconfigure('paletteSelected', outline='#999999')  

def doneStroke(event):
    canvas.itemconfigure('currentline', width=1)
        
        
canvas.bind("<Button-1>", xy)
canvas.bind("<B1-Motion>", addLine)
# canvas.bind("<B1-ButtonRelease>", doneStroke)

id = canvas.create_rectangle((10, 10, 30, 30), fill="red", tags=('palettePen', 'palettered'))
canvas.tag_bind(id, "<Button-1>", lambda x: setColor("red"))
id = canvas.create_rectangle((10, 35, 30, 55), fill="blue", tags=('palettePen', 'paletteblue'))
canvas.tag_bind(id, "<Button-1>", lambda x: setColor("blue"))
id = canvas.create_rectangle((10, 60, 30, 80), fill="black", tags=('palettePen', 'paletteblack', 'paletteSelected'))
canvas.tag_bind(id, "<Button-1>", lambda x: eraseLine(x))

id = canvas.create_rectangle((10, 85, 30, 105), fill="black", tags=('paletteSize', 'palette1', 'paletteSelected'))
canvas.tag_bind(id, "<Button-1>", lambda x: changeLineWidth(1))
id = canvas.create_rectangle((10, 110, 30, 130), fill="black", tags=('paletteSize', 'palette5'))
canvas.tag_bind(id, "<Button-1>", lambda x: changeLineWidth(5))
id = canvas.create_rectangle((10, 135, 30, 155), fill="black", tags=('paletteSize', 'palette10'))
canvas.tag_bind(id, "<Button-1>", lambda x: changeLineWidth(10))



setColor('black')
canvas.itemconfigure('palette', width=lineWidth)
root.mainloop()