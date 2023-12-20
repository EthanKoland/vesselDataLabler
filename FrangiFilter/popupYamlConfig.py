# This will import all the widgets
# and modules which are available in
# tkinter and ttk module
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog

 
# class NewWindow(Toplevel):
     
#     def __init__(self, master = None):
         
#         super().__init__(master = master)
#         self.title("New Window")
#         self.geometry("200x200")
#         label = Label(self, text ="This is a new Window")
#         label.pack()
        
class congMenu(Toplevel):
    def __init__(self, parent = None,  data = None):
            
        super().__init__(parent)
        self.title("New Window")
        # self.geometry("200x200")
        # self.titleLabel = Label(self, text="Yaml Config", font=("Arial", 20))
        # self.titleLabel.grid(column=0, row=0, sticky=W + E)
        
        self.data = data    
        
        if(data != None):
            de = dataEnties(self, self.data)
        else:
            self.noDataLabel = Label(self, text="No Data", font=("Arial", 20))
            self.noDataLabel.grid(column=0, row=1, sticky=W + E)
        
        # self.pack()
        
    def close(self):
        self.destroy()
            
        
        
        
class dataEnties(Frame):
    def __init__(self, parent, data):
        super().__init__(parent)
        
        allChars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-+'
        floatChars = '0123456789.'
        intChars = '0123456789'
        
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.titleLabel = Label(self, text="Yaml Config", font=("Arial", 20))
        self.titleLabel.grid(column=0, row=0, sticky=W + E, columnspan=2)
        self.tkVars = {}
        i = 1
        print(data)
        
        self.data = data
        self.parent = parent
        
        
        for key, value in self.data.items():
            itemValue, itemType = value["Value"], value["Type"]
            print(itemValue, itemType)
            upperItemType = itemType.upper()
            
            if(upperItemType == 'BOOLEAN' or upperItemType == 'BOOL'):
                self.tkVars[key] = BooleanVar(value=itemValue)
                item = TFdropdown(self, key, self.tkVars[key])
                item.grid(column=0, row=i, sticky=W + E, columnspan=2)
            elif(upperItemType == 'INT'):
                self.tkVars[key] = IntVar(value=itemValue)
                item = dataEntry(self, key, self.tkVars[key], itemType, intChars)
                item.grid(column=0, row=i, sticky=W + E, columnspan=2)
            elif(upperItemType == 'FLOAT'):
                self.tkVars[key] = DoubleVar(value=itemValue)
                item = dataEntry(self, key, self.tkVars[key], itemType, floatChars)
                item.grid(column=0, row=i, sticky=W + E, columnspan=2)
            elif(upperItemType == 'FOLDER'):
                self.tkVars[key] = StringVar(value=itemValue)
                item = FolderSelector(self, key, self.tkVars[key], itemType, floatChars)
                item.grid(column=0, row=i, sticky=W + E, columnspan=2)
            else:
                self.tkVars[key] = StringVar(value=itemValue)
                item = dataEntry(self, key, self.tkVars[key], itemType, allChars)
                item.grid(column=0, row=i, sticky=W + E, columnspan=2)
                
            i += 1
        
        self.updateButton = Button(self, text="Update")
        self.updateButton.configure(command=lambda: print(self.retreiveData()))
        self.updateButton.grid(column=0, row=i + 1, sticky=W + E, columnspan=2)
        
        self.closeButton = Button(self, text="Close")
        self.closeButton.configure(command=lambda: self.parent.destroy())
        self.closeButton.grid(column=0, row=i + 2, sticky=W + E, columnspan=2)
        
        self.pack()
        
    def retreiveData(self):
        
        for key, value in self.tkVars.items():
            self.data[key]["Value"] = value.get()
        
        print(self.data)
    

    
class dataEntry(Frame):
    def __init__(self, parent, label, value, itemType, validChars = '0123456789.-+'):
        super().__init__(parent)
        self.label = Label(self, text=label)
        self.value = value
        self.validChars = validChars
        self.itemType = itemType
        
        print(type(self.value))
        print(self.value.get())
        
        self.vcmd = (self.register(self.validate),
                '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        
        self.valueEntry = Entry(self, font = ('calibre',10,'normal'), textvariable=self.value)
        self.valueEntry.configure(validate = 'key', validatecommand = self.vcmd)
        # self.valueEntry.insert(0, self.value.get())
        # self.valueEntry.bind("<FocusOut>", print("Focus Out"))
        
        self.label.grid(column=0, row=0, sticky=W + E)
        self.valueEntry.grid(column=1, row=0, sticky=W + E)
        
 
        
    def update(self, t):
        print(self.valueEntry.get(), self.value.get())
        print("update", self.label.cget("text"))
        self.value = self.valueEntry.get()
        
    def validate(self, action, index, value_if_allowed, prior_value, text, validation_type, trigger_type, widget_name):
    # action=1 -> insert

        if(action=='1'):
            if text in self.validChars:
                print(text, self.validChars)
                try:
                    if(self.itemType == 'Int'):
                        int(value_if_allowed)
                    elif(self.itemType == 'Float'):
                        float(value_if_allowed)
                    return True
                except ValueError:
                    return False
            else:
                return False
        else:
            return True
        
class TFdropdown(Frame):
    def __init__(self, parent, label, value):
        super().__init__(parent)
        self.options = ["True", "False"]
        
        self.label = Label(self, text=label)
        self.value = value
        self.localValue = StringVar(value = "True" if self.value.get() else "False")
        
        self.updateValue = lambda x: self.value.set(True) if x == "True" else self.value.set(False)
        
        conversionLambda = lambda x: "True" if x  else "False"
        
        self.dropdown = OptionMenu(self, self.localValue, self.localValue.get(), *self.options, command=self.updateValue)
        #self.dropdown.configure(command=lambda x: print(x))
        
        self.label.grid(column=0, row=0, sticky=W + E)
        self.dropdown.grid(column=1, row=0, sticky=W + E)
        
    def updateValue(self, *args):
        print("update")
        self.label.configure(text=self.localValue.get())
        self.value.set(self.localValue.get())
        
class FolderSelector(Frame):
    def __init__(self, parent, label, value, itemType, validChars = '0123456789.-+'):
        super().__init__(parent)
        self.cleanedLabel = value.get().split("/")[-1]
        self.label = Label(self, text=label )
        self.value = value
        self.validChars = validChars
        self.itemType = itemType
        
        self.valueEntry = Button(self, text=self.cleanedLabel, command=self.selectFolder)
        # self.valueEntry.insert(0, self.value.get())
        # self.valueEntry.bind("<FocusOut>", print("Focus Out"))
        
        self.label.grid(column=0, row=0, sticky=W + E)
        self.valueEntry.grid(column=1, row=0, sticky=W + E)
        
    def selectFolder(self):
        self.value.set(filedialog.askdirectory())
        self.valueEntry.configure(text=self.value.get().split("/")[-1])
        
    def update(self, t):
        print(self.valueEntry.get(), self.value.get())
        print("update", self.label.cget("text"))
        self.value = self.valueEntry.get()
        
     
        
    
        
# class errorWindow(Toplevel):
#     def __init__(self, master = None, message = None):
            
#         super().__init__(master = master)
#         self.title("Error")
#         self.geometry("200x200")
#         label = Label(self, text = message)
#         label.pack()
        
def loadYamlFile(file):
    t = {}
    with open(file, 'r') as stream:
        try:
            t = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    return t
                
def updateYamlFile(file, data):
    with open(file, 'w') as stream:
        try:
            yaml.dump(data, stream)
        except yaml.YAMLError as exc:
            print(exc)
            
            
if(__name__ == "__main__"):

    # creates a Tk() object
    master = Tk()
    
    # sets the geometry of 
    # main root window
    master.geometry("200x200")
    
    label = Label(master, text ="This is the main window")
    label.pack(side = TOP, pady = 10)
    
    # a button widget which will
    # open a new window on button click
    btn = Button(master, 
                text ="Click to open a new window")
    
    data = data = {
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
            "OutputFolder":{
                "Value" : "output",
                "Type" : 'Folder',
                "Description" : 'Output folder for the images',
            }
        }
    
    # Following line will bind click event
    # On any click left / right button
    # of mouse a new window will be opened
    btn.bind("<Button>", 
            lambda e: congMenu(master, data))
    
    btn.pack(pady = 10)
    
    # mainloop, runs infinitely
    mainloop()