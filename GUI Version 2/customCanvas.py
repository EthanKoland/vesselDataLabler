import tkinter as tk

class App(tk.Tk):
    def __init__(self, controller):
        super().__init__()
        
        self.canvasController = canvasController(self)
        self.canvasController.pack(fill="both", expand=True)
        
        self.mainloop()
        
        
class canvasController(tk.Frame):
    def __init__(self, master, initalSize = (512,512)):
        super().__init__()
        self.initalSize = initalSize
        
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(self, bg="red")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        
if(__name__ == "__main__"):
    app = App(None)
       