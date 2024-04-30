import tkinter as tk



class Application(tk.Frame):
    def __init__(self, master, width, height):
        tk.Frame.__init__(self, master)
        self.grid(sticky=tk.N + tk.S + tk.E + tk.W)
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)
        self._create_widgets()
        self.bind('<Configure>', self._resize)
        self.winfo_toplevel().minsize(150, 150)

    def _create_widgets(self):
        self.content = tk.Frame(self, bg='blue')
        self.content.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

    def _resize(self, event):
        '''Modify padding when window is resized.'''
        w, h = event.width, event.height
        w1, h1 = self.content.winfo_width(), self.content.winfo_height()
        print(w1, h1)  # should be equal
        if w > h:
            self.rowconfigure(0, weight=1)
            self.rowconfigure(1, weight=0)
            self.columnconfigure(0, weight=h)
            self.columnconfigure(1, weight=w - h)
        elif w < h:
            self.rowconfigure(0, weight=w)
            self.rowconfigure(1, weight=h - w)
            self.columnconfigure(0, weight=1)
            self.columnconfigure(1, weight=0)
        else:
            # width = height
            self.rowconfigure(0, weight=1)
            self.rowconfigure(1, weight=0)
            self.rowconfigure(0, weight=1)
            self.columnconfigure(1, weight=0)

root = tk.Tk()
app = Application(master=root, width=100, height=100)
app.mainloop()