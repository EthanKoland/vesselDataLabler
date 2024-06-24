import tkinter as tk
import math

class WhiteboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whiteboard")

        self.canvas = tk.Canvas(root, width=300, height=300, bg="white")
        self.canvas.pack()

        self.draw_button = tk.Button(root, text="Draw", command=self.start_draw)
        self.draw_button.pack(side=tk.LEFT)

        self.erase_button = tk.Button(root, text="Erase", command=self.start_erase)
        self.erase_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.drawing = False
        self.erasing = False
        self.last_x, self.last_y = None, None

        self.canvas.bind("<Button-1>", self.start_action)
        self.canvas.bind("<B1-Motion>", self.draw_or_erase)
        self.canvas.bind("<ButtonRelease-1>", self.stop_action)

    def start_action(self, event):
        if self.drawing:
            self.start_draw(event)
        elif self.erasing:
            self.start_erase(event)

    def stop_action(self, event):
        if self.drawing:
            self.stop_draw(event)
        elif self.erasing:
            self.stop_erase(event)

    def start_draw(self, event=None):
        self.drawing = True
        self.erasing = False
        if event:
            self.last_x, self.last_y = event.x, event.y

    def stop_draw(self, event=None):
        self.drawing = False
        self.erasing = False

    def start_erase(self, event=None):
        self.erasing = True
        self.drawing = False
        if event:
            self.last_x, self.last_y = event.x, event.y

    def stop_erase(self, event=None):
        self.erasing = False
        self.drawing = False

    def draw_or_erase(self, event):
        if self.drawing:
            prevX, prevY = int(self.last_x), int(self.last_y)
            endX, endY = event.x, event.y
            print(prevX, prevY, endX, endY)
            if(prevX == endX):
                minY = min(prevY, endY)
                maxY = max(prevY, endY)
                for i in range(minY, maxY):
                    self.canvas.create_line(endX, i , endX+1, i+1, fill="black")
                pass
            elif(prevY == endY):
                minX = min(prevX, endX)
                maxX = max(prevX, endX)
                for i in range(minX, maxX):
                    self.canvas.create_line(i, endY, i+1, endY+1, fill="black")
                pass
            else:
                slope = (endY - prevY) / (endX - prevX)
                b = prevY - slope * prevX
                minX = min(prevX, endX)
                maxX = max(prevX, endX)
                # for i in range(minX, maxX):
                #     newY = slope + prevY
                #     self.canvas.create_line(i, prevY, i+1, newY, fill="black")
                #     prevY = newY
                points = [(newX,round(newX * slope + b)) for newX in range(minX, maxX+1)]
                for newX, newY in points:
                    self.canvas.create_line(prevX, prevY, newX, newY, fill="black", width=1)
                    prevX = newX
                    prevY = newY
                    self.last_x, self.last_y = prevX, prevY
                
                
            self.last_x, self.last_y = endX, endY
        elif self.erasing:
            x, y = event.x, event.y
            self.canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5, fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    app = WhiteboardApp(root)
    root.mainloop()
