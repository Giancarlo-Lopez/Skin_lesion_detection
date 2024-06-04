import tkinter as tk
from my_package.gui import GUI

if __name__ == '__main__':
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
