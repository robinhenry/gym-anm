import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as Tk
import networkx as nx

root = Tk.Tk()
root.wm_title("Animated Graph embedded in TK")
# Quit when the window is done
root.wm_protocol('WM_DELETE_WINDOW', root.quit)

f = plt.figure(figsize=(5,4))
a = f.add_subplot(111)
plt.axis('off')

# the networkx part
G=nx.complete_graph(5)
pos=nx.circular_layout(G)
nx.draw_networkx(G,pos=pos,ax=a)
xlim=a.get_xlim()
ylim=a.get_ylim()



# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

def next_graph():
    if G.order():
        a.cla()
        G.remove_node(G.nodes()[-1])
        nx.draw_networkx(G, pos, ax=a)
        a.set_xlim(xlim)
        a.set_ylim(ylim)
        plt.axis('off')
        canvas.draw()

b = Tk.Button(root, text="next",command=next_graph)
b.pack()

Tk.mainloop()