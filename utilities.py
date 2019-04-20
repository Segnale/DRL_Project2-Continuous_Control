import matplotlib.pyplot as plt

class Plotting():
    def __init__(self, title, y_label = 'x axe', x_label = 'x_axe', x_range = None, fig_size = [None,None], x_values =[], y_values = [], show = False):
        self.title = title
        self.y_label = y_label
        self.x_label = x_label
        self.x_range = x_range
        self.fig_size = fig_size
        self.x_values = x_values
        self.y_values = y_values
        self.plt = plt
        # Initialize and show the plot
        if None in fig_size:
            #self.fig, self.ax = plt.subplots()
            self.fig = self.plt.figure()
        else:
            self.fig = self.plt.figure(figsize=self.fig_size)
        self.fig.canvas.set_window_title(title) 
        self.ax = self.fig.add_subplot(111)
        self.plt.title(self.title)
        self.ax.plot(self.x_values, self.y_values,'c-')
        self.plt.ylabel(self.y_label)
        self.plt.xlabel(self.x_label)

        if x_range != None:
            self.set_x_range(0, x_range)
        if show:
            self.plt.draw()
            self.plt.pause(0.001)
         
    def Update(self, x_in, y_in):
        if len(x_in) > self.x_range:
            self.x_values = x_in[-self.x_range:]
            self.y_values = y_in[-self.x_range:]
            self.set_x_range(self.x_values[0], self.x_values[-1])
        else:
            self.x_values = x_in
            self.y_values = y_in

        self.ax.clear
        self.ax.plot(self.x_values, self.y_values,'c-')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def show(self):
        self.plt.draw()
        self.plt.pause(0.001)

    def set_x_range(self, x_min, x_max):
        self.ax.set_xlim(x_min, x_max)

    def save(self, direction):
        self.plt.savefig(direction)

    def legend(self, text,x,y):
        self.plt.text(x, y, text, ha='left')