import sys

class ShowProcess:
    i = 0
    max_steps = 0
    max_arrow = 50
    visible = True

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    def show_process(self,i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        if self.visible is False:
            return
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        
    def close(self):
        self.i = 0
        if self.visible:
            print("")