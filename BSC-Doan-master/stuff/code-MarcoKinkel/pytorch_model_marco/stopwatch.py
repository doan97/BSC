import time

class Stopwatch(object):
    def __init__(self):
        self.categories = {}
        self.start('__TOTAL')

    def start(self, c):
        if c in self.categories:
            self.categories[c]['start'] = time.time()

        else:
            # Create
            self.categories[c] = {'sum': 0.0, 'start': time.time()}

    def stop(self, c):

        if (c not in self.categories) or ('start' not in self.categories[c]):
            print('Timewatch for category', c, 'has not been started')
            return

        self.categories[c]['sum'] += time.time() - self.categories[c]['start']
        self.categories[c].pop('start')

        return self.categories[c]['sum']

    def summary(self):
        self.stop('__TOTAL')
        total_sum = self.categories.pop('__TOTAL')['sum']

        for c in self.categories:
            
            if 'start' in self.categories[c]:
                self.stop(c)
            
            sum = self.categories[c]['sum']
            ratio = (sum * 100) / total_sum
            
            print(c, '|', "{0:0.2f}".format(sum), 's (', "{0:0.2f}".format(ratio), '%)')

        print('Total time', "{0:0.2f}".format(total_sum), 's')
