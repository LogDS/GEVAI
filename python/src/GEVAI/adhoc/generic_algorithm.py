

class GenericAlgorithm:
    def __init__(self, n=1):
        self.n = n

    def initQueue(self):
        return []

    def rho(self, h, f):
        return []

    def Quality(self, h, f):
        return False

    def stop(self, Queue, Theory):
        return True

    def Prune(self, Queue, Theory, f):
        return Queue, Theory, f

    def __call__(self, *args, **kwargs):
        """
        Implementation of a generic learning algorithm
        :param args:
        :param kwargs:
        :return:
        """
        f = args[0]
        Queue = self.initQueue()
        Theory = []
        while not self.stop(Queue, Theory):
            L = Queue[:self.n]  # extracting the first n elements
            del Queue[:self.n]  # removing n elements
            for h in L:
                if self.Quality(h, f):
                    Theory.append(h)
                    Queue = Queue + [x for x in self.rho(h, f)]
            Queue, Theory, f = self.Prune(Queue, Theory, f)
        return Theory
