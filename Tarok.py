import datetime
from itertools import cycle

from Igra import Igra


class Tarok:
    def __init__(self, igralci,st_iger = None):
        assert  len({i.ime for i in igralci}) == 4
        self.igralci = igralci
        self.rezultati = {i: 0 for i in igralci}
        self.radelci = {i: 0 for i in igralci}
        self.st_iger = st_iger

    def stream(self):
        if self.st_iger is None:
            for i in  cycle([0,1,2,3]):
                yield i
        else:
            for i in range(self.st_iger):
                yield i

    def start(self):
        for i in self.stream():
            b = self.igralci[i%4:]+ self.igralci[:i%4]
            igra = Igra(b)
            rezultat =igra.start()
            #print(rezultat)
            for igralec, pise in rezultat.items():
                self.rezultati[igralec] += pise
        print(self.rezultati)
