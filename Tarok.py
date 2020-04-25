import datetime
from itertools import cycle
from collections import deque
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
            rezultat =next(igra.start())
            #print(rezultat)
            for igralec, pise in rezultat.items():
                self.rezultati[igralec] += pise
        print(self.rezultati)

    def paralel_start(self):
        igre_gen = []
        igre = [Igra(self.igralci,multi_games=True,id=i) for i in range(self.st_iger)]
        for i in igre:
            igre_gen.append(i.start())
        tmp_lst = [next( x ) for x in igre_gen]  # tle mamo zdej dejanske igre(Navadna igra...)
        for i in self.igralci:
            i.predict_licitiram()
        print('Igralci predictar',igre_gen)
        igre_gen = [next( x )for x in igre_gen]
        tmp_lst = [next( x )for x in igre_gen] # tle mamo zdej dejanske igre(Navadna igra...)
        [next( x )for x in igre_gen] # zalozim
        list(map(lambda i:i.predict_izberi_iz_talona(),self.igralci))
        list( map( lambda x: next( x ), igre_gen ) )  # zalozil in prides do glavne zanke
        koncane_igre = [None for i in range(self.st_iger)]
        for _ in range(48):#12 krogov 4 karte gredo gor na krog
            for i in self.igralci:
                i.predict_igraj_karto()

            for id,g in enumerate(igre_gen):
                    if koncane_igre[id] is None:
                        r = next(g)
                        if  isinstance( r,dict ):
                            koncane_igre[id] = r


        for d in koncane_igre:
            for k,v in d.items():
                self.rezultati[k] += v
        print(self.rezultati)