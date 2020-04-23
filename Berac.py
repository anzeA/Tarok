from Klop import Klop


class Berac(Klop):
    def __init__(self,igralci,berac,talon,odprti):
        self.igralci = igralci
        self.berac = berac
        self.talon = talon
        self.odprti = odprti
        self.zgodovina = []


    def start(self):
        #print('Berac')
        zacne = self.igralci.index(self.berac)
        vrednost_igre = 70
        if self.odprti:
            vrednost_igre = 90
        pisejo = {i:0 for i in self.igralci}
        for i in range(12):
            if self.odprti and i == 1:
                for i in self.igralci:
                    if i != self.berac:
                        i.poglej_karte_odprtega_beraca( self.berac.roka )

            zmaga = self.krog(zacne,False)
            zacne += zmaga
            zacne = zacne%4
            if self.igralci[zacne] == self.berac:
                pisejo[self.berac] = -vrednost_igre
                for i in self.igralci:
                    i.rezultat_igre( pisejo.setdefault( i, 0 ), self.zgodovina )
                return pisejo
        pisejo[self.berac] = vrednost_igre

        for i in self.igralci:
            i.rezultat_igre(pisejo.setdefault(i,0),self.zgodovina)
        return pisejo