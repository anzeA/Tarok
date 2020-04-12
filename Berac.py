from Klop import Klop


class Berac(Klop):
    def __init__(self,igralci,berac,talon,odprti):
        self.igralci = igralci
        self.berac = berac
        self.talon = talon
        self.odprti = odprti
        self.zgodovina = []


    def start(self):
        print('Berac')
        zacne = self.igralci.index(self.berac)
        vrednost_igre = 70
        if self.odprti:
            for i in self.igralci:
                if i != self.berac:
                    i.poglej_karte_odprtega_beraca(self.berac.roka)

            vrednost_igre = 90
        pisejo = {i:0 for i in self.igralci}
        for i in range(12):
            zmaga = self.krog(zacne)
            zacne += zmaga
            zacne = zacne%4
            if self.igralci[zacne] == self.berac:
                pisejo[self.berac] = -vrednost_igre
                return pisejo
        pisejo[self.berac] = vrednost_igre
        return pisejo