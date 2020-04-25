from Klop import Klop


class Berac(Klop):
    def __init__(self,igralci,berac,talon,odprti,id_igre):
        self.igralci = igralci
        self.berac = berac
        self.talon = talon
        self.odprti = odprti
        self.zgodovina = []
        self.id_igre = id_igre

    def start(self):
        #print('Berac')
        zacne = self.igralci.index(self.berac)
        vrednost_igre = 70
        if self.odprti:
            vrednost_igre = 90
        pisejo = {i:0 for i in self.igralci}
        yield 'Pripravljen menjat' # to je sam zarad konsistence
        for i in range(12):
            if self.odprti and i == 1:
                for i in self.igralci:
                    if i != self.berac:
                        i.poglej_karte_odprtega_beraca( self.berac.roka ,self.id_igre)
            krog_gen = self.krog( zacne,False )
            for j in range( 4 ):
                yield next( krog_gen )
            zmaga = next( krog_gen )  # tle more bit

            zacne += zmaga
            zacne = zacne%4
            if self.igralci[zacne] == self.berac:
                pisejo[self.berac] = -vrednost_igre
                for i in self.igralci:
                    i.rezultat_igre( pisejo.setdefault( i, 0 ), self.zgodovina,self.id_igre )
                #print(self.zgodovina)
                yield pisejo
                return
        pisejo[self.berac] = vrednost_igre

        for i in self.igralci:
            i.rezultat_igre(pisejo.setdefault(i,0),self.zgodovina,self.id_igre)
        yield pisejo