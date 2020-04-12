from Karta import Karta, Barva


class Roka:
    def __init__(self,karte):
        self.karte = {b:[]for b in Barva}
        #karte.sort()
        #print(karte)
        for k in karte:
            self.karte[k.barva].append(k)
        for v in self.karte.values():
            v.sort()
        #print(self.karte)
        #print()

    def igraj_karto(self,k):
        self.karte[k.barva].remove(k)

    def dodaj_karte(self,karte):
        for k in karte:
            self.karte[k.barva].append(k)

    def mozno_zalozit(self):
        mozno = []
        for v in self.karte.values():
            mozno.extend([k for k in v if k.vrednost() < 5])
        return mozno

    def __contains__(self, karta):
        if isinstance(karta,Karta):
            for i in self.karte.values():
                if karta in i:
                   return True
        return False

    def __str__(self):

        l = []
        for  i in self.karte.values():
            l.extend(i)
        l.sort()
        return str(l)

    def __iter__(self):
        for k in self.karte.values():
            for karta in k:
                yield karta

    def __len__(self):
        return sum([len(l) for l in self.karte.values()])

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def tri_po_tri(kupcek):
        for i in range( len( kupcek ) // 3 ):
            yield kupcek[i * 3:(i + 1) * 3]
        if len( kupcek ) % 3 != 0:
            yield kupcek[-(len( kupcek ) % 3):]

    '''
    Za končni seštevek se pike seštevajo na poseben način: s kupa priigranih kart se jemljejo po tri karte hkrati.
    Vrednost vseh treh kart se sešteje, pri tem štejejo platelci in taroki (razen trule) le po eno piko.
    Sešteti vrednosti treh močnih kart (vse zgoraj naštete) je treba odvzeti dve piki. 
    Kadar sta med tremi kartami le dve močni, se odvzame le ena pika.
    Če med tremi kartami ni nobene močne, šteje cela trojica platelcev samo eno piko.
     
    Na koncu lahko ostaneta po dve karti ali le ena.
    Dvema močnima se odvzame ena pika od njune seštete vrednosti, eni sami pa ni treba ničesar odvzemati.
    Platelca štejeta piko, če ostaneta dva; če je edina, zadnja karta platelc, pa ne šteje ničesar. 
    Vse vrednosti treh trojic vsaki igralec sešteva sproti.
    '''

    @staticmethod
    def vrednost_stiha(stih):
        st_polnih = 0
        vrednost = 0
        for k in stih:
            if k.barva == Barva.TAROK:
                if 1 < k.st < 21:
                    vrednost+=1
                elif k.barva == Barva.TAROK:
                    vrednost += 5
                    st_polnih += 1
            else:
                if 4 < k.st:
                    vrednost += k.st - 3  # prever
                    st_polnih += 1
                else:
                    vrednost +=1
        if len(stih)  in [1,2]:
            return vrednost-1

        return vrednost-2 # ce so tri prazne je usejen 1
    @staticmethod
    def prestej(kupcek):
        return  sum([Roka.vrednost_stiha(stih) for stih in Roka.tri_po_tri(kupcek)])
