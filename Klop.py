from copy import deepcopy

from Karta import Barva, Karta
from Roka import Roka

'''
igralci: 4-je igralci. Igralec na mestu 0 zacne igro
igra: tip_igre ena,dva,tri,solo 1,2,3, solo brez
barva_kralja: eden izmed razred Tip razen tarok
igralec: on igra
solo = None, navadna igra
    1,2,3 solo 1,2,3
    'solo_brez' je solo brez talona 
'''
class Klop():
    def __init__(self,igralci,talon):
        self.igralci = igralci
        self.talon = talon
        self.zgodovina = []


    def start(self):
        #print('Klop')
        zacne = 0
        for i in range(12):
            zmaga = self.krog(zacne)
            zacne += zmaga
            zacne = zacne%4

        pisejo = {i:-Roka.prestej(i.kupcek) for i in self.igralci}
        if any( [v < -35 for v in pisejo.values() ] ):
            for i in self.igralci:
                if Roka.prestej(i.kupcek) <  -35:
                    pisejo[i] = -70
                else:
                    pisejo[i] = 0
        for k,v in pisejo.items():
            k.rezultat_igre(v,self.zgodovina)
        return pisejo

    def krog(self,start_index):
        stih = []
        spodnja = None
        for i in range(4):
            igralec = self.igralci[ (start_index+i)%4]
            mozne = self.mozne_karte( spodnja,igralec.roka)
            mozne_copy = deepcopy(mozne)
            karta = igralec.igraj_karto(stih,mozne,self.zgodovina)
            if karta not in mozne_copy:
                raise Exception( str( igralec ) + str( igralec.__class__ ) + ' Karte ne mores igarti. Karta: ' + str(
                    karta ) + ' karte na mizi:' + str( stih ) + ' Roka' + str( igralec.roka ) + 'Mozne' + str(
                    mozne ) + 'deep mozne:' + str( mozne_copy ) )


            self.zgodovina.append((igralec,karta))
            if spodnja is None:
                spodnja = karta
            stih.append(karta)
        if len(self.talon) > 0:
            #TODO izpis
            talon_karta = self.talon.pop()
            stih.append(talon_karta)
            self.zgodovina.append( (None, talon_karta) )
        zmaga = self.pobere_stih(stih)
        #print(self.igralci[(start_index+zmaga)%4].ime, stih )
        self.igralci[(start_index+zmaga)%4].kupcek.extend(stih)
        zmagovalni_index = (start_index+zmaga)%4
        for i in range(4):
            self.igralci[i].rezultat_stiha(stih,i == zmagovalni_index)

        return zmaga

    def pobere_stih(self,stih):
        zmaga= 0
        for i in range(1,4):
            if self.primerjaj_karti(stih[zmaga],stih[i]):
                zmaga=i
        return zmaga

    def primerjaj_karti(self,k1:Karta,k2:Karta):
        if k1.barva == k2.barva:
            return k1.st < k2.st
        elif k2.barva == Barva.TAROK:
            return True
        else:
            return False

    def mozne_karte(self,spodnja_karta,roka):
        #TODO ce sta gor skis in 21 more bit palca obvezna
        #print('---------------------------------')
        #print( 'spodnja_karta:', spodnja_karta, 'roka:', roka )
        if spodnja_karta is not None and len(roka.karte[spodnja_karta.barva]) > 0:
            mozne =  roka.karte[spodnja_karta.barva]
            filtrirano = [k for k in mozne if self.primerjaj_karti(spodnja_karta,k)]
            if len(filtrirano) > 0:
                brez_palcke = [k for k in mozne if not (k.barva == Barva.TAROK and k.st == 1)]
                if len(brez_palcke) > 0:
                    filtrirano = brez_palcke
                #print(filtrirano)
                return filtrirano
            else:
                #print(mozne)
                if len(mozne) > 1:
                    mozne = [k for k in mozne if not (k.barva == Barva.TAROK and k.st == 1 )]
                return mozne
        elif spodnja_karta is not None and len(roka.karte[Barva.TAROK]) > 0:
            taroki = roka.karte[Barva.TAROK]
            filtrirano = [k for k in taroki if self.primerjaj_karti( spodnja_karta, k )]
            if len(filtrirano) > 0 :
                mozne = [t for t in taroki if t.st != 1] # odstrani palcko
                if len(mozne) == 0:
                    mozne = taroki
                #print(mozne)
                return mozne
            #print(taroki)
            return taroki
        else:
            vse = []
            for r in roka.karte.values():
                vse.extend(r)
            vse.sort()
            if len(vse) > 1: # umaknes palcko ker je vec k 1 karta mozna
                vse = [k for k in vse if not (k.barva == Barva.TAROK and k.st == 1 )]
            #print(vse)
            return vse
