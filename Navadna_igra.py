from Karta import Barva, Karta
from Roka import Roka
from copy import deepcopy
from Tip_igre import Tip_igre
'''
igralci: 4-je igralci. Igralec na mestu 0 zacne igro
igra: tip_igre ena,dva,tri,solo 1,2,3, solo brez
barva_kralja: eden izmed razred Tip razen tarok
igralec: on igra
solo = None, navadna igra
    1,2,3 solo 1,2,3
    'solo_brez' je solo brez talona 
'''
class Navadna_igra():
    def __init__(self,igralci,igra,barva_kralja,igralec,talon,id_igre):
        self.igralci = igralci
        self.igra = igra
        self.igralec = igralec
        self.id_igre = id_igre
        if igra in [Tip_igre.Ena,Tip_igre.Dve,Tip_igre.Tri]:
            assert barva_kralja != Barva.TAROK
            self.barva_kralja = barva_kralja
            self.solo = False
            self.ekipa = [i for i in self.igralci if (Karta( self.barva_kralja, 8 ) in i.roka[self.id_igre]) or i == igralec]
            self.ekipa2 = [i for i in self.igralci if not (Karta( self.barva_kralja, 8 ) in i.roka[self.id_igre] or i == igralec)]
        else:
            self.barva_kralja = None
            self.solo = True
            self.ekipa = [i for i in self.igralci if i == igralec]
            self.ekipa2 = [i for i in self.igralci if not i == igralec]


        self.talon = talon
        self.zgodovina = []

    def odpri_talon(self):
        if self.igra == Tip_igre.Solo_tri or self.igra == Tip_igre.Tri:
            korak = 3
        elif self.igra == Tip_igre.Solo_dve or self.igra == Tip_igre.Dve:
            korak = 2
        else:# ena ,soloaena al pa solo_brez
            korak = 1

        return [self.talon[i:i+korak] for i in range(0,6,korak)]

    def start(self):
        kupcki_talona = self.odpri_talon()
        if self.igra != Tip_igre.Solo_brez:
            if self.igra == Tip_igre.Solo_tri or self.igra == Tip_igre.Tri:
                st_kart_za_menjat = 3
            elif self.igra == Tip_igre.Solo_dve or self.igra == Tip_igre.Dve:
                st_kart_za_menjat = 2
            elif self.igra == Tip_igre.Solo_ena or self.igra == Tip_igre.Ena:
                st_kart_za_menjat = 1
            elif self.igra == Tip_igre.Solo_brez:  # ena ,soloaena al pa solo_brez
                st_kart_za_menjat = 0
            else:
                raise Exception('Ni implentirano '+str(self.igra))
            if self.igra != Tip_igre.Solo_brez:
                self.igralec.pripravi_izbral_iz_talona(deepcopy( kupcki_talona ), st_kart_za_menjat,self.id_igre)
                yield 'Pripravljen menjat'
                stevilka_kupcka = self.igralec.menjaj_iz_talona(deepcopy(kupcki_talona),st_kart_za_menjat,self.id_igre)
                self.zgodovina.append( ("Talon",(stevilka_kupcka,deepcopy(kupcki_talona))) )
                for i in self.igralci:
                    i.izbral_iz_talona(deepcopy(kupcki_talona),stevilka_kupcka,self.id_igre)
                del kupcki_talona[stevilka_kupcka]
            else:
                yield 'Solo ne rab menjat iz talona'

        zacne = 0

        for i in range(12):
            krog_gen = self.krog( zacne )
            for j in range(4):
                yield next(krog_gen)
            zmaga =next(krog_gen) # tle more bit
            assert isinstance(zmaga, int)
            zacne += zmaga
            zacne = zacne%4
        skupek_kupckov = []
        skupek_kupckov2 = []
        for i in self.ekipa:
            #print('Ekipa')
            skupek_kupckov.extend(i.kupcek[self.id_igre])
        for i in self.ekipa2:
            skupek_kupckov2.extend(i.kupcek[self.id_igre])
        if self.igra != Tip_igre.Solo_brez and len(self.ekipa) == 1 and self.barva_kralja != None and Karta(self.barva_kralja,8) in self.igralec.kupcek[self.id_igre]:
            dodaj_talon = skupek_kupckov
        else:
            dodaj_talon = skupek_kupckov2
        for s in kupcki_talona:
            dodaj_talon.extend(s)
        #print()
        vrednost = Roka.prestej(skupek_kupckov)
        #print('Ekipa 1',vrednost)
        #vrednost2 = Roka.prestej( skupek_kupckov2 )

        #print('Ekipa 2',vrednost2)


        pisejo = {}
        razlika = vrednost-35
        razlika = int(round(razlika/5))*5
        for i in self.ekipa:
            if vrednost > 35:
                pisejo[i] = int(self.igra) + razlika
            else:
                pisejo[i] = -int(self.igra) + razlika

        for i in self.igralci:
            i.rezultat_igre(pisejo.setdefault(i,0),self.zgodovina,self.id_igre)
        #print(self.igra,vrednost,vrednost2,pisejo,razlika)
        yield pisejo

    def krog(self,start_index):
        stih = []
        spodnja = None
        for i in range(4):
            igralec = self.igralci[ (start_index+i)%4]
            mozne = self.mozne_karte( spodnja,igralec.roka[self.id_igre])
            mozne_copy = deepcopy(mozne)
            igralec.pripravi_igraj_karto( deepcopy(stih), mozne, self.zgodovina,self.id_igre )
            yield 'Pripravljen igrat karto'
            karta = igralec.igraj_karto(deepcopy(stih),mozne,self.zgodovina,self.id_igre)
            if karta not in mozne_copy:
                raise Exception(str(igralec)+str(igralec.__class__)+' Karte ne mores igarti. Karta: '+str(karta)+' karte na mizi:'+str(stih)+' Roka'+str(igralec.roka)+'Mozne'+str(mozne) + 'deep mozne:'+str(mozne_copy))
            self.zgodovina.append((igralec,karta))
            if spodnja is None:
                spodnja = karta
            stih.append(karta)
            #print(igralec.ime,karta)

        zmaga = self.pobere_stih(stih)
        #print(self.igralci[(start_index+zmaga)%4].ime, stih )
        self.igralci[(start_index+zmaga)%4].kupcek[self.id_igre].extend(stih)
        #print()
        zmagovalni_index = (start_index+zmaga)%4
        for i in range(4):
            self.igralci[i].rezultat_stiha(stih,i == zmagovalni_index,self.id_igre)

        yield zmaga

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
        if spodnja_karta is not None and len(roka.karte[spodnja_karta.barva]) > 0:
            return roka.karte[spodnja_karta.barva]
        elif spodnja_karta is not None and len(roka.karte[Barva.TAROK]) > 0:
            return roka.karte[Barva.TAROK]
        else:
            vse = []
            for r in roka.karte.values():
                vse.extend(r)
            vse.sort()
            return vse


