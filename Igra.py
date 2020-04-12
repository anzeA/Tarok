'''
Glavni razred za igro
'''
import enum
from enum import Enum

from Berac import Berac
from Karta import Barva, Karta

from random import shuffle

from Navadna_igra import Navadna_igra
from Klop import Klop
from Roka import Roka

from Tip_igre import Tip_igre

class Igra:
    def __init__(self, igralci):
        self.igralci = igralci
        self.zgodovina = []

    def start(self):
        talon = self.razdeli()
        # licitacija
        index_igralca , lic_igra = self.licitacija()
        barva_kralja = None
        if lic_igra in [Tip_igre.Ena,Tip_igre.Dve,Tip_igre.Tri,
                        Tip_igre.Solo_brez,Tip_igre.Solo_ena,Tip_igre.Solo_dve,Tip_igre.Solo_tri
                        ]:
            if lic_igra in [Tip_igre.Ena,Tip_igre.Dve,Tip_igre.Tri]:
                barva_kralja = self.igralci[index_igralca].izberi_barvo_kralja()

            igra = Navadna_igra(self.igralci,lic_igra,barva_kralja,self.igralci[index_igralca],talon)

        elif lic_igra == Tip_igre.Klop:
            igra = Klop(self.igralci,talon)
        elif lic_igra == Tip_igre.Berac:
            igra = Berac(self.igralci,self.igralci[index_igralca],talon,False)

        elif lic_igra == Tip_igre.Odprti_berac:
            igra = Berac(self.igralci,self.igralci[index_igralca],talon,True)
        else:
            raise Exception("Igra ni definirana:" + str(lic_igra))
        #TODO napovedovanje
        for igralec in self.igralci:
            igralec.konec_licitiranja(self.igralci[index_igralca],lic_igra,barva_kralja)
        return igra.start()

        # zacni to igro

    def razdeli(self):
        karte = list( range( 54 ) )
        shuffle( karte )
        talon = karte[-6:]
        for i, igralec in enumerate( self.igralci ):
            roka = [Karta.iz_id( id ) for id in karte[i * 12: (i + 1) * 12]]

            igralec.nova_igra( Roka( roka ), self.igralci )
        return [Karta.iz_id( id ) for id in talon]

    def licitacija(self):
        # 0 ma obvezno 3
        lic = set()
        max_igra = Tip_igre.Tri
        for i,ig in enumerate(self.igralci[1:],1):
            napoved = ig.licitiram(max_igra)
            if napoved != Tip_igre.Naprej:
                lic.add(i)
            max_igra = max(max_igra,napoved)

        if max_igra == Tip_igre.Tri:
            napoved = self.igralci[0].licitiram(Tip_igre.Naprej,Tip_igre.Klop)
            return 0,napoved
        else:
            napoved = self.igralci[0].licitiram( max_igra,prednost = True )
            if napoved != Tip_igre.Naprej:
                lic.add(0)
            max_igra = max(max_igra,napoved)
        ima_igro = min(lic)
        while len(lic) != 1 :
            new_lic = set()
            keys = list(lic)
            keys.sort()
            if keys[0] ==0 :
                keys = keys[1:]+[0]
            for k in keys:
                if  k == ima_igro:
                    napoved = self.igralci[k].licitiram( max_igra,max_igra )
                else:
                    napoved = self.igralci[k].licitiram( max_igra)
                if napoved != Tip_igre.Naprej:
                    new_lic.add(k)
                    ima_igro = k
                    max_igra = napoved
            lic = new_lic
        return ima_igro,max_igra



