import enum
from enum import Enum


class Karta:
    def __init__(self,barva,st):
        self.barva = barva
        self.st = st

    def vrednost(self):
        if self.barva == Barva.TAROK and self.st in [1,21,22]: # skis, palcka, 21
            return 5
        elif self.st > 4: # slika
            return self.st - 3
        else: #platlc
            return 1

    # TODO pretesteri te dve fun
    def v_id(self):
        if self.barva == Barva.TAROK:
            return 4*8 + self.st -1 # -1 ker id gre od 0-53 in st gre pr tarok od 1-22
        else:
            return self.barva*8 + self.st-1

    def __eq__(self, other):
        if isinstance(other,Karta):
            if self.barva == other.barva and self.st == other.st:
                return True
        return False

    @staticmethod
    def iz_id(id):
        if id > 31:
            barva = Barva.TAROK
            st = id-31
        else:
            barva = id//8
            if barva == 0:
                barva = Barva.KARA
            elif barva == 1:
                barva = Barva.SRCE
            elif barva == 2:
                barva = Barva.PIK
            elif barva == 3:
                barva = Barva.KRIZ
            st = ( id % 8 ) +1
        return Karta(barva,st)




    def __str__(self):
        if self.barva != Barva.TAROK and self.st > 4:
            m = {5:'J',6:'K',7:'D',8:'KR'}[self.st]
        else:
            m = self.st
        return str(self.barva)+'_'+str(m)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        assert isinstance(other,Karta)
        if other.barva != self.barva:
            return  self.barva <other.barva
        else:
            return self.st < other.st

class Barva( enum.IntEnum ):
    KARA = 0
    SRCE = 1
    PIK = 2
    KRIZ = 3
    TAROK = 4

    def __str__(self):
        return super().__str__()[6:]

    def __repr__(self):
        return str(self)

