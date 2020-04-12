import enum


class Tip_igre(enum.IntEnum):
    Naprej = -10
    Klop = 0
    Tri = 10
    Dve = 20
    Ena = 30
    Solo_tri = 40
    Solo_dve = 50
    Solo_ena = 60
    Berac = 70
    Solo_brez = 80
    Odprti_berac = 90

    def __str__(self):
        return super().__str__()[9:]

    def __repr__(self):
        return str(self)