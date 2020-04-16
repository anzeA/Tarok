import gc
import os
import random
import time
import warnings
from copy import deepcopy
from datetime import datetime
from enum import Enum
from sklearn.model_selection import KFold
import numpy as np
from keras.callbacks import EarlyStopping
from keras.engine.saving import load_model

import train
from Tip_igre import Tip_igre
from Karta import Barva, Karta
from Roka import Roka

vse_igre = {str(i).lower() : i for i in Tip_igre}
counter = 0
class Igralec:
    def __init__(self,ime=None):
        if ime is None:
            global counter
            self.ime = str(counter)
            counter +=1
        else:
            self.ime = str(ime)
        self.roka = None
        self.igra = None
        self.kupcek = None
        self.napovedi = []

    def nova_igra(self,roka,igralci):
        self.roka = roka
        self.igra = None
        self.kupcek = []
        self.napovedi = []

    def licitiram(self,licitiram,min_igra, obvezno=None, prednost=False):
        if prednost:
            if licitiram >= min_igra:
                return licitiram
            else:
                if obvezno is None:
                    return Tip_igre.Naprej
                else:
                    return obvezno
        else:
            if licitiram > min_igra:
                return licitiram
            else:
                if obvezno is None:
                    return Tip_igre.Naprej
                else:
                    return obvezno

    def igraj_karto(self, karta):
        self.roka.igraj_karto(karta)
        return karta

    def rezultat_stiha(self,stih,sem_pobral):
        pass

    def rezultat_igre(self,st_tock,povzetek_igre):
        pass

    def poglej_karte_odprtega_beraca(self,roka):
         warnings.warn( 'Ne uporablam podatka za odprtega beraca' )

    def konec_licitiranja(self,igralec_ki_igra,tip_igre,barva_kralja=None):
        pass

    def izberi_barvo_kralja(self):
        warnings.warn( 'Ne izberem barve kralja' )

    def __contains__(self, item):
        return item in self.roka

    def __str__(self):
        return 'Igralec_' + str(self.ime)

    def __repr__(self):
        return str(self)



class Clovekski_igralec(Igralec):
    def __init__(self):
        super().__init__()

    def licitiram(self, min_igra, obvezno=None, prednost=False):
        print(vse_igre)
        while True:
            igram = input('Min igra je '+str(min_igra)+'. Jaz igram : ' ).strip().lower()
            if igram in vse_igre.keys():
                licitiram = vse_igre[igram]
                break
        return  super().licitiram(licitiram,min_igra, obvezno, prednost)

    def izberi_barvo_kralja(self):
        return Barva.KRIZ

class Bot_igralec(Igralec):
    'tip = "min","max","rand" '
    def __init__(self):
        super().__init__()


    def licitiram(self,min_igra,obvezno=None,prednost =False):
        #licitiram = Tip_igre.Dve#random.choice(list(vse_igre.values()))
        #return super().licitiram( licitiram, min_igra, obvezno, prednost )
        return super().licitiram( np.random.choice( [Tip_igre.Naprej,Tip_igre.Tri, Tip_igre.Dve, Tip_igre.Ena],p=[0.5,0.5/3,0.5/3,0.5/3] ), min_igra, obvezno, prednost )


    def izberi_barvo_kralja(self):
        return random.choice( [Barva.SRCE, Barva.KRIZ, Barva.KARA, Barva.PIK] )

    def igraj_karto(self,karte_na_mizi,mozne,zgodovina):
        return super().igraj_karto(random.choice(mozne))

    def menjaj_iz_talona(self,kupcki,st_kart):
        izbrani_kupcek = 0
        self.roka.dodaj_karte(kupcki[izbrani_kupcek])
        mozno = self.roka.mozno_zalozit()
        #print(mozno)
        izberi = random.sample(mozno,k=st_kart)
        self.kupcek.extend(izberi)
        #print(izberi)
        for k in izberi:
            self.roka.igraj_karto(k)
        return izbrani_kupcek

class Nevronski_igralec(Igralec):
    class Tipi_NN(Enum):
        NN_Navadna_igra = 1
        NN_Klop = 0
        NN_Solo = 2
        NN_Berac = 3

    tip_igre_v_tip_izbire = {Tip_igre.Klop: 'Klop',
    Tip_igre.Tri : 'Navadna_igra',
    Tip_igre.Dve : 'Navadna_igra',
    Tip_igre.Ena : 'Navadna_igra',
    Tip_igre.Solo_tri : 'Solo',
    Tip_igre.Solo_dve : 'Solo',
    Tip_igre.Solo_ena : 'Solo',
    Tip_igre.Berac : 'Berac',
    Tip_igre.Solo_brez : 'Solo',
    Tip_igre.Odprti_berac : 'Berac',}
    def __init__(self,load_path='models/',save_path='models/',random_card=0.05,learning_rate=0.99,final_reword_factor=0.1,ime=None):
        super().__init__(ime)
        self.zgodovina = {} #cele igre
        self.roka2tocke =[]
        self.trenutna_igra = [] #trenutna igra (stanje nagrada)
        self.zacetna_roka = None
        self.mozne = None
        self.tip_igre = None
        self.moja_igra= None # tmp za shranjevanje podatkov kaj sem igral
        self.barva_kralja = None
        self.lic  = None
        self.stanje = None
        self.next_Q_max = None
        self.learning_rate= learning_rate
        self.igralci2index = {}
        self.save_path = save_path
        self.random_card = random_card
        self.final_reword_factor = final_reword_factor

        self.igra2index,self.index2igra = Nevronski_igralec.generete_igra2index_and_index2igra()
        if load_path is not None:
            print('Load models')
            self.models={}
            try:
                for f in os.listdir(load_path):
                    self.models[f[:-3]] = load_model(load_path+f)
            except Exception as e:
                print('Problem ',e,'loading models create new one')
                self.models = {'Navadna_igra': train.test_navadna_mreza(), 'Klop': train.test_klop(),
                               'Vrednotenje_roke': train.model_za_vrednotenje_roke()}  # ,'Berac':train.test_berac(),'Solo':train.test_solo(),

        else:
            if load_path is None:
                print( 'Create new models' )
                self.models = {'Navadna_igra': train.test_navadna_mreza(), 'Klop': train.test_klop(),'Vrednotenje_roke': train.model_za_vrednotenje_roke()}  # ,'Berac':train.test_berac(),'Solo':train.test_solo(),

        self.t = 0
        self.since_last_update = 0

    def nova_igra(self,roka,igralci):
        self.tip_igre = None
        self.lic = None
        self.next_Q_max = None
        self.zacetna_roka =deepcopy(roka)
        self.mozne = []
        self.igralci2index = {}
        for i in igralci:
            if i != self:
                self.igralci2index[i] = len(self.igralci2index)
        self.igralci2index[self] = 3
        self.t = 0
        super().nova_igra(roka,igralci)


    def licitiram(self,min_igra,obvezno=None,prednost =False):
        if self.lic is None:
            x = np.zeros((1,54))
            x[0,[k.v_id() for k in self.roka]] = 1
            p = self.models['Vrednotenje_roke'].predict(x)
            id = np.argmax( p )
            self.lic,self.if_play_barva_kralja = self.index2igra[id]
        if self.lic not in [Tip_igre.Tri,Tip_igre.Dve,Tip_igre.Ena]:
            if True:
                self.lic = random.choice( [Tip_igre.Tri,Tip_igre.Dve,Tip_igre.Ena] )
                self.if_play_barva_kralja = random.choice( [Barva.SRCE,Barva.KRIZ,Barva.KARA,Barva.PIK] )
            else:
                self.lic = Tip_igre.Naprej
        self.lic = super().licitiram( self.lic, min_igra, obvezno, prednost )
        #print(self.lic)
        return self.lic

    def izberi_barvo_kralja(self):
        assert self.if_play_barva_kralja is not None
        return self.if_play_barva_kralja

    def igraj_karto(self,karte_na_mizi,mozne,zgodovina):
        self.stanje = self.stanje_v_vektor_rek_navadna( karte_na_mizi, mozne, zgodovina )
        # self.t += time.time()-t
        self.p = self.models[self.tip_igre].predict_on_batch( self.stanje[:-1] )[0]
        mozne_id = [k.v_id() for k in mozne]
        id = np.argmax( self.p[mozne_id] )
        karta = mozne[id]
        self.next_Q_max = self.p[karta.v_id()]
        if random.random() < self.random_card:
            karta = random.choice( mozne )
        self.igrana_karta = karta
        return super().igraj_karto(karta)

    #TODO add extra net
    def menjaj_iz_talona(self,kupcki,st_kart):
        izbrani_kupcek = 0
        self.roka.dodaj_karte(kupcki[izbrani_kupcek])
        mozno = self.roka.mozno_zalozit()
        izberi = random.sample(mozno,k=st_kart)
        self.kupcek.extend(izberi)
        for k in izberi:
            self.roka.igraj_karto(k)
        return izbrani_kupcek

    def rezultat_stiha(self, stih, sem_pobral): #crate train data
        #print('Rezultat stiha',stih,sem_pobral)
        vrednost_stiha = Roka.vrednost_stiha(stih)
        v = self.igrana_karta.vrednost()

        dy = np.zeros(54)#self.p
        dy[self.stanje[-1][0].nonzero()] = -70
        if self.tip_igre in  ["Klop"]:
            if sem_pobral:
                dy[Karta.v_id( self.igrana_karta )] = -vrednost_stiha
            else:
                dy[Karta.v_id( self.igrana_karta )] = vrednost_stiha
        else:
            if sem_pobral:
                dy[Karta.v_id( self.igrana_karta )] = vrednost_stiha
            else:
                dy[Karta.v_id( self.igrana_karta )] = -vrednost_stiha#-(v + vrednost_stiha) / vrednost_stiha
        if len (self.trenutna_igra ) != 0:
            self.trenutna_igra[-1] [3] = self.next_Q_max
        self.trenutna_igra.append( [self.stanje, dy,self.igrana_karta,0] )

    def rezultat_igre(self,st_tock,povzetek_igre):
        #TODO nared discount factor za reword, pa končen reword
        if self.lic in [Tip_igre.Ena,Tip_igre.Dve,Tip_igre.Tri]:
            index_igre = self.igra2index[(self.lic, self.barva_kralja)]
        else:
            if self.lic == Tip_igre.Klop:
                index_igre = self.igra2index[(Tip_igre.Naprej, None)]
            else:
                index_igre = self.igra2index[(self.lic, None)]
        self.roka2tocke.append( (self.zacetna_roka,st_tock,index_igre))

        for stanje,dy,igrana_karta,next_max in self.trenutna_igra:
            dy[igrana_karta.v_id()] = dy[igrana_karta.v_id()]+next_max*self.final_reword_factor
            (self.zgodovina.setdefault( (self.tip_igre,stanje[0].shape[1]),[] )).append((stanje,dy))
        #print('rezultat_igre end',str(self.zgodovina[:1])[:10])

        self.trenutna_igra = []
        if self.since_last_update == 500:
            self.t = time.time()
            mean_batch_size = self.nauci()
            self.t = time.time() - self.t
            self.final_reword_factor = min(self.final_reword_factor*1.1,0.99)
            print(datetime.now(), 'Time used:', self.t,self.since_last_update, "Mean batch_size:",mean_batch_size )
            self.since_last_update = 0
            self.roka2tocke = []
        else:
            self.since_last_update = self.since_last_update+ 1


    def konec_licitiranja(self,igralec_ki_igra,tip_igre,barva_kralja=None):
        self.tip_igre = Nevronski_igralec.tip_igre_v_tip_izbire[tip_igre]
        self.barva_kralja = barva_kralja
        self.index_igralec_ki_igra = self.igralci2index[igralec_ki_igra]

    def stanje_v_vektor_rek_navadna(self,karte_na_mizi,mozne,zgodovina):
        #[input_layer_nasprotiki, kralj, roka_input, talon_input, index_tistega_ki_igra, mozne]
        roka = np.zeros(54)
        roka[[k.v_id() for k in self.roka]] = 1
        #barva_kralja
        barva_kralja = np.zeros(4)
        if self.barva_kralja is not None:
            barva_kralja[ int(self.barva_kralja) ] = 1

        #roka
        if len(zgodovina) > 0:
            roka_input = np.zeros((len(zgodovina),54))
            input_layer_nasprotiki = np.zeros( (len( zgodovina ), 3, 54) )  # nasprotniki
        else:
            input_layer_nasprotiki = np.zeros( (1, 3, 54) )  # nasprotniki
            roka_input = np.zeros( (1, 54) )
            roka_input[0,:] = roka
        # TODO talon
        if self.tip_igre == "Navadna_igra":
            talon_input = np.zeros(( 6,55))
        elif self.tip_igre == 'Klop':
            talon_input = np.zeros((54,))
        elif self.tip_igre == 'Solo':
            talon_input = np.zeros(( 6,55))
        elif self.tip_igre == 'Berac':
            talon_input = None#np.zeros(( 6,55))
        else:
            raise Exception("Ni implementerano"+str(self.tip_igre))


        index_tistega_ki_igra = np.zeros(4)
        index_tistega_ki_igra[self.index_igralec_ki_igra] = 1
        mozne_vec = np.zeros( 54 )
        talon_counter = 0
        for i , (igralec,k) in enumerate(zgodovina):
            if igralec is None: #Talon berac:
                talon_input[k.v_id()] = 1
            elif igralec is not self:
                input_layer_nasprotiki[i,self.igralci2index[igralec],k.v_id()] = i
            elif igralec is self:
                roka_input[i,:] = roka
                roka[k.v_id()] = 0
            else:
                raise Exception("Case not covered")

        ids = [k.v_id() for k in mozne]
        mozne_vec[ids] = 1
        if self.tip_igre == "Navadna_igra":
            r =  [input_layer_nasprotiki, barva_kralja, roka_input, talon_input, index_tistega_ki_igra, mozne_vec]
            return [np.expand_dims(n,axis=0) for n in r]
        elif self.tip_igre == 'Klop':
            r = [input_layer_nasprotiki, roka_input, talon_input, mozne_vec]
            return [np.expand_dims( n, axis=0 ) for n in r]
        else:
            raise Exception("Ni implementerano")

    def nauci(self):
        batch_size_list = []
        for (tip_igre,time_stamp),v in self.zgodovina.items():
            #v je list k ti shran vsa stanja z isto dolzino
            batch_size_list.append(len(v))
            (stanje,y) = v[0]
            batch_size  = len(v)
            X = []
            dy = np.zeros((len(v),54))
            for n in stanje:
                shape = list(n.shape)

                shape[0] = len(v)
                shape = tuple(shape)
                X.append(np.empty(shape))
            del stanje
            #print([n.shape for n in X])
            for i,(stanje,y) in enumerate(v):
                dy[i,:] = y
                #print(time_stamp,stanje[0].shape)
                for j,x in enumerate(stanje):
                    X[j][i,:] = x
            for _ in range(3):
                p = self.models[tip_igre].predict_on_batch(X[:-1])
                non_z = dy.nonzero()
                p[non_z] = dy[non_z]
                p = p*X[-1] # krat mozne
                #self.models[tip_igre].fit( X[:-1], p,epochs=5,verbose=0,callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='min')] )
                kf = KFold( n_splits=len(v)//32 +1 ,shuffle=True,)
                for train_index, test_index in kf.split( X[0] ):
                    #ignore train index
                    X_batch = [x_tensor[test_index] for x_tensor in X[:-1]]
                    y_batch = p[test_index]
                    for __ in range(5): # memory issues
                        #self.models[tip_igre].train_on_batch( X[:-1], p)
                        self.models[tip_igre].train_on_batch( X_batch, y_batch)


        X = np.zeros((len(self.roka2tocke),54))
        for i,(r,t,tip) in enumerate(self.roka2tocke):
            X[i,[k.v_id() for k in r]] = 1
        for _ in range(3):
            y = self.models['Vrednotenje_roke'].predict_on_batch(X)
            for i,(r,t,index_igre) in enumerate(self.roka2tocke):
                y[i,index_igre] = t
            #self.models['Vrednotenje_roke'].fit(X,y,epochs=5,verbose=0,callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='min')])
            kf = KFold( n_splits=len( y ) // 32 + 1, shuffle=True, )
            for train_index, test_index in kf.split( X ):
                X_batch,y_batch = X[test_index], y[test_index]
                for __ in range( 5 ):  # memory issues
                    self.models['Vrednotenje_roke'].train_on_batch( X_batch,y_batch )

        self.save_models()
        self.zgodovina = {}
        for  _ in range(20):
            gc.collect()
        return np.mean(batch_size_list)

    @staticmethod
    def generete_igra2index_and_index2igra():
        igra2index = {}
        index2igra = {}
        i = 0
        for t in Tip_igre:
            if t in [Tip_igre.Tri,Tip_igre.Dve,Tip_igre.Ena]:
                for b in Barva:
                    if b == Barva.TAROK:
                        continue
                    index2igra[i] = (t,b)
                    igra2index[(t,b)] = i
                    i = i+1
            elif t == Tip_igre.Klop:
                continue #To bi blo isto k Naprej
            else:
                index2igra[i] = (t,None)
                igra2index[(t, None)] = i
                i = i+1
        return igra2index,index2igra

    def save_models(self):
        for k,v in self.models.items():
            v.save(self.save_path+k+".h5")