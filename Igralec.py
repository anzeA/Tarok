import gc
import math
import os
import random
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from enum import Enum
import torch.multiprocessing as mp
#from keras.optimizers import SGD
#from sklearn.model_selection import KFold
import numpy as np
#from keras.callbacks import EarlyStopping
#from keras.engine.saving import load_model
#import tensorflow as tf
#import train
from pytorch_lightning.callbacks import EarlyStopping

from Tip_igre import Tip_igre
from Karta import Barva, Karta
from Roka import Roka
from torch_models import *

import pytorch_lightning as pl
from pytorch_lightning import Trainer


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
        self.roka = dict()
        self.igra = dict()
        self.kupcek = dict()
        self.napovedi = []

    def nova_igra(self,roka,igralci,id_igre):
        self.roka[id_igre] = roka
        self.igra[id_igre] = None
        self.kupcek[id_igre] = []
        self.napovedi = []

    def pripavi_licitiram(self,id_igre):
        pass

    def predict_licitiram(self):
        pass
        #print('Lecitiram v igralce',self)

    def licitiram(self,licitiram,min_igra,id_igre, obvezno=None, prednost=False,):
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

    def pripravi_igraj_karto( self, karte_na_mizi, mozne, zgodovina,id_igre ):
        pass

    def predict_igraj_karto(self):
        pass

    def igraj_karto(self, karta,id_igre):
        self.roka[id_igre].igraj_karto(karta)
        #print('Igralec prou klice super v igraj_karto')
        return karta

    def rezultat_stiha(self,stih,sem_pobral,id_igre):
        pass

    def rezultat_igre(self,st_tock,povzetek_igre,id_igre):
        pass

    def poglej_karte_odprtega_beraca(self,roka,id_igre):
         warnings.warn( 'Ne uporablam podatka za odprtega beraca' )

    def konec_licitiranja(self,igralec_ki_igra,tip_igre,id_igre,barva_kralja=None):
        pass

    def izberi_barvo_kralja(self,id_igre):
        raise NotImplementedError()

    def pripravi_izbral_iz_talona(self,talon,st_kupcka,id_igre):
        pass

    def predict_izberi_iz_talona(self):
        pass
        #print('Predict iz talona')

    def izbral_iz_talona(self,talon,st_kupcka,id_igre):
        pass

    def menjaj_iz_talona(self,kupcki,st_kart,id_igre):
        raise NotImplementedError()

    def __contains__(self, item):
        return item in self.roka

    def __str__(self):
        return 'Igralec_' + str(self.ime)

    def __repr__(self):
        return str(self)



class Clovekski_igralec(Igralec):
    def __init__(self):
        super().__init__()

    def licitiram(self, min_igra, id_igre,obvezno=None, prednost=False):
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


    def licitiram(self,min_igra,id_igre,obvezno=None,prednost =False):
        #licitiram = Tip_igre.Dve#random.choice(list(vse_igre.values()))
        #return super().licitiram( licitiram, min_igra, obvezno, prednost )
        return super().licitiram( np.random.choice( [Tip_igre.Naprej,Tip_igre.Tri, Tip_igre.Dve, Tip_igre.Ena],p=[0.5,0.5/3,0.5/3,0.5/3] ), min_igra, id_igre,obvezno, prednost )
        #return super().licitiram( Tip_igre.Berac, min_igra,id_igre, obvezno, prednost )


    def izberi_barvo_kralja(self,id_game):
        return random.choice( [Barva.SRCE, Barva.KRIZ, Barva.KARA, Barva.PIK] )

    def igraj_karto(self,karte_na_mizi,mozne,zgodovina,id_igre):
        return super().igraj_karto(random.choice(mozne),id_igre)

    def menjaj_iz_talona(self,kupcki,st_kart,id_igre):
        izbrani_kupcek = 0
        self.roka[id_igre].dodaj_karte(kupcki[izbrani_kupcek])
        mozno = self.roka[id_igre].mozno_zalozit()
        #print(mozno)
        izberi = random.sample(mozno,k=st_kart)
        self.kupcek[id_igre].extend(izberi)
        #print(izberi)
        for k in izberi:
            self.roka[id_igre].igraj_karto(k)
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
    def __init__(self,load_path=None,save_path=None,random_card=0.9,learning_rate=0.1,final_reword_factor=0.1,ime=None,ignor_models=False):
        super().__init__(ime)
        self.zgodovina = {} #cele igre
        self.roka2tocke =[]
        self.zalaganje2tocke ={} #
        self.trenutna_igra = dict()  #trenutna igra (stanje nagrada)
        self.zacetna_roka = dict()
        self.tip_igre = dict()
        self.moja_igra= dict() # tmp za shranjevanje podatkov kaj sem igral
        self.barva_kralja = dict()
        self.lic  = dict()
        self.stanje = dict()
        self.next_Q_max = dict()
        self.zalozil = dict()
        self.if_play_barva_kralja = dict()
        self.igrana_karta = dict()
        self.igralci2index = {}
        self.index_igralec_ki_igra = {}

        self.learning_rate= learning_rate
        self.save_path = save_path
        self.random_card = random_card
        self.final_reword_factor = final_reword_factor
        self.load_path = load_path
        self.igra2index,self.index2igra,self.igra_zalozi2index = Nevronski_igralec.generete_igra2index_and_index2igra()
        self.t = 0
        self.since_last_update = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if ignor_models == False:
            if load_path is not None:
                print('Load models')
                self.models={}
                try:
                    self.load_models()
                except Exception as e:
                    print('Problem ',e,'loading models create new one')
                    self.models = self.create_models()
            else:
                self.load_path = self.save_path
                print( 'Create new models' )
                self.models = self.create_models()

        keys = ['Navadna_igra', 'Klop','Solo','Berac','Vrednotenje_roke','Zalaganje']  # ,'Berac':train.test_berac(),'Solo':train.test_solo(),

        self.predict_queue = {k:[]for k in keys}  #
        self.predicted_resoult = {k: {}for k in keys}  # key is tip napovidi, in id_igre value je predikcija

    def create_models(self):
        d = {'Navadna_igra': Net_Navadna_igra(), 'Klop': Net_Klop(),'Solo':Net_Solo(),'Berac':Net_Berac(),
                                   'Vrednotenje_roke': Net_vrednotenje_roke(),'Zalaganje':Net_zalaganje()}
        for v in d.values():
            v.to(device = self.device)
        return d

    def load_models(self):
        self.models = dict()
        for f,m in [('Navadna_igra',Net_Navadna_igra), ('Klop',Net_Klop),('Solo',Net_Solo),('Berac',Net_Berac),('Vrednotenje_roke',Net_vrednotenje_roke),('Zalaganje',Net_zalaganje)]:#os.listdir( self.load_path ):
            self.models[f] = m.load_from_checkpoint(os.path.join( self.load_path, f+".cktp" ))
            #tf.keras.models.load_model( os.path.join( self.load_path, f ) ,compile=True)
            #self.models[f[:-3]].compile( optimizer=tf.keras.optimizers.Adadelta(lr=self.learning_rate),
            #       loss=tf.keras.losses.Huber(delta=15.0),
            #       metrics=['accuracy'] )

    def del_models(self):
        del self.models
        self.models = None

    def nova_igra(self,roka,igralci,id_igre):
        self.tip_igre[id_igre] = None
        self.lic[id_igre] = None
        self.next_Q_max[id_igre] = None
        self.zalozil[id_igre] = None
        self.stanje[id_igre] = None
        self.zacetna_roka[id_igre] =deepcopy(roka)
        self.igralci2index[id_igre] = {}
        self.index_igralec_ki_igra[id_igre] = None
        #self.zalaganje2tocke[id_igre] = None
        self.trenutna_igra[id_igre] = []
        #self.roka2tocke[id_igre] = None
        self.igrana_karta[id_igre] = None
        for i in igralci:
            if i != self:
                self.igralci2index[id_igre][i] = len(self.igralci2index[id_igre])
        self.igralci2index[id_igre][self] = 3
        self.t = 0
        super().nova_igra(roka,igralci,id_igre)

    def pripavi_licitiram(self,id_igre):
        x = np.zeros( (1, 54) )
        x[0, [k.v_id() for k in self.roka[id_igre]]] = 1
        self.predict_queue['Vrednotenje_roke'].append((id_igre,x))

    def predict_licitiram(self):
        X = np.zeros( (len(self.predict_queue['Vrednotenje_roke']),54) )
        for i,(id,x) in enumerate(self.predict_queue['Vrednotenje_roke']):
            X[i,:] = x
        Y = self.models['Vrednotenje_roke'](torch.from_numpy(X).to(device=self.device,dtype=torch.float32))
        Y =Y.cpu().detach().numpy()
        #Y = self.models['Vrednotenje_roke'].predict_on_batch( X )
        for i,(id, x) in enumerate(self.predict_queue['Vrednotenje_roke']):
            self.predicted_resoult['Vrednotenje_roke'][id] = Y[i,:]
        self.predict_queue['Vrednotenje_roke'] = []

    def licitiram(self,min_igra,id_igre,obvezno=None,prednost =False):
        if self.lic[id_igre] is None:
            p = self.predicted_resoult['Vrednotenje_roke'][id_igre]
            id = np.argmax( p )
            self.lic[id_igre],self.if_play_barva_kralja[id_igre] = self.index2igra[id]

            if random.random() < self.random_card:
                self.lic[id_igre], self.if_play_barva_kralja[id_igre] = random.choice( list(self.index2igra.values()) )

        assert self.lic[id_igre] != Tip_igre.Odprti_berac
        self.lic[id_igre] = super().licitiram( self.lic[id_igre], min_igra,id_igre, obvezno, prednost )
        #print(self.lic)
        return self.lic[id_igre]

    def izberi_barvo_kralja(self,id_igre):
        assert self.if_play_barva_kralja[id_igre] is not None
        return self.if_play_barva_kralja[id_igre]

    def pripravi_igraj_karto( self, karte_na_mizi, mozne, zgodovina, id_igre ):
        self.stanje[id_igre] = self.stanje_v_vektor_rek_navadna( karte_na_mizi, mozne, zgodovina,id_igre )
        self.predict_queue[self.tip_igre[id_igre]].append( (id_igre,self.stanje[id_igre]))

    def predict_igraj_karto(self):
        with torch.no_grad():
            for tip_igre,X_list in self.predict_queue.items():
                if len(X_list) == 0:
                    continue
                stanje = X_list[0][1]
                X = []
                for n in stanje:
                    shape = list( n.shape )
                    shape[0] = len( X_list )
                    shape = tuple( shape )
                    X.append( np.zeros( shape ) )
                del stanje
                # print([n.shape for n in X])
                for i,(id, stanje) in  enumerate(X_list) :
                    for j, x in enumerate( stanje ):
                        X[j][i, :] = x
                X = X[:-1]
                X = [torch.from_numpy(x).to(device=self.device,dtype=torch.float32) for x in X]
                self.models[tip_igre].eval()
                Y = self.models[tip_igre]( X )
               # print('Napovedal',tip_igre)
                for i,(id, stanje) in enumerate(X_list):
                    self.predicted_resoult[tip_igre][id] =Y[i,:]

            for k in self.predict_queue.keys():
                self.predict_queue[k]  = []

    def igraj_karto(self,karte_na_mizi,mozne,zgodovina,id_igre):
        # self.t += time.time()-t
        p = self.predicted_resoult[self.tip_igre[id_igre]][id_igre].cpu()
        #p = self.models[self.tip_igre].predict( self.stanje[:-1],batch_size=1,use_multiprocessing=True )[0]
        mozne_id = [k.v_id() for k in mozne]
        id = np.argmax( p[mozne_id] )
        karta = mozne[id]
        self.next_Q_max[id_igre] = p[karta.v_id()]
        if random.random() < self.random_card:
            karta = random.choice( mozne )
        self.igrana_karta[id_igre] = karta
        return super().igraj_karto(karta,id_igre)

    def pripravi_izbral_iz_talona(self,talon,st_kart,id_igre):
        stanje = self.menjaj_talon_v_vektor( talon,id_igre )
        self.zalaganje2tocke[id_igre] = [stanje]
        self.predict_queue['Zalaganje'].append( (id_igre,stanje) )

    def predict_izberi_iz_talona(self):
        self.predict_igraj_karto()

    def menjaj_iz_talona(self,kupcki,st_kart,id_igre):
        p = self.predicted_resoult['Zalaganje'][id_igre].cpu()
        izbrani_kupcek = np.argmax(p[54:54+len(kupcki)])

        if random.random() < self.random_card:
            izbrani_kupcek = random.randint(0,len(kupcki)-1)

        self.roka[id_igre].dodaj_karte(kupcki[izbrani_kupcek])
        mozno = self.roka[id_igre].mozno_zalozit()
        zalozi = [k.v_id() for k in mozno]
        if random.random() < self.random_card:
            zalozi = random.sample(mozno,st_kart)
        else:
            zalozi = [mozno[i] for i in np.argsort(p[zalozi])[-st_kart:]]
        self.kupcek[id_igre].extend(zalozi)
        self.zalozil[id_igre] = True
        for k in zalozi:
            self.roka[id_igre].igraj_karto(k)

        self.zalaganje2tocke[id_igre] = [self.zalaganje2tocke[id_igre][0],zalozi,st_kart,None,mozno]
        return izbrani_kupcek

    def rezultat_stiha(self, stih, sem_pobral, id_igre): #crate train data
        #print('Rezultat stiha',stih,sem_pobral)
        vrednost_stiha = Roka.vrednost_stiha(stih)
        v = self.igrana_karta[id_igre].vrednost()
        igrana_karta = self.igrana_karta[id_igre]
        dy = np.zeros(54)
        dy[self.stanje[id_igre][-1][0]== 0] = -70
        if self.tip_igre ==  "Klop":
            if sem_pobral:
                dy[Karta.v_id( igrana_karta )] = -vrednost_stiha
            else:
                dy[Karta.v_id( igrana_karta[id_igre] )] = vrednost_stiha
        elif self.tip_igre == 'Berac':
            if self.index_igralec_ki_igra ==3: # jaz igram
                if sem_pobral:
                    dy[Karta.v_id( igrana_karta )] = -1
                else:
                    dy[Karta.v_id( igrana_karta )] = 1
            else:
                if sem_pobral:
                    dy[Karta.v_id( igrana_karta )] = -1
                else:
                    dy[Karta.v_id( igrana_karta )] = 1
                # ce nasprotnik zmaga

        else:
            if sem_pobral:
                dy[Karta.v_id( igrana_karta )] = vrednost_stiha
            else:
                dy[Karta.v_id( igrana_karta )] = -vrednost_stiha#-(v + vrednost_stiha) / vrednost_stiha
        if len (self.trenutna_igra[id_igre] ) != 0:
            self.trenutna_igra[id_igre][-1] [3] = self.next_Q_max[id_igre]
        self.trenutna_igra[id_igre].append( [self.stanje[id_igre], dy,igrana_karta,None] )

    def rezultat_igre(self,st_tock,povzetek_igre,id_igre):
        if self.lic[id_igre] in [Tip_igre.Ena,Tip_igre.Dve,Tip_igre.Tri]:
            index_igre = self.igra2index[(self.lic[id_igre], self.barva_kralja[id_igre])]
        else:
            if self.lic[id_igre] == Tip_igre.Klop:
                index_igre = self.igra2index[(Tip_igre.Naprej, None)]
            else:
                index_igre = self.igra2index[(self.lic[id_igre], None)]
        self.roka2tocke.append( (self.zacetna_roka[id_igre],st_tock,index_igre))
        if self.zalozil[id_igre] is not None:
            self.zalaganje2tocke[id_igre][-2] = st_tock

        #final reword za beraca ker sicer je v vsaem rimeru 0 more zajebat tistega ki igra
        if self.tip_igre[id_igre] == 'Berac' and self.index_igralec_ki_igra[id_igre] != 3 and len(self.roka[id_igre]) == 0:
            st_tock = -20
        elif self.tip_igre[id_igre] == 'Berac' and self.index_igralec_ki_igra[id_igre] != 3 and len( self.roka[id_igre] ) != 0:
            st_tock = 20
        self.trenutna_igra[id_igre][-1][3] = st_tock # final reword

        for stanje,dy,igrana_karta,next_max in self.trenutna_igra[id_igre]:
            dy[igrana_karta.v_id()] = dy[igrana_karta.v_id()]+next_max*self.final_reword_factor
            (self.zgodovina.setdefault( (self.tip_igre[id_igre],stanje[0].shape[1]),[] )).append((stanje,dy))
        #print('rezultat_igre end',str(self.zgodovina[:1])[:10])

        self.trenutna_igra[id_igre] = []
        self.since_last_update = self.since_last_update +1

    def konec_licitiranja(self,igralec_ki_igra,tip_igre,id_igre,barva_kralja=None):
        self.tip_igre[id_igre] = Nevronski_igralec.tip_igre_v_tip_izbire[tip_igre]
        self.barva_kralja[id_igre] = barva_kralja
        self.index_igralec_ki_igra[id_igre] = self.igralci2index[id_igre][igralec_ki_igra]

    def stanje_v_vektor_rek_navadna(self,karte_na_mizi,mozne,zgodovina,id_igre):
        #[input_layer_nasprotiki, kralj, roka_input, talon_input, index_tistega_ki_igra, mozne]
        st_igranih_kart = 0
        for ig,k in zgodovina:
            if ig is not None or ig != 'Talon' :
                st_igranih_kart += 1

        st_igranih_kart = st_igranih_kart + (8-st_igranih_kart%8)
        roka = np.zeros(54)
        zalozil = np.zeros(54)
        if self.zalozil[id_igre] is not None:
            zalozil[ [k.v_id() for k in self.zalaganje2tocke[id_igre][1]] ] = 1
        roka[[k.v_id() for k in self.zacetna_roka[id_igre]]] = 1
        #barva_kralja
        barva_kralja = np.zeros(4)
        if self.barva_kralja[id_igre] is not None:
            barva_kralja[ int(self.barva_kralja[id_igre]) ] = 1

        #roka
        if st_igranih_kart > 0:
            roka_input = np.zeros((st_igranih_kart,54))
            input_layer_nasprotiki = np.zeros( (st_igranih_kart, 3, 54) )  # nasprotniki
        else:
            input_layer_nasprotiki = np.zeros( (1, 3, 54) )  # nasprotniki
            roka_input = np.zeros( (1, 54) )
            roka_input[0,:] = roka
        # TODO talon
        if self.tip_igre[id_igre] == "Navadna_igra":
            talon_input = np.zeros(( 6,55))
        elif self.tip_igre[id_igre] == 'Klop':
            talon_input = np.zeros((54,))
        elif self.tip_igre[id_igre] == 'Solo':
            talon_input = np.zeros(( 6,55))
        elif self.tip_igre[id_igre] == 'Berac':
            talon_input = None#np.zeros(( 6,55))
        else:
            raise Exception("Ni implementerano"+str(self.tip_igre))


        index_tistega_ki_igra = np.zeros(4)
        index_tistega_ki_igra[self.index_igralec_ki_igra[id_igre]] = 1
        mozne_vec = np.zeros( 54 )
        i = 0
        for  (igralec,k) in zgodovina:
            if igralec is None: #Talon klop:
                talon_input[k.v_id()] = 1
            elif igralec == "Talon": #navadna igra ko se talon odpre
                stevilka_kupcka, kupcki_talona = k
                stevec_karte_v_talonu  = 0
                for st_kupkca,kup in enumerate(kupcki_talona):
                    for k in kup:
                        talon_input[stevec_karte_v_talonu,k.v_id()] = 1
                        if st_kupkca== stevilka_kupcka:
                            talon_input[stevec_karte_v_talonu, 54] = 1
                        stevec_karte_v_talonu = stevec_karte_v_talonu+1
            elif igralec is not self:
                input_layer_nasprotiki[i,self.igralci2index[id_igre][igralec],k.v_id()] = 1
                i +=1
            elif igralec is self:
                roka_input[i,:] = roka
                roka[k.v_id()] = 0
                i += 1
            else:
                raise Exception("Case not covered")

        ids = [k.v_id() for k in mozne]
        mozne_vec[ids] = 1
        if self.tip_igre[id_igre] == "Navadna_igra":
            r =  [input_layer_nasprotiki, barva_kralja, roka_input, talon_input, index_tistega_ki_igra,zalozil, mozne_vec]
            return [np.expand_dims(n,axis=0) for n in r]
        elif self.tip_igre[id_igre] == "Solo":
            r =  [input_layer_nasprotiki, roka_input, talon_input, index_tistega_ki_igra,zalozil, mozne_vec]
            return [np.expand_dims(n,axis=0) for n in r]
        elif self.tip_igre[id_igre] == 'Klop':
            r = [input_layer_nasprotiki, roka_input, talon_input, mozne_vec]
            return [np.expand_dims( n, axis=0 ) for n in r]
        elif self.tip_igre[id_igre] == 'Berac':
            r = [input_layer_nasprotiki,roka_input,index_tistega_ki_igra,mozne_vec]
            return [np.expand_dims( n, axis=0 ) for n in r]
        else:
            raise Exception("Ni implementerano")

    def menjaj_talon_v_vektor(self,kupcki,id_igre):
        roka = np.zeros((1,54))
        talon = np.zeros((1,54,6))
        igra = np.zeros((1,15))
        igra[0,self.igra_zalozi2index[(self.lic[id_igre],self.barva_kralja[id_igre])]] = 1
        roka[0,[k.v_id() for k in self.roka[id_igre]]] = 1
        for i,k in enumerate(kupcki):
            talon[0,[karta.v_id() for karta in k],i] = 1
        return [roka,talon,igra,np.empty((1,1))]

    def nauci(self,save=True):
        time_train = time.time()
        loss_vals = []
        data_sizes = []
        batch_size = 32
        for (tip_igre,time_stamp),v in self.zgodovina.items():
            #v je list k ti shran vsa stanja z isto dolzino

            (stanje,y) = v[0]
            batch_size  = len(v)
            data_sizes.append(batch_size)
            X = []
            dy = np.zeros((len(v),54))
            for n in stanje:
                shape = list(n.shape)
                shape[0] = len(v)
                shape = tuple(shape)
                X.append(np.zeros(shape))
            del stanje
            #print([n.shape for n in X])
            for i,(stanje,y) in enumerate(v):
                dy[i,:] = y
                #print(time_stamp,stanje[0].shape)
                if len(stanje) != len(X):
                    raise  Exception( str( (len(stanje),len(X)) ))
                for j,x in enumerate(stanje):
                    try:
                        X[j][i,:] = x
                    except Exception as e:
                        raise Exception(str(e)+' '+ str((tip_igre,len(v))))
            #assert np.isnan( X ).any()
            if X[0].shape[0] < batch_size:
                continue
            X = [torch.from_numpy(xx).to(device=self.device,dtype=torch.float32)for xx in X[:-1]]

            with torch.no_grad():
                self.models[tip_igre].eval()
                self.models[tip_igre].to(device=self.device)
                p = self.models[tip_igre](X).to(device='cpu')
            self.models[tip_igre].train()
            non_z = dy.nonzero()
            p[non_z] = torch.from_numpy(dy[non_z]).to(device='cpu',dtype=torch.float32)
            X.append(p)
            X = [xx.to(device='cpu') for xx in X]
            if X[0].shape[0] < batch_size:
                continue
            #mp.set_start_method('spawn')
            t = TensorDataset(*X)
            data = torch.utils.data.DataLoader( t,batch_size=batch_size, num_workers=8, drop_last=True)

            early_stop_callback = EarlyStopping(
                monitor='loss',
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode='min'
            )
            #early_stop_callback=early_stop_callback,
            gc.collect()
            trainer = Trainer(early_stop_callback=early_stop_callback,gpus=1,weights_summary=None,progress_bar_refresh_rate=0)

            trainer.fit(self.models[tip_igre],data)

            #self.models[tip_igre].fit( X[:-1], p,epochs=10,verbose=0,use_multiprocessing=True )
            #loss_vals.append(hist.history['loss'][-1])
            #del hist
            '''
            kf = KFold(n_splits=max(len(v)//32,2) ,shuffle=True,)
            for train_index, test_index in kf.split( X[0] ):
                #ignore train index
                X_batch = [x_tensor[test_index] for x_tensor in X[:-1]]
                y_batch = p[test_index]
                for __ in range(5): # memory issues
                    #self.models[tip_igre].train_on_batch( X[:-1], p)
                    self.models[tip_igre].train_on_batch( X_batch, y_batch)
            '''

        "Train roka"
        if len(self.roka2tocke)>=batch_size:
            X = np.zeros((len(self.roka2tocke),54))
            for i,(r,t,tip) in enumerate(self.roka2tocke):
                X[i,[k.v_id() for k in r]] = 1
            X = torch.from_numpy(X).to(device=self.device,dtype=torch.float32)
            with torch.no_grad():
                self.models['Vrednotenje_roke'].eval()
                y = self.models['Vrednotenje_roke'](X)
            self.models[tip_igre].train()
            #assert np.isnan( y ).any() == False and np.isinf( y ).any() == False
            for i,(r,t,index_igre) in enumerate(self.roka2tocke):
                y[i,index_igre] = t
            #assert any([np.isnan(x).any() or np.isinf( x ).any() for x in X]) == False
            #assert np.isnan(y).any() == False and np.isinf(y).any() == False
            #hist = self.models['Vrednotenje_roke'].fit(X,y,epochs=10,verbose=0,use_multiprocessing=True )
            X.to(device='cpu')
            y.to(device='cpu')
            t = TensorDataset(*[X,y])
            if x.shape[0] >= batch_size:
                data = torch.utils.data.DataLoader(t, batch_size=batch_size, pin_memory=True, num_workers=4, drop_last=True)
                self.models['Vrednotenje_roke'].train()
                early_stop_callback = EarlyStopping(
                    monitor='loss',
                    min_delta=0.00,
                    patience=3,
                    verbose=False,
                    mode='min'
                )
                trainer =Trainer(early_stop_callback=early_stop_callback,gpus=1,weights_summary=None,progress_bar_refresh_rate=0)
                trainer.fit(self.models['Vrednotenje_roke'],data)

            # loss_vals.append( hist.history['loss'][-1] )
            # if not math.isfinite( hist.history['loss'][-1] ):
            #     print(hist.history['loss'][-1])
            #     print(self)
            #     raise AssertionError()
            # del hist

        '''Train zalozi'''
        n_training_samp = len( self.zalaganje2tocke.values() )
        if n_training_samp>batch_size:
            X = [np.zeros( (n_training_samp, 54) ), np.zeros( (n_training_samp, 54, 6) ),
                 np.zeros( (n_training_samp, 15) )]
            for i, (stanje, zalozi, st_kart, st_tock, mozno) in enumerate( self.zalaganje2tocke.values() ):
                for j in range( 3 ):
                    X[j][i, :] = stanje[j]
            X = [torch.from_numpy(xx).to(device=self.device,dtype=torch.float32) for xx in X ]
            with torch.no_grad():
                self.models['Zalaganje'].eval()
                y = self.models['Zalaganje']( X )
            self.models[tip_igre].train()
            X = [xx.to(device='cpu', dtype=torch.float32) for xx in X]
            y = y.to(device='cpu', dtype=torch.float32)
            for i, (stanje,zalozi,st_kart,st_tock,mozno) in enumerate( self.zalaganje2tocke.values()):
                y[i,[id for id in range(54) if Karta.iz_id(id) not in mozno]] = -70
                y[i, [k.v_id() for k in zalozi]] = st_tock
                y[i,54+6//st_kart:] = -70
            #assert np.isnan( y ).any() == False and np.isinf( y ).any() == False
            #assert any( [np.isnan( x ).any() or np.isinf( x ).any() for x in X] ) == False
            X.append(y)
            t = TensorDataset(*X)
            data = torch.utils.data.DataLoader(t, batch_size=batch_size, pin_memory=True, num_workers=7, drop_last=True)
            early_stop_callback = EarlyStopping(
                monitor='loss',
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode='min'
            )
            trainer = Trainer(early_stop_callback=early_stop_callback, gpus=1, weights_summary=None,
                              progress_bar_refresh_rate=0)
            trainer.fit(self.models['Zalaganje'], data)

            #hist = self.models['Zalaganje'].fit( X, y, epochs=10, verbose=0,use_multiprocessing=True  )
            #assert math.isfinite( hist.history['loss'][-1] )
            #loss_vals.append( hist.history['loss'][-1] )
            #del hist
        if save:
            self.save_models()
        self.zgodovina = {}
        self.roka2tocke = []
        self.zalaganje2tocke = {}
        time_train = time.time() - time_train
        self.final_reword_factor = min(self.final_reword_factor*1.1,0.99)
        self.random_card = max(0.05,self.random_card*0.9)
        mean_loss= np.mean( loss_vals )
        print(datetime.now(), str(self),'Time used:', time_train,self.since_last_update,'Mean number of data per fit:',np.mean(data_sizes),'Sum of data:',np.sum(data_sizes) )
        self.since_last_update = 0
        del loss_vals
        for _ in range(20):
            gc.collect()
        return mean_loss


    @staticmethod
    def generete_igra2index_and_index2igra():
        igra2index = {}
        index2igra = {}
        igra_zalozi2index = {}
        i = 0
        tip_igre = [t for t in Tip_igre ]
        tip_igre.sort() # da so zih zmer isti
        barve = [b for b in Barva]
        barve.sort()
        for t in tip_igre:
            if t in [Tip_igre.Tri,Tip_igre.Dve,Tip_igre.Ena]:
                for b in barve:
                    if b == Barva.TAROK:
                        continue
                    index2igra[i] = (t,b)
                    igra2index[(t,b)] = i
                    igra_zalozi2index[(t,b)] = len(igra_zalozi2index)
                    i = i+1
            elif t == Tip_igre.Klop or t == Tip_igre.Odprti_berac:
                print('Ignor odprti berac')
                continue #To bi blo isto k Naprej
            else:
                index2igra[i] = (t,None)
                igra2index[(t, None)] = i
                i = i+1
                if t in [Tip_igre.Solo_tri,Tip_igre.Solo_dve,Tip_igre.Solo_ena]:
                    igra_zalozi2index[(t, None)] = len( igra_zalozi2index )
        return igra2index,index2igra,igra_zalozi2index

    def save_models(self):
        pass
        #for k,v in self.models.items():
            #t = Trainer(v)
            #t.model
            #t.save_checkpoint(f)
            #v.save(os.path.join(self.save_path,k+".h5"),include_optimizer=True)

    def clean(self):
        self.zgodovina = {}
        self.roka2tocke =[]
        self.zalaganje2tocke ={}
        self.trenutna_igra = dict()
        self.zacetna_roka = dict()
        self.tip_igre = dict()
        self.moja_igra= dict()
        self.barva_kralja = dict()
        self.lic  = dict()
        self.stanje = dict()
        self.next_Q_max = dict()
        self.zalozil = dict()
        self.if_play_barva_kralja = dict()
        self.igrana_karta = dict()
        self.igralci2index = {}
        self.index_igralec_ki_igra = {}
        gc.collect()


class Double_Nevronski_Igralec(Nevronski_igralec):

    def __init__(self,load_path=None,save_path=None,random_card=0.05,learning_rate=0.1,final_reword_factor=0.1,ime=None):
        super().__init__(load_path,save_path,random_card,learning_rate,final_reword_factor,ime,True)
        if load_path is not None:
            print( 'Load models' )
            try:
                self.load_models()
            except Exception as e:
                print( 'Problem ', e, 'loading models create new one' )
                self.models ,self.models_B = self.create_models()
        else:
            self.load_path = self.save_path
            print( 'Create new models' )
            self.models ,self.models_B = self.create_models()

    def create_models(self):
        d = {'Navadna_igra': Net_Navadna_igra(), 'Klop': Net_Klop(), 'Solo': Net_Solo(), 'Berac': Net_Berac()}
        for v in d.values():
            v.to(device=self.device)
        return  super().create_models(),d
    def del_models(self):
        del self.models
        del self.models_B
        self.models = None
        self.models_B = None

    def save_models(self):
        for k,v in self.models.items():
            if k in ['Vrednotenje_roke', 'Zalaganje']:
                PATH = os.path.join( self.save_path, k + ".pth" )
            else:
                PATH = os.path.join(self.save_path,k+"_A.pth")
            torch.save(v.state_dict(), PATH)
        for k,v in self.models_B.items():
            torch.save(v.state_dict(), os.path.join(self.save_path,k+"_B.pth"))

    def load_models(self):
        self.models, self.models_B = self.create_models()
        for f in os.listdir( self.load_path ):

            state_dict = torch.load(os.path.join( self.load_path, f ))

            if f[-6:-4] == '_A':
                self.models[f[:-6]].load_state_dict(state_dict)#tf.keras.models.load_model( os.path.join( self.load_path, f ), compile=True )
            elif f[-6:-4] == '_B':
                self.models_B[f[:-6]].load_state_dict(state_dict)
            elif f[:-4] in ['Vrednotenje_roke','Zalaganje']:
                self.models[f[:-4]].load_state_dict(state_dict)
            else:
                raise IOError( "Napaka pri loadanju modelov. "+str(os.path.join( self.load_path, f )) )

    def nauci(self,save=True):
        mean_loss = super().nauci(False)
        self.models, self.models_B = self.models_B, self.models  # sweap two models duo to duble q-learnig
        self.models['Vrednotenje_roke'] = self.models_B['Vrednotenje_roke']
        self.models['Zalaganje'] = self.models_B['Zalaganje']
        del self.models_B['Vrednotenje_roke']
        del self.models_B['Zalaganje']
        if save:
            self.save_models()
        for m in self.models.values():
            m.eval()
        for m in self.models_B.values():
            m.eval()
        gc.collect()
        return mean_loss


    def predict_igraj_karto(self):
        with torch.no_grad():
            for tip_igre,X_list in self.predict_queue.items():
                if len(X_list) == 0:
                    continue
                stanje = X_list[0][1]
                X = []
                for n in stanje:
                    shape = list( n.shape )
                    shape[0] = len( X_list )
                    shape = tuple( shape )
                    X.append( np.zeros( shape ) )
                del stanje
                for i,(id, stanje) in  enumerate(X_list) :
                    for j, x in enumerate( stanje ):
                        X[j][i, :] = x
                X = [torch.from_numpy(xx).to(device=self.device,dtype=torch.float32)for xx in X[:-1]]
                Y = self.models[tip_igre]( X )
                if tip_igre == 'Zalaganje':
                    for i, (id, stanje) in enumerate( X_list ):
                        self.predicted_resoult[tip_igre][id] = Y[i, :]
                else:
                    Y_b = self.models_B[tip_igre]( X )
                   # print('Napovedal',tip_igre)
                    for i,(id, stanje) in enumerate(X_list):
                        self.predicted_resoult[tip_igre][id] =(Y[i,:],Y_b[i,:])

        for k in self.predict_queue.keys():
            self.predict_queue[k]  = []

    def igraj_karto(self,karte_na_mizi,mozne,zgodovina,id_igre):
        # self.t += time.time()-t
        p ,p_b= self.predicted_resoult[self.tip_igre[id_igre]][id_igre]
        p = p.cpu()
        p_b = p_b.cpu()
        #p = self.models[self.tip_igre].predict( self.stanje[:-1],batch_size=1,use_multiprocessing=True )[0]
        mozne_id = [k.v_id() for k in mozne]
        id = np.argmax( p[mozne_id] )
        karta = mozne[id]
        self.next_Q_max[id_igre] = p_b[karta.v_id()]
        if random.random() < self.random_card:
            karta = random.choice( mozne )
        self.igrana_karta[id_igre] = karta
        return Igralec.igraj_karto(self,karta,id_igre)


