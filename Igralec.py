import gc
import math
import os
import random
import time
import warnings
from copy import deepcopy
from datetime import datetime
from enum import Enum

#from keras.optimizers import SGD
#from sklearn.model_selection import KFold
import numpy as np
#from keras.callbacks import EarlyStopping
#from keras.engine.saving import load_model
import tensorflow as tf
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

    def izbral_iz_talona(self,talon,st_kupcka):
        pass

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
    def __init__(self,load_path=None,save_path=None,random_card=0.9,learning_rate=0.1,final_reword_factor=0.1,ime=None,ignor_models=False):
        super().__init__(ime)
        self.zgodovina = {} #cele igre
        self.roka2tocke =[]
        self.zalaganje2tocke =[]
        self.trenutna_igra = [] #trenutna igra (stanje nagrada)
        self.zacetna_roka = None
        self.mozne = None
        self.tip_igre = None
        self.moja_igra= None # tmp za shranjevanje podatkov kaj sem igral
        self.barva_kralja = None
        self.lic  = None
        self.stanje = None
        self.next_Q_max = None
        self.zalozil = None
        self.learning_rate= learning_rate
        self.igralci2index = {}
        self.save_path = save_path
        self.random_card = random_card
        self.final_reword_factor = final_reword_factor
        self.load_path = load_path
        self.igra2index,self.index2igra,self.igra_zalozi2index = Nevronski_igralec.generete_igra2index_and_index2igra()
        self.t = 0
        self.since_last_update = 0
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

    def create_models(self):
        return {'Navadna_igra': train.test_navadna_mreza(self.learning_rate), 'Klop': train.test_klop(self.learning_rate),'Solo':train.test_solo(self.learning_rate),'Berac':train.test_berac(self.learning_rate),
                                   'Vrednotenje_roke': train.model_za_vrednotenje_roke(self.learning_rate),'Zalaganje':train.model_za_zalaganje(self.learning_rate)}  # ,'Berac':train.test_berac(),'Solo':train.test_solo(),

    def load_models(self):
        self.models = dict()
        for f in os.listdir( self.load_path ):
            self.models[f[:-3]] = tf.keras.models.load_model( os.path.join( self.load_path, f ) ,compile=False)
            self.models[f[:-3]].compile( optimizer=tf.keras.optimizers.Adadelta(lr=self.learning_rate),
                   loss=tf.keras.losses.Huber(delta=15.0),
                   metrics=['accuracy'] )

    def del_models(self):
        del self.models
        self.models = None

    def nova_igra(self,roka,igralci):
        self.tip_igre = None
        self.lic = None
        self.next_Q_max = None
        self.zalozil = None
        self.stanje = None
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

            if random.random() < self.random_card:
                self.lic, self.if_play_barva_kralja = random.choice( list(self.index2igra.values()) )

        if self.lic == Tip_igre.Odprti_berac:
            self.lic,self.if_play_barva_kralja = Tip_igre.Berac,None
        self.lic = super().licitiram( self.lic, min_igra, obvezno, prednost )
        #print(self.lic)
        return self.lic

    def izberi_barvo_kralja(self):
        assert self.if_play_barva_kralja is not None
        return self.if_play_barva_kralja

    def igraj_karto(self,karte_na_mizi,mozne,zgodovina):
        self.stanje = self.stanje_v_vektor_rek_navadna( karte_na_mizi, mozne, zgodovina )
        # self.t += time.time()-t
        p = self.models[self.tip_igre].predict_on_batch( self.stanje[:-1] )[0]
        mozne_id = [k.v_id() for k in mozne]
        id = np.argmax( p[mozne_id] )

        karta = mozne[id]
        self.next_Q_max = p[karta.v_id()]
        if random.random() < self.random_card:
            karta = random.choice( mozne )
        self.igrana_karta = karta
        return super().igraj_karto(karta)

    def menjaj_iz_talona(self,kupcki,st_kart):
        stanje = self.menjaj_talon_v_vektor(kupcki)
        p = self.models['Zalaganje'].predict_on_batch(stanje)[0]
        izbrani_kupcek = np.argmax(p[54:54+len(kupcki)])

        if random.random() < self.random_card:
            izbrani_kupcek = random.randint(0,len(kupcki)-1)

        self.roka.dodaj_karte(kupcki[izbrani_kupcek])
        mozno = self.roka.mozno_zalozit()
        zalozi = [k.v_id() for k in mozno]
        if random.random() < self.random_card:
            zalozi = random.sample(mozno,st_kart)
        else:
            zalozi = [mozno[i] for i in np.argsort(p[zalozi])[-st_kart:]]
        self.kupcek.extend(zalozi)
        self.zalozil = True
        for k in zalozi:
            self.roka.igraj_karto(k)

        self.zalaganje2tocke.append([stanje,zalozi,st_kart,None,mozno])
        return izbrani_kupcek

    def rezultat_stiha(self, stih, sem_pobral): #crate train data
        #print('Rezultat stiha',stih,sem_pobral)
        vrednost_stiha = Roka.vrednost_stiha(stih)
        v = self.igrana_karta.vrednost()

        dy = np.zeros(54)#self.p
        dy[self.stanje[-1][0]== 0] = -70
        if self.tip_igre ==  "Klop":
            if sem_pobral:
                dy[Karta.v_id( self.igrana_karta )] = -vrednost_stiha
            else:
                dy[Karta.v_id( self.igrana_karta )] = vrednost_stiha
        elif self.tip_igre == 'Berac':
            if self.index_igralec_ki_igra ==3: # jaz igram
                if sem_pobral:
                    dy[Karta.v_id( self.igrana_karta )] = -1
                else:
                    dy[Karta.v_id( self.igrana_karta )] = 1
            else:
                if sem_pobral:
                    dy[Karta.v_id( self.igrana_karta )] = -1
                else:
                    dy[Karta.v_id( self.igrana_karta )] = 1
                # ce nasprotnik zmaga

        else:
            if sem_pobral:
                dy[Karta.v_id( self.igrana_karta )] = vrednost_stiha
            else:
                dy[Karta.v_id( self.igrana_karta )] = -vrednost_stiha#-(v + vrednost_stiha) / vrednost_stiha
        if len (self.trenutna_igra ) != 0:
            self.trenutna_igra[-1] [3] = self.next_Q_max
        self.trenutna_igra.append( [self.stanje, dy,self.igrana_karta,None] )

    def rezultat_igre(self,st_tock,povzetek_igre):
        if self.lic in [Tip_igre.Ena,Tip_igre.Dve,Tip_igre.Tri]:
            index_igre = self.igra2index[(self.lic, self.barva_kralja)]
        else:
            if self.lic == Tip_igre.Klop:
                index_igre = self.igra2index[(Tip_igre.Naprej, None)]
            else:
                index_igre = self.igra2index[(self.lic, None)]
        self.roka2tocke.append( (self.zacetna_roka,st_tock,index_igre))
        if self.zalozil is not None:
            self.zalaganje2tocke[-1][-2] = st_tock

        #final reword za beraca ker sicer je v vsaem rimeru 0 more zajebat tistega ki igra
        if self.tip_igre == 'Berac' and self.index_igralec_ki_igra != 3 and len(self.roka) == 0:
            st_tock = -20
        elif self.tip_igre == 'Berac' and self.index_igralec_ki_igra != 3 and len( self.roka ) != 0:
            st_tock = 20
        self.trenutna_igra[-1][3] = st_tock # final reword

        for stanje,dy,igrana_karta,next_max in self.trenutna_igra:
            dy[igrana_karta.v_id()] = dy[igrana_karta.v_id()]+next_max*self.final_reword_factor
            (self.zgodovina.setdefault( (self.tip_igre,stanje[0].shape[1]),[] )).append((stanje,dy))
        #print('rezultat_igre end',str(self.zgodovina[:1])[:10])

        self.trenutna_igra = []
        self.since_last_update = self.since_last_update +1

    def konec_licitiranja(self,igralec_ki_igra,tip_igre,barva_kralja=None):
        self.tip_igre = Nevronski_igralec.tip_igre_v_tip_izbire[tip_igre]
        self.barva_kralja = barva_kralja
        self.index_igralec_ki_igra = self.igralci2index[igralec_ki_igra]

    def stanje_v_vektor_rek_navadna(self,karte_na_mizi,mozne,zgodovina):
        #[input_layer_nasprotiki, kralj, roka_input, talon_input, index_tistega_ki_igra, mozne]
        st_igranih_kart = 0
        for ig,k in zgodovina:
            if ig is not None or ig != 'Talon' :
                st_igranih_kart += 1
        roka = np.zeros(54)
        roka[[k.v_id() for k in self.zacetna_roka]] = 1
        #barva_kralja
        barva_kralja = np.zeros(4)
        if self.barva_kralja is not None:
            barva_kralja[ int(self.barva_kralja) ] = 1

        #roka
        if st_igranih_kart > 0:
            roka_input = np.zeros((st_igranih_kart,54))
            input_layer_nasprotiki = np.zeros( (st_igranih_kart, 3, 54) )  # nasprotniki
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
                input_layer_nasprotiki[i,self.igralci2index[igralec],k.v_id()] = 1
                i +=1
            elif igralec is self:
                roka_input[i,:] = roka
                roka[k.v_id()] = 0
                i += 1
            else:
                raise Exception("Case not covered")

        ids = [k.v_id() for k in mozne]
        mozne_vec[ids] = 1
        if self.tip_igre == "Navadna_igra":
            r =  [input_layer_nasprotiki, barva_kralja, roka_input, talon_input, index_tistega_ki_igra, mozne_vec]
            return [np.expand_dims(n,axis=0) for n in r]
        elif self.tip_igre == "Solo":
            r =  [input_layer_nasprotiki, roka_input, talon_input, index_tistega_ki_igra, mozne_vec]
            return [np.expand_dims(n,axis=0) for n in r]
        elif self.tip_igre == 'Klop':
            r = [input_layer_nasprotiki, roka_input, talon_input, mozne_vec]
            return [np.expand_dims( n, axis=0 ) for n in r]
        elif self.tip_igre == 'Berac':
            r = [input_layer_nasprotiki,roka_input,index_tistega_ki_igra,mozne_vec]
            return [np.expand_dims( n, axis=0 ) for n in r]
        else:
            raise Exception("Ni implementerano")

    def menjaj_talon_v_vektor(self,kupcki):
        l = []
        roka = np.zeros((1,54))
        talon = np.zeros((1,54,6))
        igra = np.zeros((1,15))
        igra[0,self.igra_zalozi2index[(self.lic,self.barva_kralja)]] = 1
        roka[0,[k.v_id() for k in self.roka]] = 1
        for i,k in enumerate(kupcki):
            talon[0,[karta.v_id() for karta in k],i] = 1
        return [roka,talon,igra]

    def nauci(self,save=True):
        time_train = time.time()
        loss_vals = []
        data_sizes = []
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
                    raise  Exception( str(len(stanje),len(X)))
                for j,x in enumerate(stanje):
                    try:
                        X[j][i,:] = x
                    except Exception as e:
                        raise Exception(str(e)+' '+ str((tip_igre,len(v))))
            #assert np.isnan( X ).any()
            p = self.models[tip_igre].predict_on_batch(X[:-1])
            non_z = dy.nonzero()
            p[non_z] = dy[non_z]
            p = p*X[-1] # krat mozne

            hist =self.models[tip_igre].fit( X[:-1], p,epochs=10,verbose=0,use_multiprocessing=True )
            loss_vals.append(hist.history['loss'][-1])
            del hist
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
        X = np.zeros((len(self.roka2tocke),54))
        for i,(r,t,tip) in enumerate(self.roka2tocke):
            X[i,[k.v_id() for k in r]] = 1
        y = self.models['Vrednotenje_roke'].predict_on_batch(X)
        assert np.isnan( y ).any() == False and np.isinf( y ).any() == False
        for i,(r,t,index_igre) in enumerate(self.roka2tocke):
            y[i,index_igre] = t
        assert any([np.isnan(x).any() or np.isinf( x ).any() for x in X]) == False
        assert np.isnan(y).any() == False and np.isinf(y).any() == False
        hist = self.models['Vrednotenje_roke'].fit(X,y,epochs=10,verbose=0,use_multiprocessing=True )
        loss_vals.append( hist.history['loss'][-1] )
        if not math.isfinite( hist.history['loss'][-1] ):
            print(hist.history['loss'][-1])
            print(self)
            raise AssertionError()
        del hist

        '''Train zalozi'''
        n_training_samp = len( self.zalaganje2tocke )
        X = [np.zeros( (n_training_samp, 54 ) ),np.zeros( (n_training_samp,54,6) ),np.zeros( (n_training_samp,15) ) ]
        for i, (stanje,zalozi,st_kart,st_tock,mozno) in enumerate( self.zalaganje2tocke ):
            for j in range(3):
                X[j][i,:] = stanje[j]

        y = self.models['Zalaganje'].predict_on_batch( X )
        assert np.isnan( y ).any() == False and np.isinf( y ).any() == False
        assert any( [np.isnan( x ).any() or np.isinf( x ).any() for x in X] ) == False

        for i, (stanje,zalozi,st_kart,st_tock,mozno) in enumerate( self.zalaganje2tocke ):
            y[i,[id for id in range(54) if Karta.iz_id(id) not in mozno]] = -70
            y[i, [k.v_id() for k in zalozi]] = st_tock
            y[i,54+6//st_kart:] = -70
        assert np.isnan( y ).any() == False and np.isinf( y ).any() == False
        assert any( [np.isnan( x ).any() or np.isinf( x ).any() for x in X] ) == False

        hist = self.models['Zalaganje'].fit( X, y, epochs=10, verbose=0,use_multiprocessing=True  )
        assert math.isfinite( hist.history['loss'][-1] )
        loss_vals.append( hist.history['loss'][-1] )
        del hist
        if save:
            self.save_models()
        self.zgodovina = {}
        self.roka2tocke = []
        self.zalaganje2tocke = []
        time_train = time.time() - time_train
        self.final_reword_factor = min(self.final_reword_factor*1.1,0.99)
        self.random_card = max(0.05,self.random_card*0.9)
        mean_loss= np.mean( loss_vals )
        print(datetime.now(), str(self),'Time used:', time_train,self.since_last_update, 'Mean_loss:',mean_loss,'Mean number of data per fit:',np.mean(data_sizes) )
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
        for k,v in self.models.items():
            v.save(os.path.join(self.save_path,k+".h5"),include_optimizer=False)


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
        return  super().create_models(),{'Navadna_igra': train.test_navadna_mreza(self.learning_rate), 'Klop': train.test_klop(self.learning_rate),'Solo':train.test_solo(self.learning_rate),'Berac':train.test_berac(self.learning_rate)}  # ,'Berac':train.test_berac(),'Solo':train.test_solo(),

    def del_models(self):
        del self.models
        del self.models_B
        self.models = None
        self.models_B = None

    def save_models(self):
        for k,v in self.models.items():
            if k in ['Vrednotenje_roke', 'Zalaganje']:
                v.save( os.path.join( self.save_path, k + ".h5" ), include_optimizer=False )
            else:
                v.save(os.path.join(self.save_path,k+"_A.h5"),include_optimizer=False)
        for k,v in self.models_B.items():
            v.save(os.path.join(self.save_path,k+"_B.h5"),include_optimizer=False)

    def load_models(self):
        self.models = dict()
        self.models_B = dict()
        for f in os.listdir( self.load_path ):
            if f[-5:-3] == '_A':
                self.models[f[:-5]] = tf.keras.models.load_model( os.path.join( self.load_path, f ), compile=False )
                self.models[f[:-5]].compile( optimizer=tf.keras.optimizers.Adadelta(lr=self.learning_rate),
                   loss=tf.keras.losses.Huber(delta=15.0),
                   metrics=['accuracy'] )
            elif f[-5:-3] == '_B':
                self.models_B[f[:-5]] = tf.keras.models.load_model( os.path.join( self.load_path, f ), compile=False )
                self.models_B[f[:-5]].compile( optimizer=tf.keras.optimizers.Adadelta(lr=self.learning_rate),
                   loss=tf.keras.losses.Huber(delta=15.0),
                   metrics=['accuracy'] )
            elif f[:-3]  in ['Vrednotenje_roke','Zalaganje']:
                self.models[f[:-3]] = tf.keras.models.load_model( os.path.join( self.load_path, f ), compile=False )
                self.models[f[:-3]].compile( optimizer=tf.keras.optimizers.Adadelta(lr=self.learning_rate),
                   loss=tf.keras.losses.Huber(delta=15.0),
                   metrics=['accuracy'] )
            else:
                raise IOError( "Napaka pri loadanju modelov. "+str(f) )

    def nauci(self,save=True):
        super().nauci(False)
        self.models, self.models_B = self.models_B, self.models  # sweap two models duo to duble q-learnig
        self.models['Vrednotenje_roke'] = self.models_B['Vrednotenje_roke']
        self.models['Zalaganje'] = self.models_B['Zalaganje']
        del self.models_B['Vrednotenje_roke']
        del self.models_B['Zalaganje']
        if save:
            self.save_models()

    def igraj_karto(self,karte_na_mizi,mozne,zgodovina):
        self.stanje = self.stanje_v_vektor_rek_navadna( karte_na_mizi, mozne, zgodovina )
        # self.t += time.time()-t
        p = self.models[self.tip_igre].predict_on_batch( self.stanje[:-1] )[0]
        p_b = self.models_B[self.tip_igre].predict_on_batch( self.stanje[:-1] )[0]
        mozne_id = [k.v_id() for k in mozne]
        id = np.argmax( p[mozne_id] )

        karta = mozne[id]
        self.next_Q_max = p_b[karta.v_id()]
        if random.random() < self.random_card:
            karta = random.choice( mozne )
        self.igrana_karta = karta
        return Igralec.igraj_karto(self,karta)

