import os

import gc
import pickle
import time
from datetime import datetime

import tensorflow as tf
from keras.engine.saving import load_model

import train
from Igralec import Bot_igralec, Nevronski_igralec
from Tarok import Tarok
#TODO pohendli ce so gor skis 21 in palcka da palcka pobere

def main():
    # naredi 4 igralce
    # train.muilt_model()
    # train.model_za_vrednotenje_roke()

    #igralci = [Bot_igralec() for i in range( 3 )]

    print('try to load scores')

    with open( r"scores.pickle", "rb" ) as input_file:
        scores = pickle.load( input_file )
        #print('scores loaded')
        #scores2 =[ {str(k.ime): int(v) for k,v in s.items()} for s in scores]
    #model = load_model('model_conv.h5')
    igralci = []#[Bot_igralec() for i in range( 3 )]
    igralci.append(Nevronski_igralec(load_path='models2/',save_path='models2/',ime='Igralec_1'))
    igralci.append(Nevronski_igralec(load_path='models3/',save_path='models3/',ime='Igralec_2'))
    igralci.append(Nevronski_igralec(load_path='models4/',save_path='models4/',ime='Igralec_3'))
    igralci.append(Nevronski_igralec(load_path='models/',save_path='models/',ime='Igralec_4'))
    #igralci.append(Nevronski_igralec(None))

    #igralci.append( nn )
    #igralci.append( Bot_igralec() )
    # igralci.append(Clovekski_igralec())
    """
    igra = Igra(igralci)
    igra.start()

    """
    num_games= 1000
    for i in range( 1000 ):
        t = time.time()
        tarok = Tarok( igralci, num_games)
        tarok.start()
        t = time.time()-t
        print('Time need for',num_games,'games:',t)
        scores.append({str(k):v for k,v in tarok.rezultati.items()})

        print(datetime.now(),':Vsi scori:')
        for r in scores:
            print(r)

        with open( r"scores.pickle", "wb" ) as output_file:
            pickle.dump(scores,output_file)

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices( 'GPU' )
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth( gpu, True )

    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #train.model_za_vrednotenje_roke()
    #train.test_klop()

    main()
    #print( tf.__version__ )
    #train.test_navadna_mreza()
    #train.model_igraj()