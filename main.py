import os

import gc
import pickle
import time
from datetime import datetime
from random import shuffle
import tensorflow as tf
from keras.engine.saving import load_model

import train
from Igralec import Bot_igralec, Nevronski_igralec
from Tarok import Tarok
#TODO pohendli ce so gor skis 21 in palcka da palcka pobere

def naredi_nove_igralce(path):
    os.mkdir(path)
    nn = []
    with open( os.path.join(path,'scores.pickle'), "wb" ) as output_file:
        pickle.dump( [], output_file )

    with open( os.path.join(path,'kwargs.pickle'), "wb" ) as output_file:
        pickle.dump( dict(), output_file )
    for i in range(1,5):
        p = os.path.join(path,str(i))
        os.mkdir(p)
        nn.append(Nevronski_igralec(load_path=None,save_path=p,ime=i))
    return nn,os.path.join(path,'scores.pickle'),os.path.join(path,'kwargs.pickle')

def load_igralce(path):
    with open( os.path.join( path, 'scores.pickle' ), "rb" ) as input_file:
        scores = pickle.load( input_file )
    with open( os.path.join( path, 'kwargs.pickle' ), "rb" ) as input_file:
        kwargs = pickle.load( input_file )
    os.path.join( path, 'scores.pickle' )
    return [Nevronski_igralec(load_path=os.path.join(path,str(i)),save_path=os.path.join(path,str(i)),ime=i ,**kwargs) for i in range(1,5)] ,os.path.join( path, 'scores.pickle' ),os.path.join(path,'kwargs.pickle')

def main(dir):
    if dir in os.listdir():
        igralci,scores_file,kwargs_file = load_igralce(dir)
    else:
        igralci,scores_file,kwargs_file = naredi_nove_igralce(dir)
    with open( scores_file, "rb" ) as input_file:
        scores = pickle.load( input_file )

    num_games= 500
    for i in range( 1000 ):
        shuffle(igralci)
        t = time.time()
        tarok = Tarok( igralci, num_games)
        tarok.start()
        for igr in igralci:
            if isinstance(igr,Nevronski_igralec):
                igr.nauci()
        t = time.time()-t
        print('Time need for',num_games,'games:',t)
        scores.append({str(k):v for k,v in tarok.rezultati.items()})

        if scores_file is not None:
            with open( scores_file, "wb" ) as output_file:
                pickle.dump(scores,output_file)
        if kwargs_file is not None:
            with open( kwargs_file, "wb" ) as output_file:
                pickle.dump( { 'final_reword_factor' : igralci[0].final_reword_factor },output_file)

        print(datetime.now(),':Vsi scori:')
        for r in scores:
            print(r)

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices( 'GPU' )
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth( gpu, True )

    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #train.model_za_vrednotenje_roke()
    #train.test_klop()

    #main('scores_2.pickle',models_files=['modeli_lr_0.01/model'+str(i)+'/' for i in range(1,5)])
    #main(*naredi_nove_igralce('test_max'))
    main('test_max')
    #print( tf.__version__ )
    #train.test_navadna_mreza()
    #train.model_igraj()