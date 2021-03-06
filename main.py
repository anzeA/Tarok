import os
import random

import torch
import gc
import pickle
import time
from datetime import datetime
from random import shuffle
#import tensorflow as tf
#import train
from Igralec import Bot_igralec, Nevronski_igralec, Double_Nevronski_Igralec
from Tarok import Tarok
import numpy as np
import shutil
from matplotlib import pyplot as plt
import pytorch_lightning as pl

def plot_score(path):
    plt.clf()
    plt.close()
    with open( os.path.join( path, 'scores.pickle' ), "rb" ) as input_file:
        scores = pickle.load( input_file )
    with open( os.path.join( path, 'loss.pickle' ), "rb" ) as input_file:
        loss = pickle.load( input_file )
    keys = list(scores[0].keys())
    keys.sort()
    for k in keys:
        plt.plot([d[k] for d in scores])
    plt.xlabel('Igra')
    plt.ylabel('Stevilo tock')
    plt.title('Igra za bazo:'+str(path))
    plt.legend(keys)
    plt.show()
    plt.clf()
    '''
    keys = list(loss[0].keys())
    keys.sort()
    for k in keys:
        plt.plot([d[k] for d in loss if d[k] is not None ])
    plt.xlabel('Igra')
    plt.ylabel('loss')
    plt.legend( keys )
    plt.title('Loss za bazo:'+str(path))
    plt.show()
    '''
#TODO pohendli ce so gor skis 21 in palcka da palcka pobere
def naredi_nove_igralce_debug(path,**kwargs):
    os.mkdir(path)
    nn = []
    with open( os.path.join(path,'scores.pickle'), "wb" ) as output_file:
        pickle.dump( [], output_file )

    with open( os.path.join(path,'kwargs.pickle'), "wb" ) as output_file:
        pickle.dump( dict(), output_file )
    with open( os.path.join(path,'loss.pickle'), "wb" ) as output_file:
        pickle.dump( [], output_file )
    p = os.path.join(path,str(1))
    os.mkdir(p)
    #nn.append(Double_Nevronski_Igralec(load_path=None,save_path=p,**kwargs))
    nn.append(Nevronski_igralec(load_path=None,save_path=p,**kwargs))
    nn.append(Bot_igralec())
    nn.append(Bot_igralec())
    nn.append(Bot_igralec())
    return nn,os.path.join(path,'scores.pickle'),os.path.join(path,'kwargs.pickle'),os.path.join(path,'loss.pickle')



def naredi_nove_igralce(path,**kwargs):
    os.mkdir(path)
    nn = []
    with open( os.path.join(path,'scores.pickle'), "wb" ) as output_file:
        pickle.dump( [], output_file )

    with open( os.path.join(path,'loss.pickle'), "wb" ) as output_file:
        pickle.dump( [], output_file )

    with open( os.path.join(path,'kwargs.pickle'), "wb" ) as output_file:
        pickle.dump( dict(), output_file )
    for i in range(1,5):
        p = os.path.join(path,str(i))
        os.mkdir(p)
        nn.append(Double_Nevronski_Igralec(load_path=None,save_path=p,ime=i,**kwargs))
    return nn,os.path.join(path,'scores.pickle'),os.path.join(path,'kwargs.pickle'),os.path.join(path,'loss.pickle')

def load_igralce(path):
    #with open( os.path.join( path, 'scores.pickle' ), "rb" ) as input_file:
    #    scores = pickle.load( input_file )
    with open( os.path.join( path, 'kwargs.pickle' ), "rb" ) as input_file:
        kwargs = pickle.load( input_file )
    os.path.join( path, 'scores.pickle' )
    #return [Nevronski_igralec(load_path=os.path.join(path,str(i)),save_path=os.path.join(path,str(i)),ime=i ,**kwargs) for i in range(1,5)] ,os.path.join( path, 'scores.pickle' ),os.path.join(path,'kwargs.pickle')
    return [Double_Nevronski_Igralec(load_path=os.path.join(path,str(i)),save_path=os.path.join(path,str(i)),ime=i ,**kwargs) for i in range(1,5)] ,os.path.join( path, 'scores.pickle' ),os.path.join(path,'kwargs.pickle'),os.path.join(path,'loss.pickle')

def main(dir,**kwargs):
    if dir in os.listdir():
        igralci,scores_file,kwargs_file,loss_file = load_igralce(dir)
    else:
        debug = kwargs.setdefault('debug',False)
        del kwargs['debug']
        if debug:
            igralci,scores_file,kwargs_file,loss_file = naredi_nove_igralce_debug(dir,**kwargs)
        else:
            igralci,scores_file,kwargs_file,loss_file = naredi_nove_igralce(dir,**kwargs)
    with open( scores_file, "rb" ) as input_file:
        scores = pickle.load( input_file )
    with open( loss_file, "rb" ) as input_file:
        loss = pickle.load( input_file )

    num_games= 2000
    for i in range( 1000 ):
        shuffle(igralci)
        t = time.time()
        tarok = Tarok( igralci, num_games)
        tarok.paralel_start()
        print('Time need for',num_games,'game:',time.time()-t)
        loss.append({})
        for igr in igralci:
            if isinstance(igr,Nevronski_igralec):
                loss_tmp = igr.nauci()
                loss[-1][str(igr)] = loss_tmp
                igr.clean() #clean old data empty keys and so on
        '''
        for igr in igralci:
            if isinstance(igr,Nevronski_igralec):
                igr.del_models()
        tf.keras.backend.clear_session()
        for _ in range(50):
            gc.collect()
        for igr in igralci:
            if isinstance(igr,Nevronski_igralec):
                igr.load_models()
        '''
        t = time.time()-t
        print('Time need all:',t)
        scores.append({str(k):v for k,v in tarok.rezultati.items()})

        if scores_file is not None:
            with open( scores_file, "wb" ) as output_file:
                pickle.dump(scores,output_file)
        if loss_file is not None:
            with open( loss_file, "wb" ) as output_file:
                pickle.dump(loss,output_file)
        if kwargs_file is not None:
            igr_param = [igr for igr in igralci if isinstance( igr, Nevronski_igralec )][0]
            with open( kwargs_file, "wb" ) as output_file:
                pickle.dump( { 'final_reword_factor' : igr_param.final_reword_factor, 'random_card':igr_param.random_card},output_file)

        print(datetime.now(),':Vsi scori:')
        for r in scores[-10:]:
            keys = list(r.keys())
            keys.sort()
            print([str(k)+': '+str(r[k]) for k in keys])
        if len(scores) > 0 and len(loss) >0:
            plot_score(dir)
        shutil.rmtree('lightning_logs')


if __name__ == '__main__':


    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #train.model_za_vrednotenje_roke()
    #train.test_klop()

    #main('scores_2.pickle',models_files=['modeli_lr_0.01/model'+str(i)+'/' for i in range(1,5)])
    #main(*naredi_nove_igralce('test_max'))
    print(torch.__version__)
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #main('test_nan',learning_rate=0.01,debug=True)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #random.seed(42)
    #main('test_double_1e-2_solo',learning_rate=0.01,debug=False) # Nadaljuj super rezultati
    #main('test_double_1e-2_solo',learning_rate=0.01,debug=False) # Nadaljuj super rezultati
    #main('Pararel_Adam_1e-2',learning_rate=0.01,debug=False) # Nadaljuj super rezultati
    #Nevronski_igralec(save_path='Torch_test')
    #shutil.rmtree('Torch_test')
    main('Torch_first_run',learning_rate=0.01,debug=False)