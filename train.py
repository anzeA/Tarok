import numpy as np
from keras import Input, Model
from keras.layers import Dense, Conv1D, Multiply, Concatenate, Flatten, LSTM, TimeDistributed, CuDNNLSTM, CuDNNGRU, \
    Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.utils import plot_model


def model_za_vrednotenje_roke():
    model = Sequential(name='Vrednotenje_roke')
    model.add( Dense( 32, activation='elu', input_shape=(54,) ) )
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    #model.add( Dense( 30, activation='relu'))
    model.add( Dense( 16, activation='elu' ) )
    model.add( BatchNormalization() )
    #model.add( Dense( 10, activation='relu' ) )
    model.add( Dense(
        1+ #Naprej
        4*3 # tri dva ena v vseh sterih kraljih
        #3+ # 3 solo
        #2+ #odprti in zaprti berac
        #1#solo brez
        , activation='linear' ) )

    model.compile( optimizer=SGD(learning_rate=0.01),
                   loss='mse',
                   metrics=['accuracy'] )
    model.summary()
    return model

def test_navadna_mreza():
    time = None
    channels = 54
    n_filters = 8
    #NASPROTNIKI
    input_layer_nasprotiki = Input( shape=(time, 3,channels),name='Nasprotniki' )
    conv1 = TimeDistributed(Conv1D(n_filters,1,activation='elu'))(input_layer_nasprotiki)
    conv1 = TimeDistributed(BatchNormalization())(input_layer_nasprotiki)
    #conv1 = ConvLSTM2D(filters=n_filters,kernel_size=(1,1),return_sequences=True,activation='elu')( input_layer_nasprotiki )
    conv1 = TimeDistributed(Flatten())( conv1 )
    #conv1 = TimeDistributed( Dense(8,activation='elu') )( conv1 )
    #conv1 = TimeDistributed(BatchNormalization())(conv1)
    lstm = CuDNNLSTM( 8, return_sequences=True )( conv1 )
    #MOJA ROKA
    roka_input = Input( shape=(time, 54),name='Roka' )
    #roka = Dense(16,activation='elu')(roka_input)
    #roka = BatchNormalization()(roka)
    roka = CuDNNLSTM(8,return_sequences=True)(roka_input)
    #roka = CuDNNGRU(20,return_sequences=True)(roka)

    #zdruzi
    #print(lstm.shape,roka.shape)
    lstm = Concatenate(axis=2,name='Konkat')([lstm,roka])
    lstm = CuDNNLSTM(16)(lstm)
    #lstm = CuDNNGRU(30)(lstm)

    #talon
    talon_input = Input( shape=( 6,55),name='Talon' ) # 55 je tist k je vzeto iz talona
    talon = Flatten()(talon_input) # 55 je tist k je vzeto iz talona
    #talon = Dense( 16,activation='elu' )(talon)
    #talon = BatchNormalization()(talon)

    #kraj
    kralj = Input((4,),name='Kralj')
    index_tistega_ki_igra = Input((4,),name='Tisti_ki_igra')
    concat = Concatenate()([lstm,kralj,talon,index_tistega_ki_igra])

    output_layer = Dense( 16, activation='elu' )(concat )

    output_layer = BatchNormalization()(output_layer)
    #output_layer = Dense( 54, activation='elu' )(output_layer )
    output_layer = Dense( 54, activation=None )(output_layer )

    #mozne = Input( (54,), name='Mozne' )
    #output_layer = Multiply()( [output_layer, mozne] )
    model = Model( inputs=[input_layer_nasprotiki,kralj,roka_input,talon_input,index_tistega_ki_igra], outputs=output_layer,name='Navadna')
    model.compile( optimizer=SGD(learning_rate=0.01),
                   loss='mse',
                   metrics=['accuracy'] )
    model.summary()
    #from keras.utils import plot_model
    #plot_model( model, to_file='model.png' )
    #create data

    #batch = 80
    #time = 12
    #X = [np.random.random((batch,time,4,54)), np.random.random((batch,4)), np.random.random((batch,time,54)), np.random.random((batch,6,55)), np.random.random((batch,4)), np.random.random((batch,54))]
    #model.fit(X,np.random.random((batch,54)),batch_size=8,epochs=2)
    return model

def test_klop():
    time = None
    channels = 54
    n_filters = 10
    #NASPROTNIKI
    input_layer_nasprotiki = Input( shape=(time, 3,channels),name='Nasprotniki' )
    conv1 = TimeDistributed(Conv1D(n_filters,1,activation='elu'))(input_layer_nasprotiki)
    conv1 = TimeDistributed(Flatten())( conv1 )
    #conv1 = Dense(16,activation='elu')( conv1 )
    lstm = CuDNNLSTM( 16, return_sequences=True )( conv1 )
    #MOJA ROKA
    roka_input = Input( shape=(time, 54), name='Roka' )
    #roka = Dense( 16, activation='elu' )(roka_input)

    #roka = BatchNormalization()( roka )
    roka = CuDNNLSTM( 16, return_sequences=True )( roka_input )

    #talon
    talon_input = Input( shape=( 54,),name='Talon' ) # 54 je tist k je vzeto iz talona
    #talon = TimeDistributed(Flatten())(talon_input)
    #talon = Dense( 20,activation='elu',return_sequences=True)(talon)
    #talon = Dense( 20,activation='elu')(talon_input)


    #zdruzi
    #print(lstm.shape,roka.shape)
    lstm = Concatenate(axis=2,name='Konkat')([lstm,roka])
    lstm = CuDNNLSTM(16,return_sequences=False)(lstm)
    conncat = Concatenate(name='Konkat_z_talonom')([lstm,talon_input])

    output_layer = Dense( 16, activation='elu' )( conncat )
    #output_layer = Dense( 54, activation='elu' )(output_layer )
    output_layer = BatchNormalization()(output_layer )
    output_layer = Dense( 54, activation=None )(output_layer )

    #mozne = Input( (54,), name='Mozne' )
    #output_layer = Multiply()( [output_layer, mozne] )
    model = Model( inputs=[input_layer_nasprotiki,roka_input,talon_input], outputs=output_layer,name='Klop' )
    model.compile( optimizer=SGD(lr=0.01),
                   loss='mse',
                   metrics=['accuracy'] )
    model.summary()
    #from keras.utils import plot_model
    #plot_model( model, to_file='model.png' )
    #create data

    #batch = 80
    #time = 12
    #X = [np.random.random((batch,time,3,54)),  np.random.random((batch,time,54)), np.random.random((batch,time,54)), np.random.random((batch,54))]
    #model.fit(X,np.random.random((batch,54)),batch_size=8,epochs=2)
    return model

def test_solo():
    time = None
    channels = 54
    n_filters = 10
    #NASPROTNIKI
    input_layer_nasprotiki = Input( shape=(time, 3,channels),name='Nasprotniki' )
    conv1 = TimeDistributed(Conv1D(n_filters,1,activation='elu'))(input_layer_nasprotiki)
    #conv1 = ConvLSTM2D(filters=n_filters,kernel_size=(1,1),return_sequences=True,activation='elu')( input_layer_nasprotiki )
    conv1 = TimeDistributed(Flatten())( conv1 )
    conv1 = Dense(30,activation='elu')( conv1 )
    lstm = CuDNNLSTM( 32, return_sequences=True )( conv1 )
    #MOJA ROKA
    roka_input = Input( shape=(time, 54), name='Roka' )
    roka = Dense( 30, activation='elu' )(roka_input)
    roka = CuDNNLSTM( 20, return_sequences=True )( roka )

    #zdruzi
    lstm = Concatenate(axis=2,name='Konkat')([lstm,roka])
    lstm = CuDNNLSTM(30)(lstm)

    #talon
    talon_input = Input( shape=( 6,55),name='Talon' ) # 55 je tist k je vzeto iz talona
    talon = Flatten()(talon_input) # 55 je tist k je vzeto iz talona
    talon = Dense( 20,activation='elu' )(talon)


    index_tistega_ki_igra = Input((1,),name='Tisti_ki_igra') # ali igram jaz ali ne 1 ali -1
    concat = Concatenate()([lstm,talon,index_tistega_ki_igra])

    output_layer = Dense( 54, activation='elu' )(concat )
    output_layer = Dense( 54, activation='elu' )(output_layer )
    output_layer = Dense( 54, activation=None )(output_layer )

    mozne = Input( (54,), name='Mozne' )
    output_layer = Multiply()( [output_layer, mozne] )
    model = Model( inputs=[input_layer_nasprotiki,roka_input,talon_input,index_tistega_ki_igra,mozne], outputs=output_layer )
    model.compile( optimizer=Adam(learning_rate=0.1),
                   loss='mse',
                   metrics=['accuracy'] )
    #model.summary()
    #from keras.utils import plot_model
    #plot_model( model, to_file='model.png' )
    #create data

    batch = 80
    time = 12
    X = [np.random.random((batch,time,4,54)), np.random.random((batch,time,54)), np.random.random((batch,6,55)), np.random.random((batch,1)), np.random.random((batch,54))]
    #model.fit(X,np.random.random((batch,54)),batch_size=8,epochs=2)
    return model

def test_berac():
    time = None
    channels = 54
    n_filters = 10
    #NASPROTNIKI
    input_layer_nasprotiki = Input( shape=(time, 3,channels),name='Nasprotniki' )
    conv1 = TimeDistributed(Conv1D(n_filters,1,activation='elu'))(input_layer_nasprotiki)
    #conv1 = ConvLSTM2D(filters=n_filters,kernel_size=(1,1),return_sequences=True,activation='elu')( input_layer_nasprotiki )
    conv1 = TimeDistributed(Flatten())( conv1 )
    conv1 = Dense(30,activation='elu')( conv1 )
    lstm = CuDNNLSTM( 32, return_sequences=True )( conv1 )
    #MOJA ROKA
    roka_input = Input( shape=(time, 54), name='Roka' )
    roka = Dense( 30, activation='elu' )(roka_input)
    roka = CuDNNLSTM( 20, return_sequences=True )( roka )

    #zdruzi
    lstm = Concatenate(axis=2,name='Konkat')([lstm,roka])
    lstm = CuDNNLSTM(30)(lstm)

    index_tistega_ki_igra = Input((1,),name='Tisti_ki_igra') # ali igram jaz ali ne 1 ali -1

    concat = Concatenate()([lstm,index_tistega_ki_igra])

    output_layer = Dense( 54, activation='elu' )(concat )
    output_layer = Dense( 54, activation='elu' )(output_layer )
    output_layer = Dense( 54, activation=None )(output_layer )

    mozne = Input( (54,), name='Mozne' )
    output_layer = Multiply()( [output_layer, mozne] )
    model = Model( inputs=[input_layer_nasprotiki,roka_input,index_tistega_ki_igra,mozne], outputs=output_layer )
    model.compile( optimizer='adam',
                   loss='mse',
                   metrics=['accuracy'] )
    #model.summary()
    #from keras.utils import plot_model
    #plot_model( model, to_file='model.png' )
    #create data

    batch = 80
    time = 12
    X = [np.random.random((batch,time,4,54)), np.random.random((batch,time,54)), np.random.random((batch,1)), np.random.random((batch,54))]
    #model.fit(X,np.random.random((batch,54)),batch_size=8,epochs=2)
    return model