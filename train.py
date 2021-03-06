import numpy as np
import tensorflow as tf

def model_za_vrednotenje_roke(lr,l2_rate=0.001):
    model = tf.keras.Sequential(name='Vrednotenje_roke')
    model.add( tf.keras.layers.Dense( 32, activation='elu', input_shape=(54,) ,kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate)) )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    #model.add( Dense( 30, activation='relu'))
    model.add( tf.keras.layers.Dense( 32, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate) ) )
    model.add( tf.keras.layers.BatchNormalization() )
    model.add( tf.keras.layers.Dropout( 0.2 ) )
    #model.add( Dense( 10, activation='relu' ) )
    model.add( tf.keras.layers.Dense(
        1+ #Naprej
        4*3+ # tri dva ena v vseh sterih kraljih
        3+ # 3 solo
        1+# zaprti
        #1+  odprti berac
        1#solo brez
        , activation='linear' ) )

    model.compile( optimizer=tf.keras.optimizers.Adam(lr=lr),
                   loss=tf.keras.losses.Huber(delta=25.0,),
                   metrics=['accuracy'] )
    model.summary()
    return model

def model_za_zalaganje(lr,l2_rate=0.001):
    talon_input = tf.keras.layers.Input( shape=(54,6), name='Talon') # prva dim karta v drugo smer v kerem izmed 6 kupckov je
    talon = tf.keras.layers.Flatten()(talon_input)
    roka_input = tf.keras.layers.Input( shape=(54), name='roka' )
    igra_input = tf.keras.layers.Input( shape=(15),name='igra' ) # 3*4 +3
    output_layer = tf.keras.layers.Concatenate()([talon,roka_input,igra_input])
    output_layer = tf.keras.layers.Dense(32,'elu',kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate))(output_layer)
    output_layer = tf.keras.layers.BatchNormalization()(output_layer)
    output_layer = tf.keras.layers.Dropout(0.2)(output_layer)
    output_layer = tf.keras.layers.Dense(32,'elu',kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate))(output_layer)
    output_layer = tf.keras.layers.BatchNormalization()(output_layer)
    output_layer = tf.keras.layers.Dropout(0.2)(output_layer)
    output_layer = tf.keras.layers.Dense(60)(output_layer) #kerga izmed 6 kupckov izberes + 54 kart za zalozit
    model = tf.keras.Model( inputs=[roka_input, talon_input, igra_input],
                            outputs=output_layer, name='Zalaganje' )

    model.compile( optimizer=tf.keras.optimizers.Adam(lr=lr),
                   loss=tf.keras.losses.Huber(delta=25.0,),
                   metrics=['accuracy'] )
    model.summary()
    return model

def test_navadna_mreza(lr,l2_rate=0.001):
    time = None
    channels = 54
    n_filters = 8
    #NASPROTNIKI
    input_layer_nasprotiki = tf.keras.layers.Input( shape=(time, 3,channels),name='Nasprotniki' )
    #conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(n_filters,1,activation='elu'))(input_layer_nasprotiki)
    conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(input_layer_nasprotiki)
    conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32,kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate)))(conv1)
    lstm = tf.keras.layers.CuDNNLSTM( 32, return_sequences=True ,kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate))( conv1 )
    #lstm = tf.keras.layers.Activation( 'elu'  )( lstm )

    #MOJA ROKA
    roka_input = tf.keras.layers.Input( shape=(time, 54),name='Roka' )
    #roka = Dense(16,activation='elu')(roka_input)
    #roka = BatchNormalization()(roka)
    roka = tf.keras.layers.CuDNNLSTM(32,return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate))(roka_input)
    #roka = CuDNNGRU(20,return_sequences=True)(roka)
    #roka = tf.keras.layers.Activation( 'elu' ) ( roka )

    #zdruzi
    #print(lstm.shape,roka.shape)
    lstm = tf.keras.layers.Concatenate(axis=2,name='Konkat')([lstm,roka])
    lstm = tf.keras.layers.CuDNNLSTM(32,kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate))(lstm)
    #lstm = CuDNNGRU(30)(lstm)

    #talon
    talon_input = tf.keras.layers.Input( shape=( 6,55),name='Talon' ) # 55 je tist k je vzeto iz talona
    talon = tf.keras.layers.Flatten()(talon_input) # 55 je tist k je vzeto iz talona
    #talon = Dense( 16,activation='elu' )(talon)
    #talon = BatchNormalization()(talon)

    #kraj
    kralj = tf.keras.layers.Input((4,),name='Kralj')
    index_tistega_ki_igra = tf.keras.layers.Input((4,),name='Tisti_ki_igra')
    zalozil_input = tf.keras.layers.Input((54,),name='Zalozil_input')
    concat = tf.keras.layers.Concatenate()([lstm,kralj,talon,index_tistega_ki_igra,zalozil_input])

    output_layer = tf.keras.layers.Dense( 32, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate) )(concat )

    output_layer = tf.keras.layers.BatchNormalization()(output_layer)
    output_layer = tf.keras.layers.Dropout(0.2)(output_layer)
    #output_layer = Dense( 54, activation='elu' )(output_layer )
    output_layer = tf.keras.layers.Dense( 32, activation='elu', kernel_regularizer=tf.keras.regularizers.l2( l2_rate ),
                                          bias_regularizer=tf.keras.regularizers.l2( l2_rate ) )( output_layer )

    output_layer = tf.keras.layers.BatchNormalization()( output_layer )
    output_layer = tf.keras.layers.Dropout( 0.2 )( output_layer )

    output_layer = tf.keras.layers.Dense( 54, activation=None )(output_layer )

    #mozne = Input( (54,), name='Mozne' )
    #output_layer = Multiply()( [output_layer, mozne] )
    model = tf.keras.Model( inputs=[input_layer_nasprotiki,kralj,roka_input,talon_input,index_tistega_ki_igra,zalozil_input], outputs=output_layer,name='Navadna')
    model.compile( optimizer=tf.keras.optimizers.Adam(lr=lr),
                   loss=tf.keras.losses.Huber(delta=25.0),
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

def test_klop(lr,l2_rate=0.0001):
    time = None
    channels = 54
    n_filters = 10
    #NASPROTNIKI
    input_layer_nasprotiki = tf.keras.layers.Input( shape=(time, 3,channels),name='Nasprotniki' )
    conv1 = tf.keras.layers.TimeDistributed( tf.keras.layers.Flatten() )( input_layer_nasprotiki )
    conv1 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense( 32, kernel_regularizer=tf.keras.regularizers.l2( l2_rate ),
                               bias_regularizer=tf.keras.regularizers.l2( l2_rate ) ) )( conv1 )
    lstm = tf.keras.layers.CuDNNLSTM( 16, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate) )( conv1 )
    lstm = tf.keras.layers.Activation( 'elu' ) ( lstm )

    #MOJA ROKA
    roka_input = tf.keras.Input( shape=(time, 54), name='Roka' )
    #roka = Dense( 16, activation='elu' )(roka_input)

    #roka = BatchNormalization()( roka )
    roka = tf.keras.layers.CuDNNLSTM( 16, return_sequences=True ,kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate))( roka_input )
    roka = tf.keras.layers.Activation( 'elu'  )( roka )

    #talon
    talon_input = tf.keras.layers.Input( shape=( 54,),name='Talon' ) # 54 je tist k je vzeto iz talona
    #talon = TimeDistributed(Flatten())(talon_input)
    #talon = Dense( 20,activation='elu',return_sequences=True)(talon)
    #talon = Dense( 20,activation='elu')(talon_input)

    #zdruzi
    #print(lstm.shape,roka.shape)
    lstm = tf.keras.layers.Concatenate(axis=2,name='Konkat')([lstm,roka])
    lstm = tf.keras.layers.CuDNNLSTM(16,return_sequences=False,kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate))(lstm)
    conncat = tf.keras.layers.Concatenate(name='Konkat_z_talonom')([lstm,talon_input])

    output_layer = tf.keras.layers.Dense( 32, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate) )( conncat )
    output_layer = tf.keras.layers.BatchNormalization()(output_layer )
    output_layer = tf.keras.layers.Dropout(0.2)(output_layer )
    output_layer = tf.keras.layers.Dense( 32, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate) )(output_layer )
    output_layer = tf.keras.layers.BatchNormalization()(output_layer )
    output_layer = tf.keras.layers.Dense( 54, activation=None )(output_layer )

    #mozne = Input( (54,), name='Mozne' )
    #output_layer = Multiply()( [output_layer, mozne] )
    model = tf.keras.Model( inputs=[input_layer_nasprotiki,roka_input,talon_input], outputs=output_layer,name='Klop' )
    model.compile( optimizer=tf.keras.optimizers.Adam(lr=lr),
                   loss=tf.keras.losses.Huber(delta=25.0),
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

def test_solo(lr,l2_rate=0.0001):
    time = None
    channels = 54
    n_filters = 10
    #NASPROTNIKI
    input_layer_nasprotiki = tf.keras.layers.Input( shape=(time, 3,channels),name='Nasprotniki' )
    conv1 = tf.keras.layers.TimeDistributed( tf.keras.layers.Flatten() )( input_layer_nasprotiki )
    conv1 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense( 32, kernel_regularizer=tf.keras.regularizers.l2( l2_rate ),
                               bias_regularizer=tf.keras.regularizers.l2( l2_rate ) ) )( conv1 )
    conv1 = tf.keras.layers.TimeDistributed( tf.keras.layers.Flatten() )( conv1 )

    #MOJA ROKA
    roka_input = tf.keras.layers.Input( shape=(time, 54), name='Roka' )
    roka = tf.keras.layers.Dense( 30, activation='elu' )(roka_input)
    roka = tf.keras.layers.CuDNNLSTM( 20, return_sequences=True )( roka )

    #zdruzi
    lstm = tf.keras.layers.CuDNNLSTM( 32, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2( l2_rate ),
                                      bias_regularizer=tf.keras.regularizers.l2( l2_rate ) )( conv1 )
    lstm = tf.keras.layers.Activation('elu')(lstm)
    lstm = tf.keras.layers.Concatenate(axis=2,name='Konkat')([lstm,roka])
    lstm = tf.keras.layers.CuDNNLSTM(32, kernel_regularizer=tf.keras.regularizers.l2( l2_rate ),
                                      bias_regularizer=tf.keras.regularizers.l2( l2_rate ))(lstm)

    #talon
    talon_input = tf.keras.layers.Input( shape=( 6,55),name='Talon' ) # 55 je tist k je vzeto iz talona
    talon = tf.keras.layers.Flatten()(talon_input) # 55 je tist k je vzeto iz talona

    index_tistega_ki_igra = tf.keras.layers.Input((4,),name='Tisti_ki_igra')
    zalozil_input = tf.keras.layers.Input( (54,), name='Zalozil_input' )

    concat = tf.keras.layers.Concatenate()([lstm,talon,index_tistega_ki_igra,zalozil_input])

    output_layer = tf.keras.layers.Dense( 32, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate) )( concat )
    output_layer = tf.keras.layers.BatchNormalization()(output_layer )
    output_layer = tf.keras.layers.Dropout(0.2)(output_layer )
    output_layer = tf.keras.layers.Dense( 32, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate) )(output_layer )
    output_layer = tf.keras.layers.BatchNormalization()(output_layer )
    output_layer = tf.keras.layers.Dense( 54, activation=None )(output_layer )


    model = tf.keras.Model( inputs=[input_layer_nasprotiki,roka_input,talon_input,index_tistega_ki_igra,zalozil_input], outputs=output_layer,name='Solo' )
    model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                   loss=tf.keras.losses.Huber(delta=25.0),
                   metrics=['accuracy'] )
    model.summary()
    return model

def test_berac(lr,l2_rate=0.0001):
    time = None
    channels = 54
    n_filters = 10
    #NASPROTNIKI
    input_layer_nasprotiki = tf.keras.layers.Input( shape=(time, 3,channels),name='Nasprotniki' )
    conv1 = tf.keras.layers.TimeDistributed( tf.keras.layers.Flatten() )( input_layer_nasprotiki )
    conv1 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense( 32, kernel_regularizer=tf.keras.regularizers.l2( l2_rate ),
                               bias_regularizer=tf.keras.regularizers.l2( l2_rate ) ) )( conv1 )
    conv1 = tf.keras.layers.TimeDistributed( tf.keras.layers.Flatten() )( conv1 )
    lstm = tf.keras.layers.CuDNNLSTM( 32, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate) )( conv1 )
    lstm = tf.keras.layers.Activation( 'elu' )( lstm )
    #MOJA ROKA
    roka_input = tf.keras.layers.Input( shape=(time, 54), name='Roka' )
    roka = tf.keras.layers.Dense( 32, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate) )(roka_input)
    roka = tf.keras.layers.CuDNNLSTM( 32, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate) )( roka )
    roka = tf.keras.layers.Activation( 'elu' )( roka )

    #zdruzi
    lstm = tf.keras.layers.Concatenate(axis=2,name='Konkat')([lstm,roka])
    lstm = tf.keras.layers.CuDNNLSTM(32,kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate))(lstm)
    lstm = tf.keras.layers.Activation('elu')(lstm)

    index_tistega_ki_igra = tf.keras.layers.Input((4,),name='Tisti_ki_igra') # ali igram jaz ali ne 1 ali -1

    concat = tf.keras.layers.Concatenate()([lstm,index_tistega_ki_igra])


    output_layer = tf.keras.layers.Dense( 32, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate) )( concat )
    output_layer = tf.keras.layers.BatchNormalization()(output_layer )
    output_layer = tf.keras.layers.Dropout(0.2)(output_layer )
    output_layer = tf.keras.layers.Dense( 32, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(l2_rate),bias_regularizer=tf.keras.regularizers.l2(l2_rate) )(output_layer )
    output_layer = tf.keras.layers.BatchNormalization()(output_layer )
    output_layer = tf.keras.layers.Dense( 54, activation=None )(output_layer )

    model = tf.keras.Model( inputs=[input_layer_nasprotiki,roka_input,index_tistega_ki_igra], outputs=output_layer, name='Berac' )
    model.compile( optimizer=tf.keras.optimizers.Adam( learning_rate=lr ),
                   loss=tf.keras.losses.Huber( delta=25.0 ),
                   metrics=['accuracy'] )

    #batch = 80
    #time = 12
    #X = [np.random.random((batch,time,4,54)), np.random.random((batch,time,54)), np.random.random((batch,1)), np.random.random((batch,54))]
    #model.fit(X,np.random.random((batch,54)),batch_size=8,epochs=2)
    model.summary()
    return model
