def create_dl_model_cnn(params):

    if params['regularizer'] == 'l1':
            regularizer = tf.keras.regularizers.L1(params['regularizer_value'])
    elif params['regularizer'] == 'l2':
            regularizer = tf.keras.regularizers.L2(params['regularizer_value'])
    else:
            regularizer = None

    """Create input layers for packet sequence data """
    # if params['cnn_feature_type'] == 'sequence':
    #     inputs = {name: layers.Input(shape=(params['sequence_length'],), dtype=tf.float32, name=name) for name in params['seq_packet_feature']}
    # elif params['cnn_feature_type'] == 'statistical':
    #     inputs = {name: layers.Input(shape=(params['cnn_stat_feature_length'],), dtype=tf.float32, name=name) for name in params['cnn_stat_feature']}
    # elif params['cnn_feature_type'] == 'packet_bytes':
    #     inputs = {name: layers.Input(shape=(params['cnn_byte_feature_length'],), dtype=tf.float32, name=name) for name in params['cnn_byte_feature']}
    inputs = {name: layers.Input(shape=(params['cnn_stat_feature_length'],), dtype=tf.float32, name=name) for name in params['cnn_stat_feature']}
    # inputs = {name: layers.Input(shape=(150,), dtype=tf.float32, name=name) for name in params['seq_packet_feature']}
    """Stack input layers"""
    pktseq_x = tf.stack(list(inputs.values()), axis=2)
    # pktseq_x = layers.Reshape(target_shape=(params['sequence_length'], 1))(list(inputs.values())[-1])

    pktseq_x = layers.Conv1D(200, kernel_size=7, strides=1, kernel_regularizer=regularizer,  padding='same', input_shape=(None, 3))(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)

    pktseq_x = layers.Conv1D(200, kernel_size=5, strides=1, kernel_regularizer=regularizer, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)

    pktseq_x = layers.Conv1D(200, kernel_size=5, strides=1, kernel_regularizer=regularizer, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(200, kernel_size=4, strides=1, kernel_regularizer=regularizer, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(300, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)

    pktseq_x = layers.Conv1D(300, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)

    pktseq_x = layers.Conv1D(300, kernel_size=2, strides=2, kernel_regularizer=regularizer, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)

    pktseq_x = layers.GlobalAveragePooling1D()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(0.5)(pktseq_x)

    """Output layer"""
    # outputs = layers.Dense(output_units, activation='softmax', name='softmax')(pktseq_x)
    # model = models.Model(inputs=[inputs], outputs=outputs)

    return inputs, pktseq_x
