def lstm():
    if 'lstm' in params.structure:
        # Packet sequence inputs to multi-input model
        pktseq_inputs = { name: layers.Input(shape=(params.sequence_length,), dtype=tf.float32, name=name)for name in params.packet_feature_name}

        # normalize input of first two packet features
        pktseq_x1 = tf.stack(list(pktseq_inputs.values())[:2], axis=2)
        # pktseq_x1 = preprocessor_pkt(pktseq_x1)
        # pktseq_x2 = layers.Reshape(target_shape=(params.sequence_length, 1))(list(pktseq_inputs.values())[-1])
        # pktseq_x = layers.Concatenate(axis=-1)([pktseq_x1, pktseq_x2])

        pktseq_x = layers.LSTM(64, input_shape=(30, 3))(pktseq_x1)
        pktseq_x = layers.Dropout(0.1)(pktseq_x)
        pktseq_x = layers.Dense(units=192, kernel_initializer=initializer)(pktseq_x)
        pktseq_x = layers.BatchNormalization()(pktseq_x)
        pktseq_x = layers.LeakyReLU()(pktseq_x)

        input_branches['inputs'].append(pktseq_inputs)
        input_branches['layer'].append(pktseq_x)
        
    if 'mlp' in params.structure:
            # Statistical inputs to multi-input model
            flow_inputs = {name: layers.Input(shape=(1,), dtype=tf.float32, name=name) for name in params.features}
            flow_x = layers.Concatenate(axis=-1)(list(flow_inputs.values()))
            # flow_x = layers.Reshape(target_shape=(len(params.features),))(flow_x)

            # flow_x = preprocessor_flow(flow_x)

            # Adjust size of previous set of dense layers
            flow_x = layers.Dense(units=32)(flow_x)
            flow_x = layers.LeakyReLU()(flow_x)
            flow_x = layers.Dense(units=32)(flow_x)
            flow_x = layers.LeakyReLU()(flow_x)
            flow_x = layers.Dense(units=32)(flow_x)
            flow_x = layers.LeakyReLU()(flow_x)
            flow_x = layers.Dropout(0.2)(flow_x)
            flow_x = layers.Dense(units=8, kernel_regularizer='l2')(flow_x)
            flow_x = layers.BatchNormalization()(flow_x)
            flow_x = layers.LeakyReLU()(flow_x)

            input_branches['inputs'].append(flow_inputs)
            input_branches['layer'].append(flow_x)
