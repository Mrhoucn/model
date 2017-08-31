
  x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])




def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


def BiRNN(x , regular):
  with tf.variable_scope('layer1'):
    
    x = tf.transpose(x ,[1,0,2])
    
    x = tf.reshape(x ,[-1,n_inputs])
   
    x =tf.split(x,n_steps)
    
    weights = get_weight_variable([2*n_hidden,n_hidden], regular)
    biases = tf.get_variable("biases", [n_hidden], initializer=tf.constant_initializer(0.0))
    

    output= mulBiRNN(rnn.LSTMCell,x,2,500,128)

  with tf.name_scope('Wx_plus_b'): 
      preactivate = tf.matmul(output,weights) + biases
      tf.summary.histogram("preactivate",preactivate)
    
  return preactivate

def mulBiRNN(RNN,c,num_layers,num_units,batch_size):
    _inputs = c
    for _ in range(num_layers):
        #为什么在这加个variable_scope,被逼的,tf在rnn_cell的__call__中非要搞一个命名空间检查
        #恶心的很.如果不在这加的话,会报错的.
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            rnn_cell_fw = RNN(num_units)
            rnn_cell_bw = RNN(num_units)
            initial_state_fw = rnn_cell_fw.zero_state(batch_size,dtype = tf.float32)
            initial_state_bw = rnn_cell_bw.zero_state(batch_size,dtype = tf.float32)
            output,_,_ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs,initial_state_fw, 
                                                            initial_state_bw ,dtype=tf.float32)
            _inputs = tf.concat(output, 2)
    return _inputs


