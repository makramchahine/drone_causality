import tensorflow as tf
import numpy  as np

class VanillaRNN(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units):
        self._num_units = num_units

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    # TODO: Implement RNNLayer properly,i.e, allocate variables here
    def build(self,input_shape):
        pass

    def _dense(self,units,inputs,activation,name,bias_initializer=tf.constant_initializer(0.0)):
        input_size = int(inputs.shape[-1])
        W = tf.get_variable('W_{}'.format(name), [input_size, units])
        b = tf.get_variable("bias_{}".format(name), [units],initializer=bias_initializer)

        y = tf.matmul(inputs,W) + b
        if(not activation is None):
            y = activation(y)

        return y

    def __call__(self, inputs, state, scope=None):
        self._input_size = int(inputs.shape[-1])
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("RNN",reuse=tf.AUTO_REUSE):  # Reset gate and update gate.
                fused_input = tf.concat([inputs,state],axis=-1)

                # RNN
                h_next = self._dense(units=self._num_units,inputs=fused_input,activation=tf.nn.tanh,name="step")

        return h_next, h_next

class CTRNN(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units,cell_clip=-1,global_feedback=False):
        self._num_units = num_units
        # Number of ODE solver steps
        self._unfolds = 6
        # Time of each ODE solver step, for variable time RNN change this
        # to a placeholder/non-trainable variable
        self._delta_t = 0.1

        self.global_feedback = global_feedback

        # Time-constant of the cell
        self.tau = 1
        self.cell_clip = cell_clip


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    # TODO: Implement RNNLayer properly,i.e, allocate variables here
    def build(self,input_shape):
        pass

    def _dense(self,units,inputs,activation,name,bias_initializer=tf.constant_initializer(0.0)):
        input_size = int(inputs.shape[-1])
        W = tf.get_variable('W_{}'.format(name), [input_size, units])
        b = tf.get_variable("bias_{}".format(name), [units],initializer=bias_initializer)

        y = tf.matmul(inputs,W) + b
        if(not activation is None):
            y = activation(y)

        self.W = W
        self.b = b
        
        return y

    def __call__(self, inputs, state, scope=None):
        # CTRNN ODE is: df/dt = NN(x) - f
        # where x is the input, and NN is a MLP.
        # Input could be: 1: just the input of the RNN cell
        # or 2: input of the RNN cell merged with the current state

        self._input_size = int(inputs.shape[-1])
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("RNN",reuse=tf.AUTO_REUSE):  # Reset gate and update gate.

                # Input Option 1: RNNCell input
                if(not self.global_feedback):
                    input_f_prime = self._dense(units=self._num_units,inputs=inputs,activation=tf.nn.tanh,name="step")
                for i in range(self._unfolds):
                    # Input Option 2: RNNCell input AND RNN state
                    if(self.global_feedback):
                        fused_input = tf.concat([inputs,state],axis=-1)
                        input_f_prime = self._dense(units=self._num_units,inputs=fused_input,activation=tf.nn.tanh,name="step")

                    # df/dt 
                    f_prime = -state/self.tau + input_f_prime

                    # If we solve this ODE with explicit euler we get
                    # f(t+deltaT) = f(t) + deltaT * df/dt
                    state = state + self._delta_t * f_prime

                    # Optional clipping of the RNN cell to enforce stability (not needed)
                    if(self.cell_clip > 0):
                        state = tf.clip_by_value(state,-self.cell_clip,self.cell_clip)

        return state,state

class GatedRecurrentUnit(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units,cell_clip=-1):
        self._num_units = num_units
        self.cell_clip = cell_clip

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    # TODO: Implement RNNLayer properly,i.e, allocate variables here
    def build(self,input_shape):
        pass

    def _dense(self,units,inputs,activation,name,bias_initializer=tf.constant_initializer(0.0)):
        input_size = int(inputs.shape[-1])
        W = tf.get_variable('W_{}'.format(name), [input_size, units])
        b = tf.get_variable("bias_{}".format(name), [units],initializer=bias_initializer)

        y = tf.matmul(inputs,W) + b
        if(not activation is None):
            y = activation(y)

        return y

    def __call__(self, inputs, state, scope=None):
        self._input_size = int(inputs.shape[-1])
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Gates",reuse=tf.AUTO_REUSE):  # Reset gate and update gate.
                fused_input = tf.concat([inputs,state],axis=-1)

                # Reset gate
                rk = self._dense(units=self._num_units,inputs=fused_input,activation=tf.nn.sigmoid,name="reset_gate",bias_initializer=tf.constant_initializer(1))

                reset_value = tf.concat([inputs,rk*state],axis=-1)
                qk = self._dense(units=self._num_units,inputs=reset_value,activation=tf.nn.tanh,name="detect_signal")

                # Update gate setting
                sk =  self._dense(units=self._num_units,inputs=fused_input,activation=tf.nn.sigmoid,name="update_setting")

                # Compute new state
                new_h = (1-sk)*state + sk * qk
                if(self.cell_clip > 0):
                    new_h = tf.clip_by_value(new_h,-self.cell_clip,self.cell_clip)

        return new_h, new_h

class ContinuousTimeGatedRecurrentUnit(tf.nn.rnn_cell.RNNCell):
    # https://arxiv.org/abs/1710.04110
    def __init__(self, num_units,M=8,cell_clip=-1):
        self._num_units = num_units
        self.M = M
        self.cell_clip = cell_clip
        self.ln_tau_table = np.empty(self.M)
        tau = 1
        for i in range(self.M):
            self.ln_tau_table[i] = np.log(tau)
            tau = tau * (10.0**0.5)

    @property
    def state_size(self):
        return self._num_units*self.M

    @property
    def output_size(self):
        return self._num_units

    # TODO: Implement RNNLayer properly,i.e, allocate variables here
    def build(self,input_shape):
        pass

    def _dense(self,units,inputs,activation,name,bias_initializer=tf.constant_initializer(0.0)):
        input_size = int(inputs.shape[-1])
        W = tf.get_variable('W_{}'.format(name), [input_size, units])
        b = tf.get_variable("bias_{}".format(name), [units],initializer=bias_initializer)

        y = tf.matmul(inputs,W) + b
        if(not activation is None):
            y = activation(y)

        return y

    def __call__(self, inputs, state, scope=None):
        self._input_size = int(inputs.shape[1])

        # CT-GRU input is actually a matrix and not a vector
        h_hat = tf.reshape(state,[-1,self._num_units,self.M])
        h = tf.reduce_sum(h_hat,axis=2)
        state = None # Set state to None, to avoid misuses (bugs) in the code below

        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                fused_input = tf.concat([inputs,h],axis=-1)
                ln_tau_r = tf.layers.Dense(self._num_units*self.M,activation=None,name="tau_r")(fused_input)
                ln_tau_r = tf.reshape(ln_tau_r,shape=[-1,self._num_units,self.M])
                sf_input_r = -tf.square(ln_tau_r-self.ln_tau_table)
                rki = tf.nn.softmax(logits=sf_input_r,axis=2)

                q_input = tf.reduce_sum(rki*h_hat,axis=2)
                reset_value = tf.concat([inputs,q_input],axis=1)
                qk = self._dense(units=self._num_units,inputs=reset_value,activation=tf.nn.tanh,name="detect_signal")

                qk = tf.reshape(qk,[-1,self._num_units,1]) # in order to broadcast

                ln_tau_s = tf.layers.Dense(self._num_units*self.M,activation=None,name="tau_s")(fused_input)
                ln_tau_s = tf.reshape(ln_tau_s,shape=[-1,self._num_units,self.M])
                sf_input_s = -tf.square(ln_tau_s-self.ln_tau_table)
                ski = tf.nn.softmax(logits=sf_input_s,axis=2)

                h_hat_next = ((1-ski)*h_hat + ski*qk)*np.exp(-1.0/self.ln_tau_table)

                if(self.cell_clip > 0):
                    h_hat_next = tf.clip_by_value(h_hat_next,-self.cell_clip,self.cell_clip)
                # Compute new state
                h_next = tf.reduce_sum(h_hat_next,axis=2)
                h_hat_next_flat = tf.reshape(h_hat_next,shape=[-1,self._num_units*self.M])

        return h_next, h_hat_next_flat