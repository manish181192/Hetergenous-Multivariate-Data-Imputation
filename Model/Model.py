from config import Config as config
import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn

class Model(object):

    def __init__(self, data_manager, genmode=False):

        self.data_manager = data_manager
        self.init_placeholders()
        if config.nlp:
            self.initialize_lookup_tables()
        self.hivae_model()
        # self.generate_samples()

    def init_placeholders(self):
        self.feature_placeholders = []
        self.nlp_texts_ids = []
        self.nlp_feature_ids = []
        for feature_id, feature in enumerate(self.data_manager.fields):
            if feature["type"]!="nlp":
                self.feature_placeholders.append(tf.placeholder(name=feature['name'],
                                                                 shape= [ None, feature['ndims']],
                                                                 dtype= tf.float32)
                                                 )
            else:
                if config.nlp ==True:
                    self.nlp_feature_ids.append(feature_id)
                    self.nlp_texts_ids.append(tf.placeholder(name=feature["name"],
                                                             shape=[None, feature["max_seq_length"]],
                                                             dtype= tf.int32))

        self.observed_matrix = tf.placeholder(shape=[None, config.num_features], dtype=tf.int32)
        self.tau = tf.placeholder(shape=(), dtype=tf.float32)
        self.num_samples = tf.placeholder(name="num_samples", dtype=tf.int32, shape=())

    def hivae_model(self):
        self.samples = dict()
        self.params_q = dict()
        self.params_p = dict()

        self.X = tf.concat(self.feature_placeholders, axis=-1)

        # encode nlp feature
        if config.nlp:
            encoding_texts = self.get_text_features()
            self.X = tf.concat([self.X, encoding_texts], axis=-1)

        samples_s, params_s, samples_z, params_z = self.encoder()

        self.decoder()

        self.elbo = self.compute_loss()

    ######## LSTM Text Encoder #####
    def initialize_lookup_tables(self):
        self.nlp_lookup_tables = []
        for nlp_feature_id in self.nlp_feature_ids:
            vocab_size = self.data_manager.fields[nlp_feature_id]["vocab_size"]
            lookup_table = tf.get_variable(name="nlp_"+str(nlp_feature_id),
                                           shape=[vocab_size, config.word_emb_size],
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.nlp_lookup_tables.append(lookup_table)

    def get_word_embeddings(self, text_ids, id):
        word_embeddings = tf.nn.embedding_lookup(self.nlp_lookup_tables[id],
                                                 ids=text_ids)
        return word_embeddings

    def lstm_encoder(self, input_word_embeddings, nlp_feature_id):
        _split_x = tf.split(input_word_embeddings,
                                self.data_manager.fields[nlp_feature_id]['max_seq_length'],
                                1 )
        self.split_x = [ tf.squeeze(emb, axis=1) for emb in _split_x ]
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(config.lstm_hidden_size)
        outputs, state = tf.nn.static_rnn(rnn_cell, self.split_x, dtype=tf.float32)
        return outputs[-1], state


    def get_text_features(self):
        self.text_feature_encodings = []
        self.word_embeddings = []
        self.last_states = []
        for id, nlp_feature_id in enumerate(self.nlp_feature_ids):
            word_embeddings = self.get_word_embeddings(text_ids=self.nlp_texts_ids[id],
                                                       id=id)
            feature_encoding, last_state = self.lstm_encoder(word_embeddings, nlp_feature_id)
            self.text_feature_encodings.append(feature_encoding)
            self.word_embeddings.append(word_embeddings)
            self.last_states.append(last_state)
        text_features = tf.concat(self.text_feature_encodings, axis=-1)
        return text_features

    ######## ENCODER ########

    def encoder(self):
        # Compute q(s|x^o)
        samples_s, params_s = self.compute_Ps_given_x()
        self.samples['s'] = samples_s
        self.params_q['s'] = params_s

        # Compute q(z|s,x^o)
        samples_z, params_z = self.compute_Pz_given_xs()
        self.samples['z'] = samples_z
        self.params_q['z'] = params_z

        return samples_s, params_s, samples_z, params_z

    def compute_Ps_given_x(self):
        log_alpha = tf.layers.dense(
                                inputs = self.X,
                                units = config.s_dim,
                                kernel_initializer= tf.contrib.layers.xavier_initializer(),
                                activation=None,
                                reuse=None
                                )

        # Sample using gumbel softmax trick
        u = tf.random_uniform(tf.shape(log_alpha), minval=0, maxval=1)
        _log_logU = -tf.log(-tf.log(u))

        samples_s = tf.nn.softmax(((log_alpha + _log_logU) )/self.tau)
        params_s = log_alpha
        return samples_s, params_s


    def compute_Pz_given_xs(self):

        input = tf.concat([self.samples['s'],self.X], axis=-1)

        #todo: Add more non-linear layers
        for layer_size in config.z_num_layers:
            input = tf.layers.dense(inputs=input,
                                    units= layer_size,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.nn.tanh,
                                    reuse=None)

        mean = tf.layers.dense(inputs = input,
                                units = config.z_dim,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=None,
                                reuse=None
                                )

        logvar = tf.layers.dense(inputs = input,
                                units = config.z_dim,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=None,
                                reuse=None
                                )

        # sample using reparametrization trick
        n_samples = tf.random_normal(shape= tf.shape(mean),mean=0., stddev=1.)
        samples_z = mean + tf.exp(0.5*logvar)*n_samples

        return samples_z, (mean, logvar)

    ######## DECODER ########
    def decoder(self):

        # Compute p(z|s)
        self.params_p['z'] = self.compute_Pz_given_s()

        # Compute Y
        self.samples['y'] = self.compute_y()

        # partition Y to #features
        self.grouped_y = self.group_y()

        # Estimate output distribution parameter(list of parameters for each output distribution)
        self.parameter_estimates = self.estimate_opDist_parameters()

    def compute_Pz_given_s(self):
        input = self.samples['s']
        mean = tf.layers.dense(inputs=input,
                               units=config.z_dim,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=None,
                               reuse=None
                               )
        logvar = tf.zeros([tf.shape(self.samples['s'])[0], config.z_dim])
        return [mean, logvar]

    def compute_y(self, samples_z=None):
        y_total_dims = config.y_dim * config.num_features
        if samples_z==None:
            samples_z = self.samples['z']
        input = samples_z

        #todo Add more non linear layers
        for layer_size in config.z_num_layers:
            input = tf.layers.dense(inputs=input,
                                    units= layer_size,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.nn.tanh,
                                    reuse=None)

        samples_y = tf.layers.dense(inputs = input,
                                    units = y_total_dims,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=None,
                                    reuse = None)
        return samples_y

    def group_y(self, samples_y=None):
        if samples_y == None:
            samples_y = self.samples['y']
        y_groups = []
        y_total_dims = [config.y_dim] * config.num_features
        partition_y_dims = np.insert(np.cumsum(y_total_dims),0,0)
        for i in range(config.num_features):
            y_groups.append(samples_y[:, partition_y_dims[i]:partition_y_dims[i+1]])
        return y_groups

    def estimate_opDist_parameters(self):

        parameters = []
        self.obs_outputs = []
        self.missing_outputs = []
        self.obs_ys = []
        self.missing_ys = []
        self.data_orders = []
        for feature_id, feature in enumerate(self.data_manager.fields):

            y_featureid = self.grouped_y[feature_id]
            # partition observed data and missing Y feature
            missing_y, obs_y = tf.dynamic_partition(y_featureid, self.observed_matrix[:, feature_id], num_partitions=2)
            self.missing_ys.append(missing_y)
            self.obs_ys.append(obs_y)
            #index of observed and missing indices to stitch back
            dataOrder = tf.dynamic_partition(tf.range(tf.shape(self.X)[0]), self.observed_matrix[:, feature_id], num_partitions=2)
            self.data_orders.append(dataOrder)
            current_parameter_estimate = None
            if feature['type'] == 'real':
                current_parameter_estimate = self.estimate_real_parameters(obs_y, missing_y, feature['ndims'], dataOrder, name="x_given_y_for_"+feature['name'])
            elif feature['type'] == 'posReal':
                current_parameter_estimate = self.estimate_posReal_parameters(obs_y, missing_y, feature['ndims'], dataOrder, name="x_given_y_for_"+feature['name'])
            elif feature['type'] == 'count':
                current_parameter_estimate = self.estimate_count_parameters(obs_y, missing_y, feature['ndims'], dataOrder, name="x_given_y_for_"+feature['name'])
            elif feature['type'] == 'ordinal':
                current_parameter_estimate = self.estimate_ordinal_parameters(obs_y, missing_y, feature['ndims'], dataOrder, name="x_given_y_for_"+feature['name'])
            elif feature['type'] == 'categorical':
                current_parameter_estimate = self.estimate_categorical_parameters(obs_y, missing_y, feature['ndims'], dataOrder, name="x_given_y_for_"+feature['name'])
            elif feature['type'] == 'multilabel':
                current_parameter_estimate = self.estimate_categorical_parameters(obs_y, missing_y, feature['ndims'], dataOrder, name="x_given_y_for_"+feature['name'])
            elif config.nlp and feature['type'] == 'nlp':
                current_parameter_estimate = self.estimate_nlp_parameters(y_featureid, missing_y, feature_id, dataOrder, name="x_given_y_for_"+feature['name'])
            parameters.append(current_parameter_estimate)
        return parameters

    def estimate_real_parameters(self, obs_y, missing_y, op_dims, dataOrder, name):
        mean = self.compute_x_given_y(obs_y, missing_y, op_dims, dataOrder, name=name+"_mean")
        logvar = self.compute_x_given_y(obs_y, missing_y, op_dims, dataOrder, name=name+"_stdDev")
        return [mean, logvar]

    def estimate_posReal_parameters(self, obs_y, missing_y, op_dims, dataOrder, name):
        mean = self.compute_x_given_y(obs_y, missing_y, op_dims, dataOrder, name=name+"_mean")
        logvar = self.compute_x_given_y(obs_y, missing_y, op_dims, dataOrder, name=name+"_stdDev")
        return [mean, logvar]

    def estimate_count_parameters(self, obs_y, missing_y, op_dims, dataOrder, name):
        lamda = self.compute_x_given_y(obs_y, missing_y, op_dims, dataOrder, name=name)
        return lamda

    def estimate_ordinal_parameters(self, obs_y, missing_y, op_dims, dataOrder, name):
        logits = self.compute_x_given_y(obs_y, missing_y, op_dims, dataOrder, name=name)
        return logits

    def estimate_categorical_parameters(self, obs_y, missing_y, op_dims, dataOrder, name):
        logits = self.compute_x_given_y(obs_y, missing_y, op_dims, dataOrder, name=name)
        return logits

    def compute_x_given_y(self, obs_y, missing_y, op_dims, dataOrder, name):

        input_obs_y = obs_y
        input_missing_y = missing_y

        for layer_id, layer in enumerate(config.xDecoder_num_layers):
            input_obs_y = tf.layers.dense(inputs=input_obs_y,
                                    units=layer,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.nn.relu,
                                    reuse=None,
                                    name=name+"_"+str(layer_id))
            input_missing_y = tf.layers.dense(inputs=input_missing_y,
                                    units=layer,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.nn.relu,
                                    reuse=True,
                                    name=name+"_"+str(layer_id),
                                    trainable=False)

        obs_output = tf.layers.dense(inputs=input_obs_y,
                                     units=op_dims,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     activation=None,
                                     reuse=None,
                                     name=name)
        missing_output = tf.layers.dense(inputs=input_missing_y,
                                         units=op_dims,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         activation=None,
                                         reuse=True,
                                         name=name,
                                         trainable=False)

        self.obs_outputs.append(obs_output)
        self.missing_outputs.append(missing_output)
        # combine in original order
        output = tf.dynamic_stitch(dataOrder, [missing_output, obs_output])
        return output

    ##### ESTIMATE NLP PARAMETERS ######
    def estimate_nlp_parameters(self, obs_y, missing_y, nlp_feature_id, dataOrder, name):
        # estimate logits for the sequence
        feature = self.data_manager.fields[nlp_feature_id]
        index = self.nlp_feature_ids.index(nlp_feature_id)
        encoder_state = self.last_states[index]
        def get_decoder_lstm_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(config.y_dim)

        initial_state = rnn.LSTMStateTuple(encoder_state[0], obs_y)

        outputs, last_state = tf.nn.static_rnn(cell=get_decoder_lstm_cell(),
                                               inputs=self.split_x,
                                               initial_state=initial_state,
                                               dtype=tf.float32,
                                               scope="decoder")
        # output = tf.reshape(tf.concat(1, outputs), [-1, feature['max_seq_length']])
        logits = tf.contrib.layers.fully_connected(outputs,
                                          num_outputs=feature['vocab_size'],
                                          activation_fn=None)
        return logits

    ######## LOSS FUNCTION ########
    def compute_loss(self):

        # Compute log likelihood of each feature
        self.outputs = []
        for feature_id, feature in enumerate(self.data_manager.fields):
            if feature['type'] == 'real':
                data = self.feature_placeholders[feature_id]
                observed_data_indices = self.observed_matrix[:, feature_id]
                params = self.parameter_estimates[feature_id]
                normalization = None
                output = self.likelihood_real(feature, data, observed_data_indices, params, normalization)

            elif feature['type'] == 'posReal':
                data = self.feature_placeholders[feature_id]
                observed_data_indices = self.observed_matrix[:, feature_id]
                params = self.parameter_estimates[feature_id]
                normalization = None
                output = self.likelihood_real(feature, data, observed_data_indices, params, normalization)

            elif feature['type'] == 'count':
                data = self.feature_placeholders[feature_id]
                observed_data_indices = self.observed_matrix[:, feature_id]
                params = self.parameter_estimates[feature_id]
                normalization = None
                output = self.likelihood_real(feature, data, observed_data_indices, params, normalization)

            elif feature['type'] == 'ordinal':
                data = self.feature_placeholders[feature_id]
                observed_data_indices = self.observed_matrix[:, feature_id]
                params = self.parameter_estimates[feature_id]
                normalization = None
                output = self.likelihood_real(feature, data, observed_data_indices, params, normalization)
            elif feature['type'] == 'categorical':
                data = self.feature_placeholders[feature_id]
                observed_data_indices = self.observed_matrix[:, feature_id]
                params = self.parameter_estimates[feature_id]
                output = self.likelihood_categorical(feature, data, observed_data_indices, params)
            elif feature['type'] == 'multilabel':
                data = self.feature_placeholders[feature_id]
                observed_data_indices = self.observed_matrix[:, feature_id]
                params = self.parameter_estimates[feature_id]
                output = self.likelihood_multilabel(feature, data, observed_data_indices, params)
            elif feature['type'] == 'nlp':
                observed_data_indices = self.observed_matrix[:, feature_id]
                params = self.parameter_estimates[feature_id]
                output = self.likelihood_nlp(params, feature_id, observed_data_indices)

            self.outputs.append(output)

        # Sum up all the likelihoods
        self.likelihood_sum = 0.
        for op in self.outputs:
            self.likelihood_sum += op["likelihood"]

        # Compute KL(q(s|x^o) || p(s))
        s_params = self.params_q['s']
        s_params_dist = tf.nn.softmax(s_params)
        self.kl_s = -tf.nn.softmax_cross_entropy_with_logits(logits=s_params, labels=s_params_dist) + tf.log(float(config.s_dim))
        self.kl_s_sum = tf.reduce_sum(self.kl_s)

        #Compute KL(q(z|x^o, s) || q(z|s))
        mean_pz, log_var_pz = self.params_p['z']
        mean_qz, log_var_qz = self.params_q['z']
        self.kl_z = -0.5*config.z_dim + 0.5*tf.reduce_sum(tf.exp(log_var_qz - log_var_pz) + tf.square(mean_pz - mean_qz)/tf.exp(log_var_pz) - log_var_qz + log_var_pz, 1)
        self.kl_z_sum = tf.reduce_sum(self.kl_z)

        # Compute ELBO loss
        elbo = self.likelihood_sum \
               - config.s_hp*self.kl_s_sum - config.z_hp*self.kl_z_sum

        # Mean loss for batch
        loss = tf.reduce_mean(-elbo)

        return loss

    def likelihood_real(self, feature, data, observed_data_indices, params, normalization):
        est_mean, logvar = params
        est_var = tf.exp(logvar)
        # transform using data mean and var

        # log likelihood of gaussian
        #note: 1
        lik1 = -0.5*tf.reduce_sum(((data - est_mean)**2)/est_var, 1)
        lik2 = -0.5*tf.log(2*np.pi)*feature["ndims"]
        lik3 = -0.5*tf.reduce_sum(tf.log(est_var),1)

        loglikelihood = lik1 + lik2 + lik3

        output = dict()
        output["logpx"] = tf.multiply(tf.cast(observed_data_indices, tf.float32), loglikelihood)
        output["logpx_missing"] = tf.multiply(1.-tf.cast(observed_data_indices, tf.float32), loglikelihood)
        output["params"] = est_mean, est_var
        output["samples"] = tf.contrib.distributions.Normal(est_mean, tf.sqrt(est_var)).sample(sample_shape = self.num_samples)
        output["likelihood"] = tf.reduce_mean(output["logpx"])

        return output

    def likelihood_categorical(self, feature, data, observed_data_indices, params):

        logits = params
        probs = tf.nn.softmax(logits)
        loglikelihood = -tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=data)

        output = dict()
        output["logpx"] = tf.multiply(tf.cast(observed_data_indices, tf.float32), loglikelihood)
        output["logpx_missing"] = tf.multiply(1.-tf.cast(observed_data_indices, tf.float32), loglikelihood)
        output["params"] = params
        output["samples"] = tf.one_hot(tf.contrib.distributions.Categorical(logits = logits).sample(sample_shape = self.num_samples) ,
                                       depth=feature["ndims"])
        output["likelihood"] = tf.reduce_mean(output["logpx"])
        return output

    def likelihood_multilabel(self, feature, data, observed_data_indices, params):

        logits = params
        probs = tf.nn.sigmoid(logits)
        loglikelihood = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=data), axis=-1)

        output = dict()
        output["logpx"] = tf.multiply(tf.cast(observed_data_indices, tf.float32), loglikelihood)
        output["logpx_missing"] = tf.multiply(1.-tf.cast(observed_data_indices, tf.float32), loglikelihood)
        output["params"] = params
        output["samples"] = tf.one_hot(tf.contrib.distributions.Categorical(logits = logits).sample(sample_shape = self.num_samples) ,
                                       depth=feature["ndims"])
        output["likelihood"] = tf.reduce_mean(output["logpx"])
        return output

    def likelihood_nlp(self, logits, feature_id, observed_data_indices):
        index = self.nlp_feature_ids.index(feature_id)
        likelihood = tf.contrib.seq2seq.sequence_loss(logits,
                                                      self.nlp_texts_ids[index],
                                                      tf.ones_like(self.nlp_texts_ids[index], dtype=tf.float32))
        output = dict()
        output["params"] = logits
        output["likelihood"] = -likelihood
        return output

    def generate_samples(self):

        self.num_samples = tf.placeholder(name="num_samples", dtype=tf.int32, shape=())
        # sample from q(z|s,x)
        # mean, logvar = self.params_q["z"]
        mean = tf.zeros(shape=[self.num_samples, config.z_dim], dtype=tf.float32)
        std_dev =  tf.ones(shape=[self.num_samples, config.z_dim], dtype=tf.float32)
        samples_z = tf.contrib.distributions.Normal(mean, std_dev).sample(sample_shape=self.num_samples)

        # compute Y
        samples_y = self.compute_y(samples_z)

        # partition Y to #features
        self.grouped_y = self.group_y(samples_y)

        # Estimate output distribution parameter(list of parameters for each output distribution)
        self.generated_data_list = self.sample_op_dist()

    def sample_op_dist(self):
        samples_list = []

        for feature_id, feature in enumerate(self.data_manager.fields):
            if feature["type"] == "real":
                est_mean, est_logvar = self.parameter_estimates[feature_id]
                est_var = tf.exp(est_logvar)
                samples = tf.contrib.distributions.Normal(est_mean, tf.sqrt(est_var)).sample(sample_shape = self.num_samples)
            if feature["type"] == "categorical":
                logits = self.parameter_estimates[feature_id]
                samples = tf.one_hot(tf.contrib.distributions.Categorical(logits = logits).sample(sample_shape = self.num_samples) ,
                                       depth=feature["ndims"])
            samples_list.append(samples)

        return samples_list


