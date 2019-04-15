import config
import tensorflow as tf
import numpy as np

class HI_VAE(object):

    def __init__(self, data_manager):

        self.data_manager = data_manager
        self.init_placeholders()

    def init_placeholders(self):
        self.feature_placeholders = []
        for feature in self.data_manager.fields:
            self.feature_placeholders.append(tf.get_variable(name=feature['name'],
                                                             shape= [ None, feature['ndim']],
                                                             dtype= tf.float32,
                                                             initializer=tf.contrib.layers.xavier_initializer)
                                             )
        self.observed_matrix = tf.placeholder(shape=[None, config.num_features])
        self.tau = tf.placeholder(dtype=tf.float32)


    def hivae_model(self):
        self.samples = dict()
        self.params = dict()

        self.X = tf.concat(self.feature_placeholders, axis=-1)

        samples_s, params_s, samples_z, params_z = self.encoder()

        self.decoder()

        self.elbo = self.compute_loss()

    ######## ENCODER ########

    def encoder(self):
        # Compute P(s|x^o)
        samples_s, params_s = self.compute_Ps_given_x()
        self.samples['s'] = samples_s
        self.params['s'] = params_s

        # Compute P(z|s,x^o)
        samples_z, params_z = self.compute_Pz_given_xs()
        self.samples['z'] = samples_z
        self.params['z'] = params_z

        return samples_s, params_s, samples_z, params_z

    def compute_Ps_given_x(self):
        log_alpha = tf.layers.dense(
                                inputs = self.X,
                                units = config.s_dim,
                                kernel_initializer= tf.contrib.layers.xavier_initializer,
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

        input = tf.concat([self.X, self.samples['s']], axis=-1)

        #todo: Add more non-linear layers

        mean = tf.layers.dense(inputs = input,
                                units = config.z_dim,
                               kernel_initializer=tf.contrib.layers.xavier_initializer,
                               activation=None,
                                reuse=None
                                )

        logvar = tf.layers.dense(inputs = input,
                                units = config.z_dim,
                                activation=None,
                                reuse=None
                                )

        # sample using reparametrization trick
        n_samples = tf.random_normal(shape= tf.shape(mean),mean=0., stddev=1.)
        samples_z = mean + tf.exp(0.5*logvar)*n_samples

        return samples_z, (mean, logvar)

    ######## DECODER ########
    def decoder(self):

        # Compute Y
        self.samples['y'] = self.compute_y()

        # partition Y to #features
        self.grouped_y = self.group_y()

        # Estimate output distribution parameter(list of parameters for each output distribution)
        self.parameter_estimates = self.estimate_opDist_parameters()

    def compute_y(self):
        y_total_dims = [config.y_dim] * config.num_features
        #todo Add more non linear layers
        samples_y = tf.layers.dense(input = self.samples['z'],
                                    units = y_total_dims,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer,
                                    activation=None,
                                    resuse = None)
        return samples_y

    def group_y(self):
        y_groups = []
        y_total_dims = [config.y_dim] * config.num_features
        partition_y_dims = np.insert(np.cumsum(y_total_dims),0,0)
        for i in range(config.num_features):
            y_groups.append(self.samples['y'][:, partition_y_dims[i]:partition_y_dims[i]+1])
        return y_groups

    def estimate_opDist_parameters(self):

        parameters = []

        for feature_id, feature in enumerate(self.data_manager.fields):

            y_featureid = self.grouped_y[feature_id]
            # partition observed data and missing Y feature
            obs_y, missing_y = tf.dynamic_partition(y_featureid, self.observed_matrix[:, feature_id], num_partitions=2)
            #index of observed and missing indices to stitch back
            dataOrder = tf.dynamic_partition(tf.range(tf.shape(y_featureid)[0]), self.observed_matrix[:, feature_id], num_partitions=2)

            current_parameter_estimate = None
            if feature['type'] == 'real':
                current_parameter_estimate = self.estimate_real_parameters(obs_y, missing_y, feature['ndim'], dataOrder, name="x_given_y_for_"+feature['name'])
            elif feature['type'] == 'posReal':
                current_parameter_estimate = self.estimate_posReal_parameters(obs_y, missing_y, feature['ndim'], dataOrder, name="x_given_y_for_"+feature['name'])
            elif feature['type'] == 'count':
                current_parameter_estimate = self.estimate_count_parameters(obs_y, missing_y, feature['ndim'], dataOrder, name="x_given_y_for_"+feature['name'])
            elif feature['type'] == 'ordinal':
                current_parameter_estimate = self.estimate_ordinal_parameters(obs_y, missing_y, feature['ndim'], dataOrder, name="x_given_y_for_"+feature['name'])
            elif feature['type'] == 'categorical':
                current_parameter_estimate = self.estimate_categorical_parameters(obs_y, missing_y, feature['ndim'], dataOrder, name="x_given_y_for_"+feature['name'])
            parameters.append(current_parameter_estimate)
        return parameters

    def estimate_real_parameters(self, obs_y, missing_y, op_dims, dataOrder, name):
        mean = self.compute_x_given_y(obs_y, missing_y, op_dims, dataOrder, name=name)
        std_dev = self.compute_x_given_y(obs_y, missing_y, op_dims, dataOrder, name=name)
        return [mean, std_dev]

    def estimate_posReal_parameters(self, obs_y, missing_y, op_dims, dataOrder, name):
        mean = self.compute_x_given_y(obs_y, missing_y, op_dims, dataOrder, name=name)
        std_dev = self.compute_x_given_y(obs_y, missing_y, op_dims, dataOrder, name=name)
        return [mean, std_dev]

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
        obs_output = tf.layers.dense(inputs=obs_y,
                                units=op_dims,
                                kernel_initializer=tf.contrib.layers.xavier_initializer,
                                activation=None,
                                resuse=True,
                                name=name)
        missing_output = tf.layers.dense(inputs=missing_y,
                                units=op_dims,
                                kernel_initializer=tf.contrib.layers.xavier_initializer,
                                activation=None,
                                resuse=False,
                                name=name,
                                trainable=False)

        # combine in original order
        output = tf.dynamic_stitch(dataOrder, [obs_output, missing_output])
        return output


    ######## LOSS FUNCTION ########
    def compute_loss(self):

        # Compute max log likelihood of each feature
        pass

        # Sum up all the likelihoods
        likelihood_sum = 0
        pass

        # Compute KL(q(s|x^o) || p(s))
        kl_s = 0
        pass

        #Compute KL(q(z|x^o, s) || q(z|s))
        kl_z = 0
        pass

        # Compute ELBO loss
        elbo = likelihood_sum - kl_s - kl_z

        # Mean loss for batch
        loss = tf.reduce_mean(elbo)

        return loss
