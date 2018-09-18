import tensorflow as tf
import net_code.tf_utils as U

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError
    def logp(self, x):
        return - self.neglogp(x)


class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError
    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)


class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return CategoricalPd
    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return []
    def sample_dtype(self):
        return tf.int32


class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    # def neglogp(self, x):
    #     # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
    #     # Note: we can't use sparse_softmax_cross_entropy_with_logits because
    #     #       the implementation does not allow second-order derivatives...
    #     one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
    #     return tf.nn.softmax_cross_entropy_with_logits(
    #         logits=self.logits,
    #         labels=one_hot_actions)

    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        dist = tf.log(tf.nn.softmax(self.logits) + 1e-8)
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        cross_entropy = -tf.reduce_sum(tf.multiply(dist, one_hot_actions), axis=-1)

        return cross_entropy

    def kl(self, other):
        a0 = self.logits - U.max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - U.max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = U.sum(ea0, axis=-1, keepdims=True)
        z1 = U.sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - U.max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = U.sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)