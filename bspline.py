import numpy as np


class Bspline(object):
    """
Class to construct regression or smoothing splines in a B-spline basis. See :cite:`eilers_flexible_1996`, :cite:`hastie_elements_2001`. If a smoothing spline is desired, only one vector of locations is to be passed to the constructor (since a knot is placed at every data location), otherwise a vector of knots also has to be passed, with a length < that of the location vector.

**Notes:**

    * The difference matrix is currently hardcoded to second order, equivalent to a penalization of the second derivative of the spline.
    * I'm augmenting the knot sequence with integer sequences beyond the two boundary knots. Should be checked if that works.
    * Also check treatment of boundary knots.

:param locs: Vector of values at which to evaluate the B-spline basis functions.
:type locs: :obj:`list` or :class:`~numpy.ndarray`
:param knots: Vector of knots, including the 'boundary' knots. If ``None``, will be set to be equal to ``x``.
:type knots: :obj:`list`, :class:`numpy.ndarray` or :obj:`None`
:param order: Order of the spline (``4`` is cubic).

.. bibliography:: bspline.bib

    """
    def __init__(self, locs, knots=None, order=4):
        x = np.asarray(locs).reshape((-1, 1))
        if knots is None:
            t = x.flatten()
        else:
            t = np.asarray(knots).flatten()
        a = np.arange(1, order + 1)
        T = np.r_[locs[0] - a[::-1], t, a + locs[-1]]
        B = (x >= T[:-1]) & (x < T[1:])

        for m in range(1, order):
            B = (x - T[:-m-1]) / (T[m:-1] - T[:-m-1]) * B[:, :-1] + \
                (T[m+1:] - x) / (T[m+1:] - T[1:-m]) * B[:, 1:]

        self.basis = B

    @property
    def diff2(self):
        # the first two elements of D * vector are nixed because it's second order differences
        o = np.ones(self.basis.shape[1] - 2)
        return np.diag(np.r_[0, 0, o]) - 2 * np.diag(np.r_[0, o], -1) + np.diag(o, -2)

    def fit(self, y, l=None):
        """This fits either a regression spline or a smoothing spline (if ``l`` is given).

        :param y: Vector of data values. May have :class:`~numpy.nan` values.
        :type y: :obj:`list` or :class:`~numpy.ndarray`
        :param l: Regularizing parameter (:math:`\lambda` in :cite:`eilers_flexible_1996`, :cite:`hastie_elements_2001`). If ``None``, a regression spline is fit, otherwise a smoothing spline.
        :type l: :obj:`float` or :obj:`None`
        :returns: Smoothing spline values at the same locations as input.
        :rtype: :class:`~numpy.ndarray`

        """
        x = np.asarray(y).reshape((-1, 1))
        i = np.isfinite(x).flatten()
        x = x[i]
        B = self.basis[i, :]
        if l is not None:
            D = self.diff2[2:, :]
            B = np.r_['0', B, l * D]
            x = np.pad(x, ((0, D.shape[0]), (0, 0)), 'constant')
        self.spline = self.basis.dot(np.linalg.lstsq(B, x)[0])

    def fit_df(self, y, l):
        """This uses the smoothing matrix (S) formulation, whose trace is the definition of the effective number of degrees of freedom. Otherwise the same as :meth:`.smooth`.

        """
        import tensorflow as tf
        b = np.asarray(y).reshape((-1, 1))
        A_inv = tf.matrix_inverse(self.basis.T.dot(self.basis) + l * self.D.T.dot(self.D))
        S = tf.matmul(tf.matmul(self.basis, A_inv), self.basis, transpose_b=True)
        print(tf.trace(S).eval())
        return tf.matmul(S, b.reshape((-1, 1)))

    def fit2(self, y1, y2, l=None):
        """Fit a regression or smoothing spline consisting of two pieces which may be offset by an additive constant with respect to each other. The offset is accessible as :attr:`.offset` attribute, the fitted spline as :attr:`.spline`. The second one is assumed to be the offset one; if the truely offset one is the first one, the value of the computed offset simply needs to be negated.

        :param y1: First vector of data values.
        :type y1: :obj:`list` or :class:`~numpy.ndarray`
        :param y2: Second vector of data values. (This vector is assumed to be offset.)
        :type y2: :obj:`list` or :class:`~numpy.ndarray`
        :param l: Penalty parameter for the smoothing (denoted :math:`\lambda` by :cite:`eilers_flexible_1996`, :cite:`hastie_elements_2001`). If ``None``, a regression spline is fit, otherwise a smoothing spline.
        :type l: :obj:`float` or :obj:`None`

        """
        z1 = np.zeros_like(y1).flatten()
        z2 = np.ones_like(y2).flatten()
        y = np.hstack((np.asarray(y1).flatten(), np.asarray(y2).flatten()))
        i = np.isfinite(y)
        x = y[i].reshape((-1, 1))
        z = np.hstack((z1, z2))
        B = np.r_['1', self.basis, z.reshape((-1, 1))][i, :]
        if l is not None:
            D = np.pad(self.diff2, ((0, 0), (0, 1)), 'constant')
            B = np.r_['0', B, l * D]
            x = np.pad(x, ((0, self.basis.shape[1]), (0, 0)), 'constant')
        s = np.linalg.lstsq(B, x)[0]
        self.offset = s[-1].item()
        self.spline = self.basis.dot(s[:-1]).flatten()
        self.resid = y - z * self.offset - self.spline

    def set_df(self, df, learn=0.01, n_iter=100):
        """TensorFlow_-based optimization routine to compute :math:`\lambda` from effective number of degrees of freedom.

        """
        import tensorflow as tf
        gr = tf.Graph()
        self.sess = tf.Session(graph=gr)
        with gr.as_default():
            self.lmbd = tf.get_variable('lambda', (), tf.float64, tf.constant_initializer(200.))
            A_inv = tf.matrix_inverse(self.basis.T.dot(self.basis) + self.lmbd * self.D.T.dot(self.D))
            S = tf.matmul(tf.matmul(self.basis, A_inv), self.basis, transpose_b=True)
            loss = (tf.trace(S) - df) ** 2
            op = tf.train.AdamOptimizer(learn).minimize(loss)
            tf.global_variables_initializer().run(session=self.sess)
            prog = tf.keras.utils.Progbar(n_iter)
            for i in range(n_iter):
                _, l = self.sess.run([op, loss])
                prog.update(i, [('Loss', l)])

if __name__=="__main__":
    x = np.linspace(0, 1, 100)
    f = np.sin(12 * (x + .2)) / (x + .2)
    f[49:51] = np.nan
    y1 = f[:50]# + np.random.randn(50)
    y2 = f[50:] + 2# + np.random.randn(50)
    y = np.hstack((y1, y2))
    b = Bspline(x)
    # b.fit2(y1, y2, 1, .1)
    # b = Bspline(x, x[::10])
    # b.fit2(y1, y2, 1)
