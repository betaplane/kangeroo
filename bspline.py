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
            t = np.unique(locs).flatten()
            t.sort()
        else:
            t = np.asarray(knots).flatten()
        a = np.arange(1, order + 1)
        T = np.r_[min(x) - a[::-1], t, a + max(x)]
        B = (x >= T[:-1]) & (x < T[1:])

        for m in range(1, order):
            B = (x - T[:-m-1]) / (T[m:-1] - T[:-m-1]) * B[:, :-1] + \
                (T[m+1:] - x) / (T[m+1:] - T[1:-m]) * B[:, 1:]

        self.basis = B

    def asymmetric2ndDerivativeMatrix(self, pad=False):
        """This should correspond to :cite:`eilers_flexible_1996`, but I'm not sure if it's correct - see the symmetric variant below. It does seem to give the 'more correct' results though, minus the (lower) boundary effects.
        """
        # the first two elements of D * vector are nixed because it's second order differences
        o = np.ones(self.basis.shape[1] - 2)
        D = np.diag(np.r_[0, 0, o]) - 2 * np.diag(np.r_[0, o], -1) + np.diag(o, -2)
        return np.pad(D, ((0, 0), (0, 1)), 'constant') if pad else D

    def symmetric2ndDerivativeMatrix(self, pad=False):
        """This is a symmetric variant of the 2nd derivative matrix. Have to check what is really correct since I'm using the B-spline construction from :cite:`hastie_elements_2001`, not from :cite:`eilers_flexible_1996`. Also not sure my derivation of the upwind/downwind second differences at the boundaries is correct.
        """
        o = np.ones(self.basis.shape[1])
        D = np.diag(o[1:], -1) - 2 * np.diag(o) + np.diag(o[1:], 1)
        D[0, :3] = [1, -2, 1]
        D[-1, -3:] = [1, -2, 1]
        return np.pad(D, ((0, 0), (0, 1)), 'constant') if pad else D


    def fit(self, y, l=None, split_index=None):
        """This fits either a regression spline or a smoothing spline (if ``l`` is given). Additionally, an additive offset can be fit by giving the ``split_index`` argument, which gives the index at which the input data ``y`` is assumed to be split into two separate pieces (see below). The offset is calculated such that the second piece is assumed to have the offset added with respect to the first. The fit spline is available as attribute :attr:`.spline`, the offset (if applicable) as :attr:`.offset` and the residuals of the data w.r.t. the spline in :attr:`.resid`.

        :param y: Vector of data values. May have :class:`~numpy.nan` values.
        :type y: :obj:`list` or :class:`~numpy.ndarray`
        :param l: Regularizing parameter (:math:`\lambda` in :cite:`eilers_flexible_1996`, :cite:`hastie_elements_2001`). If ``None``, a regression spline is fit, otherwise a smoothing spline.
        :type l: :obj:`float` or :obj:`None`
        :param split_index: If an additive offset is to be fit, ``split_index`` denotes the index that splits the given timeseries into two parts (say ``a`` and ``b``) according to the usual python indexing rules, i.e. ``a, b = y[:split_index], y[split_index:]``.
        :type split_index: int
        :returns: ``Bspline`` object, for chaining (e.g. ``bsp = Bspline(x).fit(y)``), with attributes ``spline``, ``offset`` (if applicable) and ``resid``/
        :rtype: :class:`Bspline`

        """
        self.resid = np.ma.masked_invalid(y).flatten()
        i = ~self.resid.mask
        x = self.resid[i].reshape((-1, 1))
        B = self.basis[i, :]
        if split_index is not None:
            z = np.r_[np.zeros(split_index), np.ones(len(y) - split_index)]
            B = np.r_['1', B, z[i].reshape((-1, 1))]
        if l is not None:
            # D = self.symmetric2ndDerivativeMatrix(split_index is not None)
            D = self.asymmetric2ndDerivativeMatrix(split_index is not None)[2:, :]
            B = np.r_['0', B, l * D]
            x = np.pad(x, ((0, D.shape[0]), (0, 0)), 'constant')
        a = np.linalg.lstsq(B, x)[0]
        if split_index:
            a, self.offset = a[:-1], a[-1].item()
            self.resid -= self.offset * z
        self.spline = self.basis.dot(a).flatten()
        self.resid -= self.spline


if __name__=="__main__":
    x = np.linspace(0, 1, 100)
    f = np.sin(12 * (x + .2)) / (x + .2)
    f[49:51] = np.nan
    y1 = f[:50]# + np.random.randn(50)
    y2 = f[50:] + 2# + np.random.randn(50)
    y = np.hstack((y1, y2))
    b = Bspline(x)
    b.fit(y, 1, 50)
