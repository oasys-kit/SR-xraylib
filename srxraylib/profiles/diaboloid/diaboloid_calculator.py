################## LSBL Reports #################################################################

# Yashchuk, V. V., “An analytical solution for shape of diaboloid mirror,” Light
# Source Beam Line Note LSBL-1436, Advanced Light Source, Berkeley (January 8, 2020).

# Yashchuk, V. V., “Shape of diaboloid mirror in laboratory coordinate system,”
# Light Source Beam Line Note LSBL-1437c, Advanced Light Source, Berkeley (January 13, 2020).

# K. Goldberg and Manuel Sanchez del Rio, “Direct Solution of
# Diaboloid Mirrors,” Light Source Beam Line Note LSBL- 1440 (ALS, Berkeley, February 18, 2020).

# Lacey, I., Sanchez del Rio, M., and Yashchuk, V. V., “Analytical expression for
# the diaboloid shape in laboratory mirror coordinates verified by ray-tracing simulations,” Light
# Source Beam Line Note LSBL-1445, Advanced Light Source, Berkeley (March 24, 2020).

# Yashchuk, V. V., “Diaboloid shape approximation with a sagittal conical cylinder
# bent to a tangential parabola: Analytical consideration,” Light Source Beam Line Note LSBL-1451,
# Advanced Light Source, Berkeley (April 14, 2020).

# Yashchuk, V. V., “Explicit algebraic derivation of an expression for the exact
# shape of diaboloid mirror in laboratory coordinate system,”
# Light Source Beam Line Note LSBL˗1462, Advanced Light Source, Berkeley (May 17, 2020).

# H. A. Padmore, “Sagittal shape difference between a cylinder and a diaboloid at 2:1
# demagnification,” Light Source Beam Line Note LSBL-1465, Advanced Light Source, Berkeley
# (June 03, 2020).

################## Papers #################################################################

# W. R. McKinney, J. M. Glossinger, H. A. Padmore, and M. R. Howells,
# "Optical path function calculation for an incoming cylindrical wave," Proc. SPIE. 7448, 744809/1-8
# (2009); https://doi.org/10.1117/12.828490.

# V. V. Yashchuk, I. Lacey, and M. Sanchez del Rio, “Analytical expressions
# of the surface shape of ‘diaboloid’ mirrors,” Proc. SPIE 11493, 114930N/1-13 (2020);
# https://doi.org/10.1117/12.2568332.



import numpy

from srxraylib.profiles.diaboloid.fqs import single_quartic
from srxraylib.profiles.diaboloid.fqs import quartic_roots

def diaboloid_approximated_point_to_segment(
        p=29.3,
        q=19.53,
        theta=4.5e-3,
        x=numpy.linspace(-0.01, 0.01, 101),
        y=numpy.linspace(-0.1, 0.1, 1001),
        detrend=0):
    """
    Create a numeric mesh of the mirror using the approximated equation of diaboloid (point to segment)
    References:

    Kenneth Goldberg & M. Sanchez del Rio
    "Direct Solution of Diaboloid Mirrors"
    LSBL-1440 (February 2020)

    M. Sanchez del Rio, K. Goldberg, V. Yashchuk, I. Lacey and H. Padmore
    Simulations of applications using diaboloid mirrors
    Journal of Synchrotron Radiation (submitted 2021)

    :param p: distance source to mirror [m]
    :param q: distance mirror to focus [m]
    :param theta: grazing incidence angle [rad]
    :param y: x (sagittal) array
    :param y: y (tangential) array
    :return: Z, X, Y
    """
    X = numpy.outer(x, numpy.ones_like(y))
    Y = numpy.outer(numpy.ones_like(x), y)

    s = p * numpy.cos(2 * theta)
    z0 = p * numpy.sin(2 * theta)
    c = p + q

    # this is Eq. 12 in LSBL-1440 (a bit reorganised)
    Z = - numpy.sqrt(c ** 2 + q ** 2 - s ** 2 - 2 * Y * (s + q) - 2 * c * numpy.sqrt(X ** 2 + (q - Y) ** 2))
    Z += z0

    if detrend == 0:
        zfit = 0
    elif detrend == 1:
        zfit = -theta * y
    elif detrend == 2:
        zcentral = Z[Z.shape[0] // 2, :]
        zcoeff = numpy.polyfit(y[(y.size // 2 - 10):(y.size // 2 + 10)],
                               zcentral[(y.size // 2 - 10):(y.size // 2 + 10)], 1)
        zfit = zcoeff[1] + y * zcoeff[0]

    for i in range(Z.shape[0]):
        Z[i, :] = Z[i, :] - zfit

    return Z, X, Y


def diaboloid_approximated_segment_to_point(
        p=29.3,
        q=19.53,
        theta=4.5e-3,
        x=numpy.linspace(-0.01, 0.01, 101),
        y=numpy.linspace(-0.1, 0.1, 1001),
        detrend=0):
    """
    Create a numeric mesh of the mirror using the approximated equation of diaboloid (segmet to point)
    References:

    Kenneth Goldberg & M. Sanchez del Rio
    "Direct Solution of Diaboloid Mirrors"
    LSBL-1440 (February 2020)

    M. Sanchez del Rio, K. Goldberg, V. Yashchuk, I. Lacey and H. Padmore
    Simulations of applications using diaboloid mirrors
    Journal of Synchrotron Radiation (submitted 2021)

    :param p: distance source to mirror [m]
    :param q: distance mirror to focus [m]
    :param theta: grazing incidence angle [rad]
    :param y: x (sagittal) array
    :param y: y (tangential) array
    :return: Z, X, Y
    """

    Z, X, Y = diaboloid_approximated_point_to_segment(p=q, q=p, theta=theta, x=x, y=y,
                                              detrend=detrend)
    for i in range(x.size):
        Z[i,:] = numpy.flip(Z[i,:])

    return Z, X, Y

def toroid_point_to_segment(
        p=29.3,
        q=19.53,
        theta=4.5e-3,
        x=numpy.linspace(-0.01, 0.01, 101),
        y=numpy.linspace(-0.1, 0.1, 1001)):
    """

    Create a numeric mesh of the toroid (point to segment)

    :param p: distance source to mirror [m]
    :param q: distance mirror to focus [m]
    :param theta: grazing incidence angle [rad]
    :param y: x (sagittal) array
    :param y: y (tangential) array
    :return: Z, X, Y
    """

    Rt = 2.0 / numpy.sin(theta) / (1 / p)

    Rs = 2.0 * numpy.sin(theta) / (1 / p + 1 / q)

    print("Toroid Rt: %9.6f m, Rs: %9.6f m" % (Rt, Rs))

    height_tangential = Rt - numpy.sqrt(Rt ** 2 - y ** 2)
    height_sagittal = Rs - numpy.sqrt(Rs ** 2 - x ** 2)

    Z = numpy.zeros((x.size, y.size))

    for i in range(x.size):
        Z[i,:] = height_tangential

    for i in range(y.size):
        Z[:,i] += height_sagittal

    return Z

def toroid_segment_to_point(
        p=29.3,
        q=19.53,
        theta=4.5e-3,
        x=numpy.linspace(-0.01, 0.01, 101),
        y=numpy.linspace(-0.1, 0.1, 1001)):
    """

    Create a numeric mesh of the toroid (segment to point)

    :param p: distance source to mirror [m]
    :param q: distance mirror to focus [m]
    :param theta: grazing incidence angle [rad]
    :param y: x (sagittal) array
    :param y: y (tangential) array
    :return: Z, X, Y
    """

    Z = toroid_point_to_segment(p=q, q=p, theta=theta, x=x, y=y)
    for i in range(x.size):
        Z[i,:] = numpy.flip(Z[i,:])
    return Z

def parabolic_cone_point_to_segment(
        p=29.3,
        q=19.53,
        theta=4.5e-3,
        x=numpy.linspace(-0.01, 0.01, 101),
        y=numpy.linspace(-0.1, 0.1, 1001)):
    """
    Create a numerical mesh for an approximated diaboloid (point to segment focusing)
    as calculated by Valeriy Yashchuk

    Valeriy V. Yashchuk
    Diaboloid shape approximation with a sagittal conical cylinder bent to a tangential parabola:
    Analytical consideration
    Light Source Beam Line Note LSBL-1451, Advanced Light Source, Berkeley (April 14, 2020).


    :param p: distance source to mirror [m]
    :param q: distance mirror to focus [m]
    :param theta: grazing incidence angle [rad]
    :param y: x (sagittal) array
    :param y: y (tangential) array
    :return: Z, X, Y
    """
    X = numpy.outer(x, numpy.ones_like(y))
    Y = numpy.outer(numpy.ones_like(x), y)

    c = numpy.cos(theta)
    s = numpy.sin(theta)
    c2 = numpy.cos(2 * theta)
    s2 = numpy.sin(2 * theta)
    pq = p + q

    # Equation 15 in V. Yashchuk LSBL 1451
    k1 = p * q * c * s2 / pq
    k2 = s2 * (q - 2 * p * c**2 ) / 2 / pq
    Z = Y * s / c - \
        2  * s / c**2 * numpy.sqrt(Y * p * c + p**2) + \
        2 * p * s / c**2 + \
        k1 + k2 * Y \
        - numpy.sqrt( (k1 + k2 * Y)**2 - X**2 )

    return Z, X, Y

def parabolic_cone_segment_to_point(
        p=29.3,
        q=19.53,
        theta=4.5e-3,
        x=numpy.linspace(-0.01, 0.01, 101),
        y=numpy.linspace(-0.1, 0.1, 1001)):
    """
    Create a numerical mesh for an approximated diaboloid (point to segment focusing)
    as calculated by Valeriy Yashchuk

    Valeriy V. Yashchuk
    Diaboloid shape approximation with a sagittal conical cylinder bent to a tangential parabola:
    Analytical consideration
    Light Source Beam Line Note LSBL-1451, Advanced Light Source, Berkeley (April 14, 2020).

    :param p: distance source to mirror [m]
    :param q: distance mirror to focus [m]
    :param theta: grazing incidence angle [rad]
    :param y: x (sagittal) array
    :param y: y (tangential) array
    :return: Z, X, Y
    """
    Z, X, Y = parabolic_cone_point_to_segment(p=q, q=p, theta=theta, x=x, y=y)
    for i in range(x.size):
        Z[i,:] = numpy.flip(Z[i,:])

    return Z, X, Y


def parabolic_cone_linearized_point_to_segment(
        p=29.3,
        q=19.53,
        theta=4.5e-3,
        x=numpy.linspace(-0.01, 0.01, 101),
        y=numpy.linspace(-0.1, 0.1, 1001)):
    """

    Create a numerical mesh for the linearized approximated diaboloid (point to segment focusing)
    as calculated by Valeriy Yashchuk

    References:

    Valeriy V. Yashchuk
    Diaboloid shape approximation with a sagittal conical cylinder bent to a tangential parabola:
    Analytical consideration
    Light Source Beam Line Note LSBL-1451, Advanced Light Source, Berkeley (April 14, 2020).

    :param p: distance source to mirror [m]
    :param q: distance mirror to focus [m]
    :param theta: grazing incidence angle [rad]
    :param y: x (sagittal) array
    :param y: y (tangential) array
    :return: Z, X, Y
    """
    X = numpy.outer(x, numpy.ones_like(y))
    Y = numpy.outer(numpy.ones_like(x), y)

    c = numpy.cos(theta)
    s = numpy.sin(theta)
    c2 = numpy.cos(2 * theta)
    s2 = numpy.sin(2 * theta)
    pq = p + q

    # use the central meridional profile like in parabolic_cone_point_to_segment()
    Z = Y * s / c - 2  * s / c**2 * numpy.sqrt(Y * p * c + p**2) + 2 * p * s / c**2 \
        - numpy.sqrt( \
        (p * q * c * s2 / pq + s2 * (q - 2 * p * c**2 ) / 2 / pq * Y)**2 - (X*0)**2) + \
        p * q * c * s2 / pq + s2 * (q - 2 * p * c**2) / 2 / pq * Y

    # we add now the sagittal profile with radius calculated using Eq 11 in LSBL 1451
    for j in range(y.size):
        # Rs = p * q * numpy.sin(2 * theta) / (p + q) # TODO missing c:  = 2 p * q * sin(theta) * cos^2(theta)
        # Rs += (q * numpy.tan(theta) - 2 * p * numpy.sin(theta) * numpy.cos(theta)) / (p + q) * y[j]
        Rs = p * q * c * numpy.sin(2 * theta) / (p + q)
        Rs += numpy.sin(2 * theta) * (q  - 2 * p * c**2) / 2 / (p + q) * y[j]
        height_sagittal = Rs - numpy.sqrt(Rs ** 2 - x ** 2)
        print("y=%f Rs=%f" % (y[j], Rs))
        Z[:,j] += height_sagittal

    return Z, X, Y

def parabolic_cone_linearized_segment_to_point(
        p=29.3,
        q=19.53,
        theta=4.5e-3,
        x=numpy.linspace(-0.01, 0.01, 101),
        y=numpy.linspace(-0.1, 0.1, 1001)):
    """

    Create a numerical mesh for the linearized approximated diaboloid (segment to point focusing)
    as calculated by Valeriy Yashchuk

    Valeriy V. Yashchuk
    Diaboloid shape approximation with a sagittal conical cylinder bent to a tangential parabola:
    Analytical consideration
    Light Source Beam Line Note LSBL-1451, Advanced Light Source, Berkeley (April 14, 2020).

    :param p: distance source to mirror [m]
    :param q: distance mirror to focus [m]
    :param theta: grazing incidence angle [rad]
    :param y: x (sagittal) array
    :param y: y (tangential) array
    :return: Z, X, Y
    """
    Z, X, Y = parabolic_cone_linearized_point_to_segment(p=q, q=p, theta=theta, x=x, y=y)
    for i in range(x.size):
        Z[i,:] = numpy.flip(Z[i,:])

    return Z, X, Y

def diaboloid_exact_point_to_segment(
        p=29.3,
        q=19.53,
        theta=4.5e-3,
        x=numpy.linspace(-0.01, 0.01, 101),
        y=numpy.linspace(-0.1, 0.1, 1001),
        ):
    """

    Create a numerical mesh for the exact diaboloid (point to segment focusing) solving
    a fourth degree equation, as calculated by Valeriy Yashchuk

    References:

    Valeriy V. Yashchuk
    Explicit algebraic derivation of an expression for the exact shape of diaboloidal
    mirror in laboratory coordinate system
    Light Source Beam Line Note LSBL˗1462, Advanced Light Source, Berkeley (May 17, 2020).

    Yashchuk, V. V., Goldberg, K., Lacey, I., McKinney, W. R., Sanchez del Rio, M. & Padmore,H.
    Journal of Synchrotron Radiation, submitted(2021).

    :param p: distance source to mirror [m]
    :param q: distance mirror to focus [m]
    :param theta: grazing incidence angle [rad]
    :param y: x (sagittal) array
    :param y: y (tangential) array
    :return: Z, X, Y
    """

    X = numpy.outer(x, numpy.ones_like(y))
    Y = numpy.outer(numpy.ones_like(x), y)

    c = numpy.cos(theta)
    s = numpy.sin(theta)

    c2 = numpy.cos(2 * theta)
    s2 = numpy.sin(2 * theta)

    # Coefficients after Valeriy V. Yashchuk LSBL-1462 May 2020
    # A = −Cos[θ] 4 .
    # B = 4(r1 − r2)Cos[θ] 2 Sin[θ] + 4Cos[θ] 3 Sin[θ]z;
    # C = 4r2((r1 + r2)Cos[θ] 2 + 4r1Sin[θ] 2 ) + 2Cos[θ](−3r1 + r2 + (r1 − 3r2)Cos[2θ])z − 6(Cos[θ] 2 Sin[θ] 2 )z 2 ;
    # D = −16r1r2(r1 + r2)Sin[θ] + 4(r1 + r2)(2r1 − r2)Sin[2θ]z + 2(3r1 + r2 + (r1 + 3r2)Cos[2θ])Sin[θ]z 2 + 4Cos[θ]Sin[θ] 3 z 3 ;
    # E = 4(r1 + r2) 2 x 2 + 4r2(r1 + r2)Sin[θ] 2 z 2 − 4((r1 + r2)Cos[θ]Sin[θ] 2 )z 3 − Sin[θ] 4 z 4 ;

    A = -c**4 * numpy.ones_like(X)
    B = 4 * (p - q) * c**2 * s \
                + 4 * c**3 * s * Y
    C = 4 * q * ( (p + q) * c**2 + 4 * p * s**2 ) \
                + 2 * c * (q - 3 * p + (p - 3 * q) * c2) * Y \
                - 6 * c**2 * s**2 * Y**2
    D = -16 * p * q * (p + q) * s \
                + 4 * (p + q) * (2 * p - q) * s2 * Y \
                + 2 * (3 * p + q + (3 * q + p) * c2) * s * Y**2 \
                + 4 * c * s**3 * Y**3
    E = 4 * (p + q)**2 * X**2 \
            + 4 * q * (p + q) * s**2 * Y**2 \
            - 4 * (p + q) * c * s**2 * Y**3 \
            - s**4 * Y**4

    # get good solution: the one that is zero at (0,0)
    ix = x.size // 2
    iy = y.size // 2
    solutions = single_quartic(A[ix, iy], B[ix, iy], C[ix, iy], D[ix, iy], E[ix, iy])
    aa = []
    for sol in solutions:
        if numpy.abs(sol.imag) < 1e-15:
            aa.append(numpy.abs(sol.real))
        else:
            aa.append(1e10)
    isel = numpy.argmin(aa)


    # calculate solutions array
    P = numpy.zeros((A.size, 5))
    P[:, 0] = A.flatten()
    P[:, 1] = B.flatten()
    P[:, 2] = C.flatten()
    P[:, 3] = D.flatten()
    P[:, 4] = E.flatten()
    SOLUTION = quartic_roots(P)

    # return result
    SOLUTION_GOOD = (SOLUTION[:,isel]).flatten()
    SOLUTION_GOOD.shape = A.shape
    Z = SOLUTION_GOOD.real
    return Z, X, Y

def diaboloid_exact_segment_to_point(
        p=29.3,
        q=19.53,
        theta=4.5e-3,
        x=numpy.linspace(-0.01, 0.01, 101),
        y=numpy.linspace(-0.1, 0.1, 1001),
        ):
    """

    Create a numerical mesh for the exact diaboloid (segment to point focusing) solving
    a fourth degree equation, as calculated by Valeriy Yashchuk

    References:

    Valeriy V. Yashchuk
    Explicit algebraic derivation of an expression for the exact shape of diaboloidal
    mirror in laboratory coordinate system
    Light Source Beam Line Note LSBL˗1462, Advanced Light Source, Berkeley (May 17, 2020).

    Yashchuk, V. V., Goldberg, K., Lacey, I., McKinney, W. R., Sanchez del Rio, M. & Padmore,H.
    Journal of Synchrotron Radiation, submitted(2021).

    :param p: distance source to mirror [m]
    :param q: distance mirror to focus [m]
    :param theta: grazing incidence angle [rad]
    :param y: x (sagittal) array
    :param y: y (tangential) array
    :return: Z, X, Y
    """
    Z, X, Y = diaboloid_exact_point_to_segment(p=q, q=p, theta=theta, x=x, y=y)
    for i in range(x.size):
        Z[i,:] = numpy.flip(Z[i,:])

    return Z, X, Y