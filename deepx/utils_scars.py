"""
utils_scars.py by Stef Garasto

Description.

Utils to help generate a scar map consisting of multiple overlapping polygons created by
using a closed spline curve as their contours (with the option to add a gap in the middle).
By varying the parameters a wide range of scar maps can be generated.

Specifics:

First, the centroids of the different objects making up the scar are created by selecting a
subset of adjacent points from a pertured spline curve (see CreateSplineCentroids).
The number of selected points gives the number of closed splines objects to generate and add together.

After it generates this number of closed spline curves (randomly perturbed), it converts them
to binary images and sums them together as an OR operation (see MakeAndSumCompositeBlob).
Individual Splines object are shifted based on their individual centroid and a global centroid.
The global centroid defines where they are in the image, the individual centroids define where
they are wrt each other. Aside from being shifted, they are also scaled to fit within a larger image.
The scale ratio is given by specifying how the big one wants the image to be and what the scale of
the scar wrt to the full image should be.

The scale factor is basically the scale factor between the width/height of the image and the shorter
side of the composite scar (well, at least it is roughly that, I think - this part is done a bit in
an approximate way).

Then, the final image is smoothed to create a tapered edge (see SoftenPolyAndSplineCurve).
To do this, the function requires a given average size (AvgEdgeSize) for the edges coming
out of the convolution. The function will do its best to find the gaussian filter that gets
us as close as possible to the required size (within a certain range). AvgEdgeSize can be
specified either in pixels (>1 values) or in proportions (<1 values). The alternative would
be to convolve with a fixed sixe Gaussian (also implemented), but this doesn't control for the
size of the resulting edge.
"""

import logging
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from scipy import interpolate
from skimage.draw import polygon

def_params = {}
def_params["maxProtrudeFactorCentroid"] = 0.2
def_params["r0Centroid"] = 1.5
def_params["maxProtrudeFactor"] = 0.2
def_params["r0"] = 1
def_params["ScarScaleFactor"] = 0.4
def_params["RequiredImageSize"] = (1200, 1200)
def_params["CONSERVATIVE_CENTROID"] = True
def_params["GaussSigma"] = 2
def_params["RequiredAvgEdgeSize"] = 22  # in pixels
def_params["ADD_GAP"] = False
def_params["GapFactor"] = 0.5

def_shortID = "BJdY4KfkSDsa4wPe7i3HPb"

def_root_file_name = "data/scars_maps/{content}_{ID}.{ext}"


def makegauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = jnp.meshgrid(jnp.arange(-m, m + 1), jnp.arange(-n, n + 1))
    h = jnp.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h = h.at[h < jnp.finfo(h.dtype).eps * h.max()].set(0)
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def drawpolyintoemptycanvas(shape, x, y, tx, ty):
    """
    Purpose is to draw a polygon into an empty canvas (image)
    for the purpose of subseuqntly combining it with an existing canvas
    The offsets can be chosen so that the polygon does not exceed
    the boundaries of the polygon
    """
    img = jnp.zeros(shape, dtype=float)

    R = shape[1] - (ty + y)
    C = tx + x

    R = jnp.clip(R, 0, shape[1])
    C = jnp.clip(C, 0, shape[0])

    rr, cc = polygon(R, C)
    img = img.at[rr, cc].set(1)

    return jnp.array(img)


def makePolyAndSplineCurve(rng, r0=30, maxProtrudeFactor=0.3, NGon=11, NSplinePts=100):
    rng_1, _ = jax.random.split(rng)
    maxPerturbMag = maxProtrudeFactor * r0

    delta_t = 1 / (NGon - 1)
    t = jnp.arange(0, 1 + delta_t, delta_t)

    # obtain perturbation to the radius based on a slightly shifted and scaled gaussian
    # this is to skew the values towards the positive range
    rperturbation = (jax.random.normal(rng_1, (NGon,)) + 0.5) * jnp.power(0.6, 2)

    rdash = r0 + maxPerturbMag * rperturbation

    Px = rdash * jnp.sin(2 * jnp.pi * t)
    Py = rdash * jnp.cos(2 * jnp.pi * t)

    # Close the curve
    Px = Px.at[-1].set(Px[0])
    Py = Py.at[-1].set(Py[0])

    delta_u = 1 / NSplinePts
    tck, u = interpolate.splprep([Px, Py], s=0)
    unew = jnp.arange(0, 1 + delta_u, delta_u)
    SplinePts = interpolate.splev(unew, tck)

    return (Px, Py), SplinePts


def PolyAndSplineCurve2Mask(SplinePts, CxSize=100, CySize=100, tx=None, ty=None):
    """Transform a Spline Curve defined by its points into a binary? image with size (CxSize, CySize) and
    with 1 inside the curve and 0 outside"""
    if tx is None:
        tx = CxSize / 2
    if ty is None:
        ty = CySize / 2

    # create image from spline points
    MapMaskSp = drawpolyintoemptycanvas(
        (CxSize, CySize), SplinePts[0], SplinePts[1], tx, ty
    )

    return MapMaskSp


def GetAvgEdgeSize(softened_mask):
    """ Get average number of pixels on each row that are between 0 and 1"""
    edge_size_sum = []
    edge_size_mean = []
    for i in range(softened_mask.shape[0]):
        row_edge_sum = ((softened_mask[i] > 0) & (softened_mask[i] < 0.99)).sum()
        row_edge_mean = ((softened_mask[i] > 0) & (softened_mask[i] < 0.99)).mean()

        if row_edge_sum > 0:
            edge_size_sum.append(row_edge_sum)
            edge_size_mean.append(row_edge_mean)
    """ edge_size_sum is in PIXELS, edge_size_mean is in PROPORTION OF IMAGE SIZE"""
    # take average across rows. Divide by 2 because there are usually two edges
    avg_edge_size_pixel = jnp.mean(edge_size_sum) / 2
    avg_edge_size_prop = jnp.mean(edge_size_mean) / 2
    return avg_edge_size_pixel, avg_edge_size_prop


def SoftenPolyAndSplineCurve(
    MapMaskSp, GaussShape=4, GaussSigma=2, AvgEdgeSize=None, verbose=False
):
    """Taper the edge of an image of a spline curve generated by PolyAndSplineCurve2Image.
    AvgEdgeSize can be specified either in pixels (>1 values) or in proportions (<1 values)"""

    def ConvolveAndAvgEdge(n, sigma):
        # create 2D filter, convolve and get avg edge size
        g2_a = makegauss2D(shape=(n, n), sigma=sigma)

        LocalSoftenedBlob = convolve2d(MapMaskSp, g2_a, "same")
        local_avg_edge_size_pixel, local_avg_edge_size_prop = GetAvgEdgeSize(
            LocalSoftenedBlob
        )
        return LocalSoftenedBlob, local_avg_edge_size_pixel, local_avg_edge_size_prop

    assert GaussShape or AvgEdgeSize

    if GaussShape:
        """used a fixed size for the Gaussian filter.
        If both GaussShape and AvgEdgeSize are specified, the former gets priority"""
        SoftenedBlob, avg_edge_size_pixel, avg_edge_size_prop = ConvolveAndAvgEdge(
            GaussShape, GaussSigma
        )
        if verbose:
            print(f"Gaussian filter shape: {GaussShape}")
            print(f"Average edge size: {avg_edge_size_pixel:.2f} (in absolute pixels)")
            print(f"Average edge size: {avg_edge_size_prop:.2f} (in relative pixels)")

    else:
        """ define the size of the Gaussian filter based on the required avg edge size"""
        # decide whether we are dealing with pixels or
        IS_PIXELS = AvgEdgeSize > 1
        # do multiple convolutions and get avg edge size

        edge_size_diff = []
        CandidateGaussShapes = jnp.arange(4, 22)
        for iGaussShape in CandidateGaussShapes:
            (
                _,
                iavg_edge_size_pixel,
                iavg_edge_size_prop,
            ) = ConvolveAndAvgEdge(iGaussShape, GaussSigma)
            iavg_edge_size = iavg_edge_size_pixel if IS_PIXELS else iavg_edge_size_prop
            edge_size_diff.append(jnp.abs(iavg_edge_size - AvgEdgeSize))

        # choose the gaussian size that gives the best results
        GaussShape = CandidateGaussShapes[jnp.argmin(edge_size_diff)]
        # recompute softened image
        SoftenedBlob, avg_edge_size_pixel, avg_edge_size_prop = ConvolveAndAvgEdge(
            GaussShape, GaussSigma
        )
        if verbose:
            print(f"Gaussian filter shape: {GaussShape}")
            print(f"Average edge size: {avg_edge_size_pixel:.2f} (in absolute pixels)")
            print(f"Average edge size: {avg_edge_size_prop:.2f} (in relative pixels)")

    # return all of the results
    return SoftenedBlob, avg_edge_size_pixel, avg_edge_size_prop, GaussShape


# Create function to determine centerline for blobs
def CreateSplineCentroids(rng, params=def_params):
    """generate centroids for all closed spline objects as a section of
    another spline curve"""
    rng_1, rng_2, rng_3 = jax.random.split(rng, 3)
    r0 = params["r0Centroid"]
    maxProtrudeFactor = params["maxProtrudeFactorCentroid"]

    if r0 is None:
        r0 = 1
    if maxProtrudeFactor is None:
        maxProtrudeFactor = 0.2

    P, _ = makePolyAndSplineCurve(
        rng_1, r0=r0, maxProtrudeFactor=maxProtrudeFactor, NGon=12
    )
    Npoints = jax.random.randint(rng_2, (1,), 3, 7)
    m = len(P[0])
    starting_point = jax.random.randint(rng_3, (1,), m / 4, m / 4 * 3)
    select_points = jnp.arange(starting_point, starting_point + Npoints) % len(P[0])
    return (P[0][select_points], P[1][select_points])


def MakeAndSumCompositeBlob(rng, params=def_params, CentroidSpline=None):
    """
    Generates a number of closed splines objects based on how many points
    there are in the CentroidSpline input.
    Converts them to binary images and sums them together as an OR operation.
    Individual Splines object are shifted based on their individual centroid
    and a global centroid. The global centroid defines where they are in the
    image, the individual centroids define where they are wrt each other.
    Aside from being shifted, they are also scaled to fit within a larger image.
    The scale ratio is given by specifying how the big one wants the image to be
    and what the scale of the scar wrt to the full image should be.
    The scale factor is basically the scale factor between the width/height of the
    image and the shorter side of the composite scar (well, at least it is roughly
    that, I think - this part is done a bit in an approximate way).

    params is a dict with the variables outlined below.

    CentroidSpline is a list of centroids returned by CreateSplineCentroids

    """
    # make rngs
    rng_1, rng_2, rng_3, rng_4 = jax.random.split(rng, 4)

    if CentroidSpline is None:
        CentroidSpline = CreateSplineCentroids(rng_1, params=params)

    # recreate parameters as individual variables
    maxProtrudeFactorCentroid = params["maxProtrudeFactorCentroid"]
    r0Centroid = params["r0Centroid"]
    maxProtrudeFactor = params["maxProtrudeFactor"]
    r0 = params["r0"]
    ScarScaleFactor = params["ScarScaleFactor"]
    RequiredImageSize = shape = params["RequiredImageSize"]
    CONSERVATIVE_CENTROID = params["CONSERVATIVE_CENTROID"]
    ADD_GAP = params["ADD_GAP"]
    GapFactor = params["GapFactor"]

    # calculate some extra factors
    nBlobs = CentroidSpline[0].shape[0]
    maxCentroidExtension = r0Centroid + maxProtrudeFactorCentroid
    maxExtension = maxCentroidExtension + maxProtrudeFactor + r0
    doubleExtension = maxExtension * 2
    halfExtension = maxExtension / 2
    ScarBaseImages = []
    ScarBasePtsNorm = []
    ScarBasePts = []

    for i in range(nBlobs):
        rng_2, rng_3 = jax.random.split(rng_2)
        # create and store 'normalised' points
        _, SplinePts = makePolyAndSplineCurve(
            rng_2, r0=r0, maxProtrudeFactor=maxProtrudeFactor
        )
        ScarBasePtsNorm.append(SplinePts)

        # center the blobs around the ith point in the centroid spline
        for j in range(2):
            SplinePts[j] += CentroidSpline[j][i]
            # P[j] += CentroidSpline[j][i]
        ScarBasePts.append(SplinePts)
        # if we already know we want to add a central gap, we should create a second, smaller, blob at the mid centroid
        if ADD_GAP & (i == nBlobs // 2):
            _, GapPts = makePolyAndSplineCurve(
                rng_3,
                r0=r0 * GapFactor,
                maxProtrudeFactor=maxProtrudeFactor * GapFactor,
            )
            for j in range(2):
                GapPts[j] += CentroidSpline[j][i]
        else:
            GapPts = 0
            GapMask = 0

    # set the overall location in the image
    maxField = jnp.ceil(maxExtension / ScarScaleFactor)
    RequiredImageRatio = RequiredImageSize[0] / RequiredImageSize[1]
    if RequiredImageRatio == 1:
        maxFieldX, maxFieldY = maxField, maxField
    elif RequiredImageRatio > 1:
        # wider image
        maxFieldX, maxFieldY = jnp.ceil(RequiredImageRatio) * maxField, maxField
    else:
        # 'taller' image
        maxFieldX, maxFieldY = maxField, jnp.ceil(RequiredImageRatio) * maxField

    rng_41, rng_42 = jax.random.split(rng_4)
    x = jax.random.normal(rng_41)
    y = jax.random.normal(rng_42)
    if CONSERVATIVE_CENTROID:
        GlobalCentroid = (
            (maxFieldX - doubleExtension) * x + maxExtension,
            (maxFieldY - doubleExtension) * y + maxExtension,
        )
    else:
        GlobalCentroid = (
            (maxFieldX - maxExtension) * x + halfExtension,
            (maxFieldY - maxExtension) * y + halfExtension,
        )

    FieldImageRatio = (
        maxFieldX / RequiredImageSize[0],
        maxFieldY / RequiredImageSize[1],
    )
    for i in range(nBlobs):
        for j in range(2):
            ScarBasePts[i][j] += GlobalCentroid[j]
            ScarBasePts[i][j] /= FieldImageRatio[j]
            ScarBasePts[i][j] = ScarBasePts[i][j] * (ScarBasePts[i][j] > 0)
    if ADD_GAP:
        for j in range(2):
            GapPts[j] += GlobalCentroid[j]
            GapPts[j] /= FieldImageRatio[j]
            GapPts[j] = GapPts[j] * (GapPts[j] > 0)

    # create images
    scar = jnp.zeros(shape)
    for i in range(nBlobs):
        x = PolyAndSplineCurve2Mask(
            ScarBasePts[i], CxSize=shape[0], CySize=shape[1], tx=0, ty=0
        )
        scar += x
    scar = jnp.clip(scar, a_max=1.0)

    if ADD_GAP:
        GapMask = PolyAndSplineCurve2Mask(
            GapPts, CxSize=RequiredImageSize[0], CySize=RequiredImageSize[1], tx=0, ty=0
        )
        scar = jnp.where(GapMask == 1.0, GapMask)

    # make sure the final image is between 0 and 1
    if scar.sum() > 0:
        scar = (scar - scar.min()) / (scar.max() - scar.min())
    else:
        scar = jnp.zeros(shape)

    res_dict = {}
    res_dict["CompositeSplineMask"] = scar
    res_dict["ScarBaseImages"] = ScarBaseImages
    res_dict["ScarBasePts"] = ScarBasePts
    res_dict["ScarBasePtsNorm"] = ScarBasePtsNorm
    res_dict["GapPts"] = GapPts
    res_dict["GapMask"] = GapMask
    res_dict["GlobalCentroid"] = GlobalCentroid
    res_dict["CentroidSpline"] = CentroidSpline
    res_dict["maxFieldX"] = maxFieldX
    res_dict["maxFieldY"] = maxFieldY
    res_dict["nBlobs"] = nBlobs

    return res_dict


def random_spline(rng, params, centroids):
    return MakeAndSumCompositeBlob(rng, params, centroids)["CompositeSplineMask"]


def blur_scar(scar, gaussian_size, sigma):
    # create 2D filter, convolve and get avg edge size
    kernel = makegauss2D(shape=(gaussian_size, gaussian_size), sigma=sigma)
    blurred_scar = convolve2d(scar, kernel, "same")
    return blurred_scar


def random_diffusivity_scar(rng, shape: Tuple[int, ...]):
    rng_1, rng_2, rng_3, rng_4 = jax.random.split(rng, 4)
    params = def_params
    # replace size
    params["RequiredImageSize"] = shape
    # params["ScarScaleFactor"] = jax.random.randint(rng_3, (1,), 1, 8) / 10
    centroids = CreateSplineCentroids(rng_1, params)

    # Create individual blobs, scale them up and combine them
    scar = random_spline(rng_2, params, centroids)

    # taper the edges
    blur_kerne_size = int(shape[0] * 0.1)  #  kernel size is 3% of the image size
    blur_sigma = jax.random.normal(rng_4, (1,)) * shape[0] / 10
    scar = blur_scar(scar, blur_kerne_size, blur_sigma)
    return jnp.array(1 - scar)


if __name__ == "__main__":
    random_diffusivity_scar(jax.random.PRNGKey(4), (256, 256))