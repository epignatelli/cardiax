'''
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
'''

from scipy.signal import convolve2d as conv2d
import skimage
from skimage.io import imsave
import numpy as np
from scipy import interpolate
import shortuuid
import uuid  
import json

def_params = {}
def_params['maxProtrudeFactorCentroid'] = .2
def_params['r0Centroid'] = 1.5
def_params['maxProtrudeFactor'] = 0.2
def_params['r0'] = 1
def_params['ScarScaleFactor'] = .4
def_params['RequiredImageSize'] = (1200, 1200)
def_params['CONSERVATIVE_CENTROID'] = True
def_params['GaussSigma'] = 2
def_params['RequiredAvgEdgeSize'] = 22 # in pixels
def_params['ADD_GAP'] = False
def_params['GapFactor'] = .5

def_shortID = 'BJdY4KfkSDsa4wPe7i3HPb'

def_root_file_name = 'data/scars_maps/{content}_{ID}.{ext}'

def makegauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def drawpolyintoemptycanvas(CS,x,y,tx,ty):
    """
    Purpose is to draw a polygon into an empty canvas (image)
    for the purpose of subseuqntly combining it with an existing canvas
    The offsets can be chosen so that the polygon does not exceed
    the boundaries of the polygon
    """
    from skimage.draw import polygon
    img = np.zeros(CS, dtype=np.float)

    R = CS[1]-(ty+y)
    C = (tx+x)
    if np.any(R<0) or np.any(R>CS[1]):
        raise ValueError('Polygon row (y) coordinates out of lower/upper bound')

    if np.any(C<0) or np.any(C>CS[0]):
        raise ValueError('Polygon column (x) coordinates out of lower/upper bound')

    rr, cc = polygon(R, C)
    img[rr, cc] = 1
    
    return img

def addnoise(x,swn=0.1,ssn=0.1,gks=0.5,rectify=False):
    """
    Adds some white and structured noise; wn is non-Gaussian
    Usage: addnoise(x,swn,ssn,rectify)
    swn: white noise variance (it is zero mean by default)
    ssn: coloured noise, filtered by a 3x3 Gaussian kernel, var:0.25 (default)
         kernel size is scaled according to spatial sd of Gaussian kernel
    gks: Gaussian kernel sigma for the correlated noise
    rectify: Boolean - says whether or not image is rectified to remove negative values
    """
    m,n = x.shape
    wn = swn*np.random.randn(m,n)
    gksize = (6*gks,6*gks)
    g2 = makegauss2D(shape=gksize,sigma=gks)
    sn = conv2d(ssn*np.random.randn(m,n),g2,'same')
    
    y = x + wn + sn
    
    if rectify:
        y = y*(y>0)
    
    return(y)
    
def makePolyAndSplineCurve(r0=30, maxProtrudeFactor=0.3, NGon=11, NSplinePts=100):
   
    maxPerturbMag = maxProtrudeFactor*r0;

    delta_t = 1/(NGon-1);
    t = np.arange(0, 1+delta_t, delta_t)
    
    # obtain perturbation to the radius based on a slightly shifted and scaled gaussian
    # this is to skew the values towards the positive range
    rperturbation = np.random.normal(loc=0.5, scale = 0.6, size = NGon) #random.randn(NGon)
    # 
    rdash = r0+maxPerturbMag*rperturbation

    Px = rdash*np.sin(2*np.pi*t)
    Py = rdash*np.cos(2*np.pi*t)

    # Close the curve
    Px[-1] = Px[0]
    Py[-1] = Py[0]

    delta_u = 1/NSplinePts;
    tck, u = interpolate.splprep([Px, Py], s=0)
    unew = np.arange(0, 1+delta_u, delta_u)
    SplinePts = interpolate.splev(unew, tck)

    return (Px, Py), SplinePts

def PolyAndSplineCurve2Mask(SplinePts, CxSize = 100, CySize = 100, tx = None, ty = None):
    ''' Transform a Spline Curve defined by its points into a binary? image with size (CxSize, CySize) and
    with 1 inside the curve and 0 outside '''
    if tx is None:
        tx = CxSize/2
    if ty is None:
        ty = CySize/2
    
    ## create image from orginal points - not needed here
    #MapMaskPoly=drawpolyintoemptycanvas((CxSize,CySize),P[0],P[1],tx,ty)
    # create image from spline points
    MapMaskSp= drawpolyintoemptycanvas((CxSize,CySize),SplinePts[0],SplinePts[1],tx,ty)

    return MapMaskSp

def GetAvgEdgeSize(SoftenedMask):
    ''' Get average number of pixels on each row that are between 0 and 1'''
    edge_size_sum = []
    edge_size_mean = []
    for i in range(SoftenedMask.shape[0]):
        row_edge_sum = ((SoftenedMask[i]>0) & (SoftenedMask[i]<.99)).sum()
        row_edge_mean = ((SoftenedMask[i]>0) & (SoftenedMask[i]<.99)).mean()

        if row_edge_sum>0:
            edge_size_sum.append(row_edge_sum)
            edge_size_mean.append(row_edge_mean)
    ''' edge_size_sum is in PIXELS, edge_size_mean is in PROPORTION OF IMAGE SIZE'''
    # take average across rows. Divide by 2 because there are usually two edges
    avg_edge_size_pixel = np.mean(edge_size_sum)/2
    avg_edge_size_prop = np.mean(edge_size_mean)/2
    return avg_edge_size_pixel, avg_edge_size_prop

def SoftenPolyAndSplineCurve(MapMaskSp, GaussShape = 4, GaussSigma= 2, 
    AvgEdgeSize = None, verbose = False):
    ''' Taper the edge of an image of a spline curve generated by PolyAndSplineCurve2Image. 
    AvgEdgeSize can be specified either in pixels (>1 values) or in proportions (<1 values) '''
    
    def ConvolveAndAvgEdge(LocalGaussShape, LocalGaussSigma):
        # create 2D filter, convolve and get avg edge size
        g2_a = makegauss2D(shape=(LocalGaussShape, LocalGaussShape), sigma= LocalGaussSigma)
        LocalSoftenedBlob = conv2d(MapMaskSp,g2_a,'same')
        local_avg_edge_size_pixel, local_avg_edge_size_prop = GetAvgEdgeSize(LocalSoftenedBlob)
        return LocalSoftenedBlob, local_avg_edge_size_pixel, local_avg_edge_size_prop  
    
    
    assert(GaussShape or AvgEdgeSize)
    
    if GaussShape:
        ''' used a fixed size for the Gaussian filter. 
        If both GaussShape and AvgEdgeSize are specified, the former gets priority '''
        SoftenedBlob, avg_edge_size_pixel, avg_edge_size_prop = ConvolveAndAvgEdge(
                GaussShape, GaussSigma)
        if verbose:
            print(f'Gaussian filter shape: {GaussShape}')
            print(f'Average edge size: {avg_edge_size_pixel:.2f} (in absolute pixels)')
            print(f'Average edge size: {avg_edge_size_prop:.2f} (in relative pixels)')
        
    else:
        ''' define the size of the Gaussian filter based on the required avg edge size'''
        # decide whether we are dealing with pixels or 
        IS_PIXELS = AvgEdgeSize>1
        # do multiple convolutions and get avg edge size
        
        edge_size_diff = []
        CandidateGaussShapes = np.arange(4,22)
        for iGaussShape in CandidateGaussShapes:
            iSoftenedBlob, iavg_edge_size_pixel, iavg_edge_size_prop = ConvolveAndAvgEdge(
                iGaussShape, GaussSigma)
            iavg_edge_size = iavg_edge_size_pixel if IS_PIXELS else iavg_edge_size_prop
            edge_size_diff.append(np.abs(iavg_edge_size - AvgEdgeSize))
        
        # choose the gaussian size that gives the best results 
        GaussShape = CandidateGaussShapes[np.argmin(edge_size_diff)]
        # recompute softened image
        SoftenedBlob, avg_edge_size_pixel, avg_edge_size_prop = ConvolveAndAvgEdge(
                GaussShape, GaussSigma)
        if verbose:
            print(f'Gaussian filter shape: {GaussShape}')
            print(f'Average edge size: {avg_edge_size_pixel:.2f} (in absolute pixels)')
            print(f'Average edge size: {avg_edge_size_prop:.2f} (in relative pixels)')
        
    # return all of the results    
    return SoftenedBlob, avg_edge_size_pixel, avg_edge_size_prop, GaussShape

# Create function to determine centerline for blobs
def CreateSplineCentroids(params = def_params):#r0 =1.2 , maxProtrudeFactor = 0.4):
    ''' generate centroids for all closed spline objects as a section of 
    another spline curve'''
    
    r0 = params['r0Centroid']
    maxProtrudeFactor = params['maxProtrudeFactorCentroid']
    
    if r0 is None:
        r0 = 1
    if maxProtrudeFactor is None:
        maxProtrudeFactor = 0.2
        
    P, _ = makePolyAndSplineCurve(r0= r0, maxProtrudeFactor = maxProtrudeFactor, NGon=12)
    Npoints = np.random.randint(3,7)
    starting_point = np.random.randint(P[0].shape)
    select_points = np.arange(starting_point,starting_point+Npoints) % P[0].shape
    return (P[0][select_points], P[1][select_points])

def MakeAndSumCompositeBlob(params = def_params, CentroidSpline = None):
    ''' 
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
    
    '''
    if CetroidSpline is None:
        CentroidSpline = CreateSplineCentroids(params = params)
        
    # recreate parameters as individual variables
    maxProtrudeFactorCentroid = params['maxProtrudeFactorCentroid']
    r0Centroid = params['r0Centroid']
    maxProtrudeFactor = params['maxProtrudeFactor']
    r0 = params['r0']
    ScarScaleFactor = params['ScarScaleFactor']
    RequiredImageSize = params['RequiredImageSize']
    CONSERVATIVE_CENTROID = params['CONSERVATIVE_CENTROID']
    GaussSigma = params['GaussSigma']
    RequiredAvgEdgeSize = params['RequiredAvgEdgeSize']
    ADD_GAP = params['ADD_GAP']
    GapFactor = params['GapFactor']

    # calculate some extra factors
    nBlobs = CentroidSpline[0].shape[0]
    maxCentroidExtension = r0Centroid + maxProtrudeFactorCentroid
    maxExtension = maxCentroidExtension + maxProtrudeFactor + r0
    doubleExtension = maxExtension*2
    halfExtension = maxExtension/2
    ScarBaseImages = []
    ScarBasePtsNorm = []
    ScarBasePts = []
    
    for i in range(nBlobs):
        # create and store 'normalised' points
        _, SplinePts = makePolyAndSplineCurve(r0 = r0, maxProtrudeFactor = maxProtrudeFactor)
        ScarBasePtsNorm.append(SplinePts)
        
        # center the blobs around the ith point in the centroid spline
        for j in range(2):
            SplinePts[j] += CentroidSpline[j][i]
            #P[j] += CentroidSpline[j][i]
        ScarBasePts.append(SplinePts)
        # if we already know we want to add a central gap, we should create a second, smaller, blob at the mid centroid
        if ADD_GAP & (i == nBlobs//2):
            _, GapPts = makePolyAndSplineCurve(r0 = r0*GapFactor, maxProtrudeFactor = maxProtrudeFactor*GapFactor)
            for j in range(2):
                GapPts[j] += CentroidSpline[j][i]
        else:
            GapPts = 0
            GapMask = 0
    
    # set the overall location in the image
    maxField = np.ceil(maxExtension/ScarScaleFactor)
    RequiredImageRatio = RequiredImageSize[0]/RequiredImageSize[1]
    if RequiredImageRatio == 1:
        maxFieldX, maxFieldY = maxField, maxField
    elif RequiredImageRatio>1:
        # wider image
        maxFieldX, maxFieldY = np.ceil(RequiredImageRatio)*maxField, maxField
    else:
        # 'taller' image
        maxFieldX, maxFieldY = maxField, np.ceil(RequiredImageRatio)*maxField


    if CONSERVATIVE_CENTROID:
        GlobalCentroid = ((maxFieldX-doubleExtension)*np.random.rand() + maxExtension, 
                          (maxFieldY-doubleExtension)*np.random.rand() + maxExtension)
    else:
        GlobalCentroid = ((maxFieldX-maxExtension)*np.random.rand() + halfExtension, 
                          (maxFieldY-maxExtension)*np.random.rand() + halfExtension)
    #print(GlobalCentroid)

    if (maxFieldX>RequiredImageSize[0]) | (maxFieldY>RequiredImageSize[1]):
        raise Exception('Scaling down not implemented yet!') #TODO!
        
    else:
        FieldImageRatio = (maxFieldX/RequiredImageSize[0], maxFieldY/RequiredImageSize[1])
        for i in range(nBlobs):
            for j in range(2):
                ScarBasePts[i][j] += GlobalCentroid[j]
                ScarBasePts[i][j] /= FieldImageRatio[j]
                ScarBasePts[i][j] = ScarBasePts[i][j]*(ScarBasePts[i][j]>0)
                #P[j] += GlobalCentroid[j]
                #P[j] *= ImageFieldRatio[j]
        if ADD_GAP:
            for j in range(2):
                GapPts[j] += GlobalCentroid[j]
                GapPts[j] /= FieldImageRatio[j]
                GapPts[j] = GapPts[j]*(GapPts[j]>0)

    # create images

    for i in range(nBlobs):
        SplineMask = PolyAndSplineCurve2Mask(ScarBasePts[i], CxSize = RequiredImageSize[0], 
                                             CySize = RequiredImageSize[1],
                                             tx = 0, ty = 0) #GlobalCentroid[j])
        ScarBaseImages.append(SplineMask)
        # sum them all up as an OR operation
        if i == 0:
            CompositeSplineMask = SplineMask.copy()
        else:
            CompositeSplineMask += SplineMask
            CompositeSplineMask[CompositeSplineMask>1.] = 1.

    if ADD_GAP:
        GapMask = PolyAndSplineCurve2Mask(GapPts, CxSize = RequiredImageSize[0], 
                                             CySize = RequiredImageSize[1],
                                             tx = 0, ty = 0)
        CompositeSplineMask[GapMask==1.] = 0
    
    # make sure the final image is between 0 and 1
    CompositeSplineMask = (CompositeSplineMask - CompositeSplineMask.min()
                                           )/(CompositeSplineMask.max() - CompositeSplineMask.min())
                                           
    res_dict = {}
    res_dict['CompositeSplineMask'] = CompositeSplineMask
    res_dict['ScarBaseImages'] = ScarBaseImages
    res_dict['ScarBasePts'] = ScarBasePts
    res_dict['ScarBasePtsNorm'] = ScarBasePtsNorm
    res_dict['GapPts'] = GapPts
    res_dict['GapMask'] = GapMask
    res_dict['GlobalCentroid'] = GlobalCentroid
    res_dict['CentroidSpline'] = CentroidSpline
    res_dict['maxFieldX'] = maxFieldX
    res_dict['maxFieldY'] = maxFieldY
    res_dict['nBlobs'] = nBlobs
    
    return res_dict

def save_scar_as_array(SoftenedComposite, params = def_params, root_file_name = def_root_file_name):
    #save results with parameter set and image    
    for content, ext in zip(['params','output_scar','scar_image'],['json','npy','png']):
        formatted_file = root_file_name.format(content = content, ID = shortu, ext = ext)
        if ext == 'npy':
            np.save(formatted_file, SoftenedComposite)
        elif ext == 'json':
            with open(formatted_file, 'w') as f:
                json.dump(params, f)
        elif ext == 'png':
            imsave(formatted_file, np.round(SoftenedComposite*255).astype(np.uint8),
                check_contrast = False)
                
def load_scar_as_array(shortID = def_shortID, root_file_name = def_root_file_name):
    SoftenedComposite = np.load(root_file_name.format(content = 'output_scar', ID = shortID, ext ='npy'),
                                allow_pickle=False)
    return SoftenedComposite
    

