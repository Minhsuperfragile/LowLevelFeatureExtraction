import pyfeats

mask = None
param_list = [
{
    'function': pyfeats.fos, #First Order Statistics/Statistical Features (FOS/SF)
    'params': {'mask': mask },
    'features_set': ['features']
},
{
    "function": pyfeats.glcm_features, #Gray Level Co-occurence Matrix (GLCM/SGLDM)
    "params": {'ignore_zeros': True},
    "features_set": ['features_mean', 'features_range']
}, 
{
    "function": pyfeats.glds_features, #Gray Level Difference Statistics (GLDS)
    "params": {'mask': mask, 'Dx': [0,1,1,1], 'Dy': [1,1,0,-1]},
    "features_set": ['features']
},
{
    "function": pyfeats.ngtdm_features, #Neighborhood Gray Tone Difference Matrix (NGTDM)
    "params": {'mask': mask, 'd': 1},
    "features_set": ['features']
},
{
    "function": pyfeats.sfm_features, #Statistical Feature Matrix (SFM)
    'params': {'mask': mask, 'Lr':4, 'Lc':4},
    'features_set': ['features']
},
{
    "function": pyfeats.lte_measures, #Law's Texture Energy Measures (LTE/TEM)
    'params': {'mask': mask, 'l': 7},
    'features_set': ['features']
},
{
    "function": pyfeats.fdta, #Fractal Dimension Texture Analysis (FDTA)
    'params': {'mask': mask, 's':3},
    'features_set': ['features']
},
{
    "function": pyfeats.glrlm_features, #Gray Level Run Length Matrix (GLRLM)
    'params': {'mask': mask, 'Ng': 256},
    'features_set': ['features']
},
{
    "function": pyfeats.fps, #Fourier Power Spectrum (FPS)
    'params': {'mask': mask},
    'features_set': ['features']
},
{
    "function": pyfeats.shape_parameters, #Shape Parameters
    'params': {'mask': mask, 'perimeter': np.ones((128,128)), 'pixels_per_mm2':1},
    'features_set': ['features']
},
{
    "function": pyfeats.glszm_features, #Gray Level Size Zone Matrix (GLSZM)
    'params': {'mask': mask},
    'features_set': ['features']
},
{
    "function": pyfeats.hos_features, #Higher Order Spectra (HOS)
    'params': {'th': [135,140]},
    'features_set': ['features']
},
{
    "function": pyfeats.lbp_features, #Local Binary Pattern (LPB)
    'params': {'mask': mask,  'P':[8,16,24], 'R':[1,2,3]}, 
    'features_set': ['features']
},
{
    "function": pyfeats.grayscale_morphology_features, #Gray-scale Morphological Analysis
    'params': {"N": 30},
    'features_set': ['pdf', 'cdf']
},
# { Deprecated: generate NaN values in the output}
#     "function": pyfeats.multilevel_binary_morphology_features, #Multilevel Binary Morphological Analysis
#     'params': {'mask': mask, 'N': 30, 'thresholds': [25, 50]},
#     'features_set': ['pdf_L', 'pdf_M', 'pdf_H', 'cdf_L', 'cdf_M', 'cdf_H']
# },
# { Deprecated: generate all 0 values in the output
#     'function': pyfeats.histogram, #Histogram
#     'params': {'mask': mask, 'bins': 32},
#     'features_set': ['H']
# },
# { Deprecated: generate all 0 values in the output
#     "function": pyfeats.multiregion_histogram, #Multi-region histogram
#     'params': {'mask': mask, 'bins': 32, 'num_eros': 3, 'square_size': 3},
#     'features_set': ['features']
# },
# { Deprecated: generate all 0 values in the output
#     "function": pyfeats.correlogram, #Correlogram
#     'params': {'mask': mask, 'bins_digitize' : 32, 'bins_hist' : 32, 'flatten' : True},
#     'features_set': ['Hd', 'Ht']
# },
{
    "function": pyfeats.fdta, #Fractal Dimension Texture Analysis (FDTA)
    'params': {'mask': mask, 's': 3},
    'features_set': ['h']
},
{
    "function": pyfeats.amfm_features, #Amplitude Modulation – Frequency Modulation (AM-FM)
    'params': {'bins': 32},
    'features_set': ['features'] # Take long time to calculate
},
# {  Deprecated: generate NaN values in the output
#     "function": pyfeats.dwt_features, #Discrete Wavelet Transform (DWT)
#     'params': {'mask': mask, 'wavelet': 'bior3.3', 'levels': 3},
#     'features_set': ['features']
# },
{
    "function": pyfeats.swt_features, #Stationary Wavelet Transform (SWT)
    'params': {'mask': mask, 'wavelet': 'bior3.3', 'levels': 3},
    'features_set': ['features']
},
{
    "function": pyfeats.wp_features, #Wavelet Packets (WP)
    'params': {'mask': mask, 'wavelet': 'coif1', 'maxlevel': 3},
    'features_set': ['features']
},
{
    "function": pyfeats.gt_features, #Gabor Transform (GT)
    'params': {'mask': mask, 'deg': 4, 'freq': [0.05, 0.4]},
    'features_set': ['features']
},
{
    "function": pyfeats.zernikes_moments, #Zernikes’ Moments
    'params': {'radius': 9},
    'features_set': ['features']
},
{
    "function": pyfeats.hu_moments, #Hu’s Moments
    'params': {},
    'features_set': ['features']
},
{
    "function": pyfeats.tas_features, #Threshold Adjacency Matrix (TAS)
    'params': {},
    'features_set': ['features']
},
{
    "function": pyfeats.hog_features, #Histogram of Oriented Gradients (HOG)
    'params': {'ppc': 8, 'cpb': 3},
    'features_set': ['features']
}
]