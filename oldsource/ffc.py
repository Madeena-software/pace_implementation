import cupy as cp
from cupyx.scipy.ndimage import median_filter
import gc
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def ffcimg(image_projection,image_gain,image_dark):
    proj = cp.asarray(image_projection)
    proj = (proj - cp.min(proj)) / (65535 - cp.min(proj))
    proj = median_filter(proj, 7)

    gain = cp.asarray(image_gain)
    gain = (gain - cp.min(gain)) / (65535 - cp.min(gain))
    gain = median_filter(gain, 7)

    dark = cp.asarray(image_dark)
    dark = (dark - cp.min(dark)) / (65535 - cp.min(dark))
    dark = median_filter(dark, 7)

    proj_dark = cp.subtract(proj, dark)
    gain_dark = cp.subtract(gain, dark)

    proj_dark[proj_dark <= 0] = 1e-12
    gain_dark[gain_dark <= 0] = 0
    
    intensity = cp.divide(gain_dark, proj_dark)
    intensity[intensity <= 0] = 1e-12
    
    miu = cp.log(intensity)
    miu = miu.astype(cp.float32)
    miu[miu < 0] = 0
    
    
    del proj, gain, dark
    del proj_dark, gain_dark, intensity
    gc.collect()

    # hasil = miu

    # saturation_percent = 0.01
    # max_val = cp.percentile(hasil, 100 - saturation_percent)
    # image_clipped = cp.clip(hasil, 0, max_val)

    # hasil = (image_clipped - cp.min(image_clipped)) / (cp.max(image_clipped) - cp.min(image_clipped))
    # corrected_image = (hasil * 65535).astype(cp.uint16)

    return miu
