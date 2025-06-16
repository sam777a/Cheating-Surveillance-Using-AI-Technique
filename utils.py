import numpy as np
import logging

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

def validate_frame(frame, expected_channels=3):
    """Validate that a frame is a valid BGR or RGB image."""
    if frame is None or frame.size == 0:
        logging.error("Invalid frame: None or empty")
        raise ValueError("Invalid frame: None or empty")
    if frame.dtype != 'uint8':
        logging.warning(f"Converting frame from {frame.dtype} to uint8")
        frame = frame.astype('uint8')
    if len(frame.shape) != 3 or frame.shape[2] != expected_channels:
        logging.error(f"Invalid frame: Expected {expected_channels}-channel image, got shape {frame.shape}")
        raise ValueError(f"Invalid frame: Expected {expected_channels}-channel image")
    return frame

def validate_grayscale(gray):
    """Validate that a grayscale image is a valid single-channel uint8 image."""
    if gray is None or gray.size == 0:
        logging.error("Invalid grayscale image: None or empty")
        raise ValueError("Invalid grayscale image: None or empty")
    if gray.dtype != 'uint8':
        logging.warning(f"Converting grayscale from {gray.dtype} to uint8")
        gray = gray.astype('uint8')
    if len(gray.shape) != 2:
        logging.error(f"Invalid grayscale image: Expected single-channel, got shape {gray.shape}")
        raise ValueError("Invalid grayscale image: Expected single-channel")
    if not gray.flags['C_CONTIGUOUS']:
        logging.debug("Making grayscale image contiguous")
        gray = np.ascontiguousarray(gray)
    return gray