import cv2 

def borderFlag(literal: str) -> int:
    if literal == "Constant":
        return cv2.BORDER_CONSTANT
    elif literal == "Replicate":
        return cv2.BORDER_REPLICATE
    elif literal == "Reflect":
        return cv2.BORDER_REFLECT
    elif literal == "Wrap":
        return cv2.BORDER_WRAP
    elif literal == "Reflect 101":
        return cv2.BORDER_REFLECT_101
    
def interpolationFlag(literal: str) -> int:
    if literal == "Nearest Neighbor":
        return cv2.INTER_NEAREST
    elif literal == "Bilinear":
        return cv2.INTER_LINEAR
    elif literal == "Bicubic":
        return cv2.INTER_CUBIC
    elif literal == "Lanczos":
        return cv2.INTER_LANCZOS4