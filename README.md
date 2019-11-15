# Multi-Image Super-resolution
Convert multiple Low resolution(LR) images to single High resolution images image.

Steps
<ol>
<li> Registration: To find a affine transformation from one image to other so, as to register them on HR grid</li>
<li> Non-uniform Interpolation: After registering images on a single HR grid, some pixels might not be filled, as the transformation doesn't gurantee to cover all possible locations</li>
<li> Smoothening: The resultant image could be noisy, due to non-linear relationships between images, and might lead to noisy image. Smoothing improves visual aspects of image (needs to be implemented)</li>
</ol>
