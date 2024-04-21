# given an image with W x H, return the index of pixels in the hibert curve
# the index is from 0 to W*H-1
# the returned index is a 1D array
#default the parttern is like ]

def xy_to_hilbert_index(x, y, W, H):
    def xy_to_d(x, y):
        d = 0
        s = min(W, H) // 2
        while s > 0:
            rx = 1 if x & s else 0
            ry = 1 if y & s else 0
            d += s * s * ((3 * rx) ^ ry)
            x, y = rotate(s, x, y, rx, ry)
            s //= 2
        return d

    def rotate(n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y

    return xy_to_d(x, y)

def hibert_index_to_xy(d, W, H):
    def d_to_xy(d):
        t = d
        x = y = 0
        s = 1
        while s < W or s < H:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = rotate(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    def rotate(n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y

    return d_to_xy(d)

# Example usage:
# W = 3  # Width of the image
# H = 3  # Height of the image
# # hibert_index = [hibert_index_to_xy(index, W, H) for index in range(W*H)]
# # print(hibert_index)

# for y in range(H):
#     for x in range(W):
#         index = xy_to_hilbert_index(x, y, W, H)
#         print(f"Pixel ({x}, {y}) has Hilbert index {index}")
# # print('--')
# # for index in range(W*H):
# #     x, y = hibert_index_to_xy(index, W, H)
# #     print(f"Index {index} has pixel ({x}, {y})")



# import numpy as np
# from hilbertcurve.hilbertcurve import HilbertCurve

# def hilbert_pixel_indices(width, height):
#     """
#     Generates pixel indices in the order of the Hilbert curve.

#     Args:
#         width (int): Width of the image.
#         height (int): Height of the image.

#     Returns:
#         numpy.ndarray: A 2D array where each row represents a pixel's 
#                        (x, y) coordinates.
#     """

#     p = np.max([np.ceil(np.log2(width)), np.ceil(np.log2(height))])  # Order of curve 
#     N = 2**p  # Length of the side of the square the curve fills

#     hilbert_curve = HilbertCurve(p, 2)  # 2 dimensions

#     coords = np.array([
#         hilbert_curve.coordinates_from_distance(d) 
#         for d in range(width * height)
#     ])

#     return coords

# # # Example usage
# # width = 32
# # height = 32
# # indices = hilbert_pixel_indices(width, height)

# # print(indices) 


# from hilbertcurve.hilbertcurve import HilbertCurve
# p=3; n=2
# hilbert_curve = HilbertCurve(p, n)
# distances = list(range(9))
# points = hilbert_curve.points_from_distances(distances)
# for point, dist in zip(points, distances):
#     print(f'point(h={dist}) = {point}')


# from hilbertcurve.hilbertcurve import HilbertCurve

# def get_hilbert_indices(W, H):
#     """
#     Returns a list of Hilbert curve indices for each (x, y) pixel coordinate
#     in an image of width W and height H using the hilbertcurve library.
#     """
#     # Determine the order of the curve needed to cover the image
#     N = max(W, H)
#     p = 0  # p is the order of the curve
#     while (1 << p) < N:
#         p += 1

#     # Initialize Hilbert Curve
#     hilbert_curve = HilbertCurve(p, 2)

#     # Generate Hilbert indices for each pixel
#     indices = []
#     for y in range(H):
#         for x in range(W):
#             index = hilbert_curve.points_from_distances([x, y])
#             indices.append(index)
    
#     return indices

# W= 8
# points = [(i,j) for i in range(W) for j in range(W)]
# distances = hilbert_curve.distances_from_points(points)
# for point, dist in zip(points, distances):
#     print(point, dist)

if __name__ == "__main__":
    for index in range(4*4):
        x, y = hibert_index_to_xy(index, 4, 4)
        print(f"Index {index} has pixel ({x}, {y})")