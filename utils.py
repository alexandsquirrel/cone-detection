import cv2
import numpy as np
import itertools


def process_image(img, boxes, nums):
    patches = []

    for box_idx in range(nums[0]):
        box = boxes[0][box_idx]
        sub_image = get_subregion(img, box)
        # sub_image = apply_transformation(sub_image, gaussian_filter)
        # sub_image = apply_transformation(sub_image, highpass_filter)
        sub_image = cv2.bilateralFilter(sub_image, 5, 80, 80)

        sub_image = find_edges(sub_image)
        cv2.imshow('hi2', sub_image)
        cv2.waitKey(0)
        mark_corners(sub_image)
        sub_image = cv2.cvtColor(sub_image, cv2.COLOR_GRAY2RGB)
        patches.append((sub_image, box))
        cv2.imshow('hi', sub_image)
        cv2.waitKey(0)
   
    for patch, box in patches:
        overlay(img, patch, box)
    
    return img


def get_subregion(im, box):
    height, width, _ = im.shape
    print(height, width)
    y1 = int(box[0] * width)
    x1 = int(box[1] * height)
    y2 = int(box[2] * width)
    x2 = int(box[3] * height)
    return im[x1:x2, y1:y2]


def apply_transformation(im, fn):
    return fn(im)


def overlay(src, patch, box):
    src_height, src_width, _ = src.shape
    x = int(box[1] * src_height)
    y = int(box[0] * src_width)
    patch_height, patch_width, _ = patch.shape
    src[x:x+patch_height, y:y+patch_width] = patch


def find_corners(im, ncorners=25):  # ncorners: more -> better but slower
    corners = cv2.goodFeaturesToTrack(im, ncorners, 0.01, 10)
    return np.int0(corners)


def mark_corners(im, ncorners=25):
    corners = find_corners(im, ncorners)
    # np.random.shuffle(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(im, (x, y), 3, 255, -1)


def find_edges(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
    return edges


def find_contours(im):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 30, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[-3]
    print(hierarchy)
    cv2.drawContours(im, [contour], -1, (0,255,0), 3)
    return im


def slope_of(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y1 - y2  # deliberated inverted
    if dx == 0:
        return np.inf
    return dy / dx


def get_boundary_point(p, slope):
    return get_point_by_y(p, slope, 0)


def get_point_by_y(p, slope, y_prime):
    x, y = p
    x_prime = int(x + (y - y_prime) / slope)
    return np.array([x_prime, y_prime])


def score_of_line_brute_force(p, slope, edge_im):  # TODO
    # note that p could be out-of-bounds
    range_neighbors = 1
    x_base, y_base = p  # y_base must be zero
    score = 0
    for y in range(y_base, edge_im.shape[0]):
        x = int(x_base - y / slope)
        # if not 0 <= x <= edge_im.shape[1]: continue
        neighbors = edge_im[y, x-range_neighbors:x+range_neighbors+1]
        white_count = 1 if np.sum(neighbors) / 255 > 0 else 0
        score += white_count
    return score


# this is specifict for hor
def score_of_line_hor(x1, x2, y1, y2, edge_im):
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    # note that p could be out-of-bounds
    range_neighbors = np.int(edge_im.shape[0] / 40)
    score = 0
    for x in range(x1, x2+1):
        y = int((y2-y1)/(x2-x1)+y1)
        # if not 0 <= x <= edge_im.shape[1]: continue
        # we only care the down side
        neighbors = edge_im[y:y+range_neighbors+1, x]
        white_count = 1 if np.sum(neighbors) / 255 > 0 else 0
        score += white_count
    # print(score)
    return score


# mark the corner to the edge image
def mark_corner_edge(corners, edge_im):
    for corner in corners:
        x, y = corner.ravel()
        edge_im[y, x] = 1  # TODO: not sure we should mark y,x or x,y


# we want to score the line based on the nearby corners and edges. 
# the corner should be seperated by a given amount -> similar like group
# the corner and we only consider a group as 1 (prev implementation as only
# seperate by given x from the previous one)
def score_of_line_with_corner(p, slope, edge_im, y1, y2, x_dif=10):
    if y1 > y2:
        y1, y2 = y2, y1
    y_start = y1 + (y2-y1)/5
    y_end = y2 - (y2-y1)/5
    # note that p could be out-of-bounds
    range_neighbors = 4
    x_base, y_base = p  # y_base must be zero
    x_dif = edge_im.shape[1] / 20
    score = 0
    prev = -100
    corners = set()
    for y in range(y1, y2+1):
        x = int(x_base - y / slope)
        # if not 0 <= x <= edge_im.shape[1]: continue
        # neighbors = edge_im[y, x-range_neighbors:x+range_neighbors+1]
        neighbors = edge_im[y, x-range_neighbors:x+range_neighbors+1]
        white_count = 1 if np.sum(neighbors) / 255 > 0 else 0
        score += white_count
        for x_app in range(x-range_neighbors, x+range_neighbors+1):
            if not 0 < x_app < edge_im.shape[1]:
                continue
            if edge_im[y, x_app] == 255:
                score += 1
            if edge_im[y, x_app] == 1:
                # we only care the cones in around the middle
                if y_start < y < y_end:
                    corners.add((x_app, y))
                # corners.add((x_app, y))
                if x_app-prev > x_dif:
                    score += 10
                    prev = x_app
    # score += (y2-y1) * 5.0 / edge_im.shape[0]
    return (score, corners)


def score_of_line(p, slope, edge_im):
    return score_of_line_brute_force(p, slope, edge_im)


def line_average_corner(lines, im_y):
    upper_average = np.array([0, 0])
    lower_average = np.array([0, 0])
    slope_average = 0
    sum_scores = sum([tup[3] for tup in lines])
    for line in lines:
        c1, _, slope, score, _ = line
        upper_boundary = get_point_by_y(c1, slope, 0)
        lower_boundary = get_point_by_y(c1, slope, im_y)
        upper_average = upper_average + upper_boundary * (score / sum_scores)
        lower_average = lower_average + lower_boundary * (score / sum_scores)
        slope_average += slope * (score / sum_scores)
    return (upper_average.astype(np.uint8), lower_average.astype(np.uint8), slope_average)


def line_average(lines, im_y):
    upper_average = np.array([0, 0])
    lower_average = np.array([0, 0])
    slope_average = 0
    sum_scores = sum([tup[3] for tup in lines])
    for line in lines:
        c1, _, slope, score = line
        upper_boundary = get_point_by_y(c1, slope, 0)
        lower_boundary = get_point_by_y(c1, slope, im_y)
        upper_average = upper_average + upper_boundary * (score / sum_scores)
        lower_average = lower_average + lower_boundary * (score / sum_scores)
        slope_average += slope * (score / sum_scores)
    return (upper_average.astype(np.uint8), lower_average.astype(np.uint8), slope_average)


# generate the slides for the cone as well as return the possible corners
# that might be the middle interest point that we want to check latter
def find_sides_corner(corners, edge_im):
    mark_corner_edge(corners, edge_im)
    slope_cutoff = 2
    interest_corners = set()
    lefts = []
    rights = []
    for c1, c2 in itertools.combinations(corners, 2):
        c1, c2 = c1.ravel(), c2.ravel()
        slope = slope_of(c1, c2)
        # print("When c1 = {} and c2 = {}, slope = {}".format(c1, c2, slope))
        if slope == np.inf: continue
        if np.abs(slope) < slope_cutoff: continue
        p = get_boundary_point(c1, slope)
        score, interest_corner = score_of_line_with_corner(p, slope, edge_im, c1[1], c2[1])
         # if slope > 0 and lefter point starts from right half, then discard
        if slope > slope_cutoff:
            lefts.append((c1, c2, slope, score, interest_corner))
        elif slope < -slope_cutoff:
            rights.append((c1, c2, slope, score, interest_corner))
            
    lefts.sort(key=lambda tup: tup[3], reverse=True)
    rights.sort(key=lambda tup: tup[3], reverse=True)
    for i in range(0, 3):
        interest_corners.update(lefts[i][4])
        interest_corners.update(rights[i][4])
    return ([line_average_corner(lefts[:3], edge_im.shape[0]), line_average_corner(rights[:3], edge_im.shape[0])], interest_corners)

def find_sides(corners, edge_im):
    slope_cutoff = 2
    lefts = []
    rights = []
    for c1, c2 in itertools.combinations(corners, 2):
        c1, c2 = c1.ravel(), c2.ravel()
        slope = slope_of(c1, c2)
        # print("When c1 = {} and c2 = {}, slope = {}".format(c1, c2, slope))
        if slope == np.inf: continue
        if np.abs(slope) < slope_cutoff: continue
        p = get_boundary_point(c1, slope)
        score = score_of_line_brute_force(p, slope, edge_im)

         # if slope > 0 and lefter point starts from right half, then discard
        if slope > slope_cutoff:
            lefts.append((c1, c2, slope, score))
        elif slope < -slope_cutoff:
            rights.append((c1, c2, slope, score))

    lefts.sort(key=lambda tup: tup[3], reverse=True)
    rights.sort(key=lambda tup: tup[3], reverse=True)
    return [line_average(lefts[:3], edge_im.shape[0]), line_average(rights[:3], edge_im.shape[0])]

def horizon_slope(lines):
    a = lines[0][2]
    b = lines[1][2]
    print(a, b, -(a + b) / (-(a**2 * b**2 + a**2 + b**2 + 1) ** 0.5 + a*b - 1))
    return -(a + b) / (-(a**2 * b**2 + a**2 + b**2 + 1) ** 0.5 + a*b - 1)

def euclidean_dist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5


def find_horizontal(corners, edge_im, base_slope):
    slope_cutoff = 0.5
    lines = []
    for c1, c2 in itertools.combinations(corners, 2):
        if (c1[1] > c2[1]):
            c1, c2 = c2, c1
        slope = slope_of(c1, c2) - base_slope
        if np.abs(slope) > slope_cutoff: continue

        score = score_of_line_hor(c1[0], c2[0], c1[1], c2[1], edge_im)
        score /= (np.abs(slope) + 0.1) ** 1  # my ugly heuristic
        score /= euclidean_dist(c1, c2)  # yet another ugly heuristic

        lines.append((c1, c2, slope, score))     

    lines.sort(key=lambda tup: tup[3], reverse=True)
    sort_num = min(len(lines), 3)
    lines = lines[:sort_num]
    return lines[0:2]



# The following functions are potentially useless (but fun)
def highpass_filter(im):
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    return cv2.filter2D(im, -1, kernel)


def gaussian_filter(im):
    kernel = np.ones((5,5),np.float32)/25
    return cv2.filter2D(im, -1, kernel)


# For testing purposes only
if __name__ == '__main__':
    image = cv2.imread('output.jpg')
    cv2.imshow('image',image)
    cv2.waitKey(0)
