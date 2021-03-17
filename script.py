from utils import *
import cv2

# from PIL import Image
# con = ["3", "4", "5", "6"]
# for i in con:
#     path = './cones/cone' + i + '.jpg'
#     op = './cones/cone' + i + '.png'
#     im1 = Image.open(path)
#     im1.save(op)

for i in range(2, 12):
    print(i)
    img = cv2.imread('./cones/cone' + str(i) + '.png', cv2.IMREAD_COLOR)
    img = cv2.bilateralFilter(img, 4, 80, 80)

    edge_im = find_edges(img)


    corners = find_corners(edge_im)
    mark_corners(edge_im)
    cv2.imwrite("edges" + str(i) + ".png", edge_im)

    # print(corners)
    # mark_corners(img)
    lines, interest_corners = find_sides_corner(corners, edge_im)
    lines = find_sides(corners, edge_im)
    base_slope = horizon_slope(lines)
    x_lines = find_horizontal(interest_corners, edge_im, base_slope)
    for x, y in interest_corners:
        cv2.circle(img, (x, y), 5, 255, -1)
    for line in lines:
        cv2.line(img, tuple(line[0]), tuple(line[1]), (0, 0, 255), 1)
    print('len =', len(x_lines))
    for x_line in x_lines:
        cv2.line(img, tuple(x_line[0]), tuple(x_line[1]), (0, 0, 255), 1)
    cv2.imwrite("lines" + str(i) + ".png", img)

