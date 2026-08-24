import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches

def get_all_tissue_bboxes(mask, min_area_ratio):
    """
    Parameters
    ----------
    mask : np.ndarray
    min_area_ratio : float. Filter out noise or falsely identified small points. If the area of a connected region is less than (total number of pixels in mask * min_area_ratio), it is ignored.

    Returns
    -------
    bboxes : list of tuples
    """
    if mask.dtype == bool:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        _, mask_uint8 = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return []

    total_pixels = mask.shape[0] * mask.shape[1]
    min_area = total_pixels * min_area_ratio

    bboxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            bboxes.append((left, top, width, height))

    return bboxes

def merge_close_coords(coords, threshold):
    """    merge close coordinates in a sorted list, return the merged sorted list    """
    if not coords:
        return []
    coords = sorted(coords)
    merged = [coords[0]]
    for c in coords[1:]:
        if c - merged[-1] <= threshold:
            merged[-1] = (c+merged[-1])//2  # take the average as the merged coordinate
        else:
            merged.append(c)
    return merged

def get_tiling_grid(bboxes, cell_size):
    """
    generate grid cells based on the boundaries of all boxes, return the rectangle area of each valid cell.
    """
    lefts, rights, tops, bottoms = [], [], [], []
    for (x, y, w, h) in bboxes:
        lefts.append(x)
        rights.append(x + w)
        tops.append(y)
        bottoms.append(y + h)

    all_x = merge_close_coords(lefts + rights, cell_size)
    all_y = merge_close_coords(tops + bottoms, cell_size)

    cells = []
    for i in range(len(all_x)-1):
        for j in range(len(all_y)-1):
            x1, x2 = all_x[i], all_x[i+1]
            y1, y2 = all_y[j], all_y[j+1]
            cells.append((x1, y1, x2-x1, y2-y1))

    return cells

# def filter_cells_by_mask(cells, mask, tissue_threshold,tile_size):
#     """
#     maintain the original cell if it contains enough tissue, otherwise ignore it. 
#     For the remaining cells, calculate the minimum bounding rectangle of the tissue area within it, and return those
#     """
#     valid_rects = []
#     for (x, y, w, h) in cells:
#         cell_mask = mask[y:y+h, x:x+w]
#         if cell_mask.size == 0:
#             continue
#         tissue_fraction = np.sum(cell_mask>0) / (w * h)
#         if tissue_fraction >= tissue_threshold:
#             # calculate the minimum bounding rectangle of the tissue area within it
#             ys, xs = np.where(cell_mask > 0)
#             if len(xs) == 0:
#                 continue
#             x_min, x_max = np.min(xs), np.max(xs)
#             y_min, y_max = np.min(ys), np.max(ys)
            
#             global_rect = (x + x_min, y + y_min, x_max - x_min + 1, y_max - y_min + 1)
#             # check if the rectangle size is sufficient for tiling
#             if global_rect[2] >= tile_size and global_rect[3] >= tile_size:
#                 valid_rects.append(global_rect)
#     return valid_rects

# faster version of filter_cells_by_mask
def filter_cells_by_mask_fast(cells, mask, tissue_threshold, tile_size):

    mask_bin = (mask > 0).astype(np.uint8)
    integral = cv2.integral(mask_bin)

    valid_rects = []

    for (x, y, w, h) in cells:
        if w == 0 or h == 0:
            continue

        x2 = x + w
        y2 = y + h
        area_sum = (
            integral[y2, x2]
            - integral[y, x2]
            - integral[y2, x]
            + integral[y, x]
        )
        tissue_fraction = area_sum / (w * h)

        if tissue_fraction < tissue_threshold:
            continue

        cell_mask = mask_bin[y:y+h, x:x+w]

        rows = np.any(cell_mask, axis=1)
        cols = np.any(cell_mask, axis=0)

        if not rows.any():
            continue

        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        y_min, y_max = y_indices[0], y_indices[-1]
        x_min, x_max = x_indices[0], x_indices[-1]
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        if width >= tile_size and height >= tile_size:
            valid_rects.append(
                (x + x_min, y + y_min, width, height)
            )

    return valid_rects

if __name__ == "__main__":

    MIN_AREA_RATIO = 0.001
    CELL_SIZE = 900
    TILE_SIZE = 224
    TISSUE_THRESHOLD = 0.1

    mask = cv2.imread("./dataset/wsi/LIHC_binary_mask/TCGA-DD-A1EB-01Z-00-DX1.705C4FB1-00CA-4726-9589-1F745FCC9729.svs_mask.png", cv2.IMREAD_GRAYSCALE)

    bboxes_original  = get_all_tissue_bboxes(mask, min_area_ratio=MIN_AREA_RATIO) 
    print(f" {len(bboxes_original )} tissue areas detected：")
    for i, (x,y,w,h) in enumerate(bboxes_original):
        print(f" Area {i+1}: upper left =({x},{y}), width={w}, height={h}, area={w*h}")

    cells = get_tiling_grid(bboxes_original, cell_size=CELL_SIZE)
    print(f"{len(cells)} cells generated")

    # e1 = cv2.getTickCount()
    # valid_rects = filter_cells_by_mask(cells, mask, tissue_threshold=TISSUE_THRESHOLD,tile_size=TILE_SIZE)
    # print(f"Valid squares for tiling: {len(valid_rects)} areas")
    # e2 = cv2.getTickCount()
    # time_ms = (e2 - e1) / cv2.getTickFrequency() * 1000
    # print(f"Filtering valid tiling rectangles took {time_ms:.2f} ms")

    e1 = cv2.getTickCount()
    valid_rects = filter_cells_by_mask_fast(cells, mask, tissue_threshold=TISSUE_THRESHOLD,tile_size=TILE_SIZE)
    print(f"Valid squares for tiling: {len(valid_rects)} areas")
    e2 = cv2.getTickCount()
    time_ms = (e2 - e1) / cv2.getTickFrequency() * 1000
    print(f"Filtering valid tiling rectangles took {time_ms:.2f} ms")



    # visualization
    fig, axes = plt.subplots(1, 2, figsize=(16,8))

    # left: original tissue bounding boxes
    display_img = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    axes[0].imshow(display_img)
    for (x,y,w,h) in bboxes_original:
        rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
    axes[0].set_title(f'Original Bounding Boxes ({len(bboxes_original)})')
    axes[0].axis('off')

    # right: valid tiling rectangles
    display_img2 = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    axes[1].imshow(display_img2)
    for (x,y,w,h) in valid_rects:
        rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        axes[1].add_patch(rect)
    axes[1].set_title(f'Valid Tiling Rectangles ({len(valid_rects)})')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()