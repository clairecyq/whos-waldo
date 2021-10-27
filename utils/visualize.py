import os
import base64
import io
import json, glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

whos_waldo_dir = '/whos_waldo'

with open('/dataset_meta/blurry_bbs.json', 'r', encoding='utf-8') as file:
    blurry_bbs = json.load(file)

BLACK = [0, 0, 0]

colors = {
    'tomato': [255,99,71],
    'gold': [255,215,0],
    'mediumaquamarine': [102,205,170],
    'blueviolet': [138,43,226],
    'cornflowerblue': [100,149,237],
    'lightpink': [255,182,193],
    'crimson': [220,20,60],
    'lightsalmon': [255,160,122],
    'hotpink': [255,105,180],
    'sienna': [160,82,45],
    'deepskyblue': [0,191,255],
    'aquamarine': [127,255,212],
    'chocolate': [210,105,30],
    'darkolivegreen': [85,107,47],
    'deeppink': [255,20,147],
    'lawngreen': [124,252,0],
    'lightskyblue': [135,206,250],
    'slateblue': [106,90,205],
    'darkmagenta': [139,0,139],
    'wheat': [245,222,179],
    'lightseagreen': [32,178,170],
    'khaki': [240,230,140],
    'orangered': [255,69,0],
    'indianred': [205,92,92],
    'cadetblue': [95,158,160],
    'teal': [0,128,128],
    'plum': [221,160,221],
}


def draw_heatmap(data, x, y, ax, vmax, cbar=False):
    map = seaborn.heatmap(data,
                    xticklabels=x, square=True, yticklabels=y, annot=True, vmin=0.0, vmax=vmax,
                    cbar=cbar, linewidths=0.5, fmt='.4f', ax=ax, annot_kws={"size": 10})
    map.set_xticklabels(map.get_xmajorticklabels(), fontsize=16)
    map.set_yticklabels(map.get_ymajorticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")


def print_img(image_path, html_file):
    if os.path.exists(image_path):
        img = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
        print('<img src="data:image/png;base64,{0}" height="300">'.format(img), file=html_file)


def print_matrix(id, matrix, matrix_name, box_cnt, names, vmax, output_dir=None, html_out=None):
    fig, ax = plt.subplots()

    draw_heatmap(matrix, range(box_cnt), names, ax=ax, vmax=vmax)
    for c in range(box_cnt):
        ax.add_patch(
            patches.Rectangle((c + 0.1, -0.9), 0.8, 0.8, facecolor=list(colors.keys())[c], fill=True, clip_on=False)
        )
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    if output_dir and html_out:
        if not os.path.exists(f'{output_dir}/matrices'):
            os.mkdir(f'{output_dir}/matrices')
        fig_path = f'{output_dir}/matrices/{id})_matrix_{matrix_name}.png'
        plt.savefig(fig_path, bbox_inches="tight")
        print_img(fig_path, html_out)

    return buf


def print_visualization(id, num_idens, boxes, sim, null_id, gt, html_out=None, output_dir=None):

    folder = os.path.join(whos_waldo_dir, id)
    image_name = [f for f in glob.glob(os.path.join(folder, '*.*'))
                  if f.endswith('.json') is False and f.endswith('.txt') is False][0]
    img_read = cv2.imread(image_name)
    h, w, _ = img_read.shape
    height = 600
    width = int(w * float(height) / h)
    dim = (width, height)
    h, w = height, width
    img_read = cv2.resize(img_read, dim)

    box_cnt = 0
    for row in range(boxes.shape[0]):
        b = boxes[row].cpu().numpy()
        if (b[6]==0): continue
        box = BoundingBox(x1=b[0] * w, x2=b[2] * w, y1=b[1] * h, y2=b[3] * h)
        bbs = BoundingBoxesOnImage([box], shape=img_read.shape)
        try:
            rgb = colors[list(colors.keys())[box_cnt]]
        except:
            print(f'not enough colors for {box_cnt} boxes, please add more colors')
        bgr = [rgb[2], rgb[1], rgb[0]]
        img_read = bbs.draw_on_image(img_read, color=bgr, size=5)
        box_cnt += 1

    vis_dir = f'{output_dir}/visualizations/{id}'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    new_img_path = os.path.join(vis_dir, id + '_with_boxes.jpg')
    cv2.imwrite(new_img_path, img_read)

    if html_out:
        print('<p>' + 'img_id: ' + id + '</p>', file=html_out)
        with open(os.path.join(folder, 'caption.txt'), 'r') as f:
            caption = [n.rstrip('\n') for n in f.readlines()][0]
        print('<p>' + 'caption: ' + caption + '</p>', file=html_out)
        print('<p>' + 'gt: ' + str(gt) + '</p>', file=html_out)
        if null_id and id in blurry_bbs.keys():
            blurry_list = blurry_bbs[id]
            print('<p>' + 'blurry boxes: ' + str(blurry_list) + '</p>', file=html_out)
        print_img(new_img_path, html_out)  # print original image with boxes

    names = ['[NAME]'] * num_idens
    if null_id:
        names.append(r'$\varnothing$')

    buf_sim = print_matrix(id, sim, 'Similarities', box_cnt, names, 1.0, output_dir, html_out)
    if html_out:
        print('<hr>', file=html_out)

    return new_img_path, buf_sim  # path of the boxed image + buf of matrix
