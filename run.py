from interactive_demo.controller import  InteractiveController
from isegm.inference import utils
from isegm.utils import exp

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SHAPE = (448, 448)
MASK_PATH = 'mask.png'
OUT_PATH = 'merged.png'

def update_image(img):
	cv2.imwrite(MASK_PATH,img)
    
def get_mask(img_path, weights_path, device, x, y):
	cfg = exp.load_config_file('config.yml', return_edict=True)

	torch.backends.cudnn.deterministic = True
	checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, weights_path)
	model = utils.load_is_model(checkpoint_path, device, False, cpu_dist_maps=True)

	img = cv2.imread(img_path)
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_res = cv2.resize(img_rgb, IMG_SHAPE) 


	controller = InteractiveController(model, device,
	                                                predictor_params={'brs_mode': 'NoBRS'},
	                                                update_image_callback=update_image)

	mask = np.zeros(img_res.shape[:2])
	controller.set_image(img_res)
	controller.set_mask(mask)
	controller.reset_predictor()
	controller.add_click(x*IMG_SHAPE[1]/img.shape[1], y*IMG_SHAPE[0]/img.shape[0], True)
	controller.finish_object()

	mask = cv2.imread(MASK_PATH)
	mask_res = cv2.resize(mask, (img.shape[1], img.shape[0]))
	mask_res = cv2.cvtColor(mask_res, cv2.COLOR_BGR2GRAY)
	
	return mask_res


if __name__ == '__main__':
	img_path = 'assets/test_imgs/parrots.jpg'
	weights_path = './weights/sbd_vit_xtiny.pth'
	device = 'cpu'
	x = 621
	y = 461

	mask = get_mask(img_path, weights_path, device, x, y)
	print("Done")
