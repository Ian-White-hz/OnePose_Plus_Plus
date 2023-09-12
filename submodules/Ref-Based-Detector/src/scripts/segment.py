import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from src.utils.bbox_utils import CropResizePad
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
from src.model.utils import Detections, convert_npz_to_json
from src.model.loss import Similarity
from src.utils.inout import save_json_bop23
from src.utils.matching import LoFTR
import cv2
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
from pathlib import Path


inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

def visualize(rgb, detections, obj_name, qry_id):
    save_path = osp.join("/mnt/data2/interns/gid-baiyan/cnos/outputs/",
                         obj_name+"_masks",
                         "qry"+f"{qry_id}")
    Path(save_path).mkdir(parents=True,exist_ok=True)
    logging.info(f"mask_save_path: {save_path} ")
    img = rgb.copy()

    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    mask_save_path = osp.join(save_path, f"origin_gray.png")
    cv2.imwrite(mask_save_path, img)

    # img = (255*img).astype(np.uint8)
    # colors = distinctipy.get_colors(len(detections))
    colors = [(0,0,128),(0,0,255),(30,144,255),(135,206,250),(245,255,250)]
    # logging.info(f"length of detections {len(detections)} ")
    # logging.info(f"colors: {colors} ")
    alpha = 0.66

    colored_img = img.copy()
    for mask_idx, det in enumerate(detections):
        mask = rle_to_mask(det["segmentation"])
        #logging.info(f"mask: {mask} ")
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        obj_id = det["category_id"]
        #temp_id = obj_id - 1
        temp_id = mask_idx

        # r = int(255*colors[temp_id][0])
        # g = int(255*colors[temp_id][1])
        # b = int(255*colors[temp_id][2])
        r = int(colors[temp_id][0])
        g = int(colors[temp_id][1])
        b = int(colors[temp_id][2])
        
        crop_img = np.zeros_like(img)
        
        crop_img[mask, 0] = img[mask, 0]
        # logging.info(f"img_r: {img[mask, 0]} ")
        # logging.info(f"crop_img_r: {crop_img[mask, 0]} ")
        crop_img[mask, 1] = img[mask, 1]
        crop_img[mask, 2] = img[mask, 2]
        # crop_img = crop_img - img

        colored_img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        colored_img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        colored_img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2] 
        max_mnum, sum_mnum = 0, 0
        global max_mnums, average_mnums
        for ref_idx in range(15):
            LT = LoFTR(obj_name = obj_name, query_idx = qry_id, 
                            ref_idx= ref_idx, top_idx = temp_id, 
                            mask_qry = True, test_mode = False,
                            crop_img = crop_img, crop_mask = mask)
            #img0, img1, ts_mask_0, ts_mask_1 = LT.load_img_mask()
            mkpts0, mkpts1, inliers0, inliers1, mnum = LT.get_matching_result(img0, img1, ts_mask_0,ts_mask_1)
            LT.add_kpc_to_vis3d(img0, img1, inliers0, inliers1)
            if max_mnum < mnum :
                max_mnum = mnum
                best_id = ref_idx
            sum_mnum = sum_mnum + mnum
            print("-----------------------------------")
        average_mnum = sum_mnum / 20
        max_mnums.append(max_mnum)
        average_mnums.append(average_mnum)
        print("obj_name", obj_name)
        print("qry_idx:",qry_id)
        print("top_idx:",temp_id)
        print("average_mnum:",average_mnum)
        print("max_mnum:",max_mnum)
        print("best_id:",best_id)
        print("-----------------------------------")
        cv2.imwrite(osp.join(save_path, f"{mask_idx}.png"), crop_img)        
        np.savetxt(osp.join(save_path, f"{mask_idx}.txt"), mask, fmt='%d')
        # img = Image.fromarray(np.uint8(crop_img))
        # img.save(save_path)
        
        #img[edge, :] = 255

    img = Image.fromarray(np.uint8(colored_img))
    save_path = osp.join(save_path, f"vis.png")
    img.save(save_path)
    prediction = Image.open(save_path)

    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def global_feats(template_dir):
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name='run_inference.yaml')

    metric = Similarity()
    logging.info("Initializing model")
    model = instantiate(cfg.model)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")


    logging.info("Initializing template")
    template_paths = glob.glob(f"{template_dir}/*.png")
    boxes, templates = [], []
    for path in template_paths:
        image = Image.open(path)
        boxes.append(image.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        templates.append(image)

    templates = torch.stack(templates).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))

    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = CropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates, boxes=boxes).cuda(3)
    save_image(inv_rgb_transform(templates), f"{template_dir}/cnos_results/templates.png", nrow=7)
    ref_feats = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                )
    logging.info(f"Ref feats: {ref_feats.shape}")
    return model, metric, ref_feats

def run_inference(template_dir, rgb_path, num_max_dets, conf_threshold, obj_name, qry_id, model, metric, ref_feats):
    # with initialize(version_base=None, config_path="../../configs"):
    #     cfg = compose(config_name='run_inference.yaml')

    # metric = Similarity()
    # logging.info("Initializing model")

    # run inference
    rgb = Image.open(rgb_path)
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    decriptors = model.descriptor_model.forward(np.array(rgb), detections)

    # get scores per proposal
    scores = metric(decriptors[:, None, :], ref_feats[None, :, :])
    score_per_detection = torch.topk(scores, k=5, dim=-1)[0]
    score_per_detection = torch.mean(
        score_per_detection, dim=-1
    )

    # get top-k detections
    scores, index = torch.topk(score_per_detection, k=num_max_dets, dim=-1)
    detections.filter(index)
    detections.add_attribute("scores", scores)
    detections.add_attribute("object_ids", torch.zeros_like(scores))

    detections.to_numpy()
    save_path = f"{template_dir}/cnos_results/detection"
    detections.save_to_file(0, 0, 0, save_path, "custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", detections)
    vis_img = visualize(rgb, detections, obj_name=obj_name, qry_id=qry_id)
    vis_img.save(f"{template_dir}/cnos_results/vis.png")


if __name__ == "__main__":
    obj_name = "ape"
    qry_id = None
    dataset_path = "/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo"
    ref_frames_path = osp.join(dataset_path,
                                obj_name,
                                obj_name+"-annotate",
                                "color",)
    query_frames_path = osp.join(dataset_path,
                                obj_name,
                                obj_name+"-test",
                                "color_full",)
    model, metric, ref_feats = global_feats(ref_frames_path)
    logging.info("start inference")
    
    for qry_id in range(20, 22):
        max_mnums, average_mnums = [], []
        query_frame_path = osp.join(query_frames_path, f"{qry_id}.png")
        run_inference(ref_frames_path, query_frame_path, num_max_dets=5, conf_threshold=0.5, 
                      obj_name=obj_name, qry_id=qry_id, model=model, metric=metric, ref_feats=ref_feats)
        logging.info(f"max:{max_mnums}")
        logging.info(f"aver:{average_mnums}")
        print("————————————————————————")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("template_dir", nargs="?", 
    #                     default="/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/datasets/LM_dataset/0801-lm1-others/lm1-1/color",
    #                     help="Path to root directory of the template")
    # parser.add_argument("rgb_path", nargs="?", 
    #                     default="/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo/lamp/lamp-test/color_full/2.png",
    #                     help="Path to RGB image")
    # parser.add_argument("qry_id", nargs="?", default=2, type=int, help="qurey frame id")
    # parser.add_argument("num_max_dets", nargs="?", default=3, type=int, help="Number of max detections")
    # parser.add_argument("confg_threshold", nargs="?", default=0.5, type=float, help="Confidence threshold")
    # args = parser.parse_args()

    # os.makedirs(f"{args.template_dir}/cnos_results", exist_ok=True)
    # run_inference(ref_frame_path, args.rgb_path, num_max_dets=args.num_max_dets, conf_threshold=args.confg_threshold)