import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from kornia_moons.viz import draw_LAF_matches
from wis3d import Wis3D as Vis3D
import os
import os.path as osp
from pathlib import Path
from src.utils.data_utils import get_image_crop_resize, get_K_crop_resize


cfgs = {
    "model": {
        "method": "LoFTR",
        "weight_path": "weight/LoFTR_wsize9.ckpt",
        "seed": 666,
    },
}

class LoFTR():
    def __init__(self,obj_name,query_idx, ref_idx , top_idx,mask_qry):
        self.matcher = KF.LoFTR(pretrained="outdoor")
        #matcher = KF.LoFTR(pretrained="indoor")
        self.transf = transforms.ToTensor()
        self.mask_qry = mask_qry
        self.ref_idx = ref_idx
        self.qry_idx = query_idx
        self.confidence = None

        self.query_frame_path = osp.join("/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo",
                                         obj_name,
                                         obj_name+"-test",
                                         "color_full",)
        
        # self.masked_query_frame_path = osp.join("/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo",
        #                                  obj_name,
        #                                  obj_name+"-test",
        #                                  "masked",)
        # obj_name_masks/qry_id/topx_(png or txt)
        self.masked_query_frame_path = osp.join("/mnt/data2/interns/gid-baiyan/cnos/outputs",
                                                obj_name+"_masks",
                                                str(query_idx),
                                                )
        
        self.ref_frame_path = osp.join("/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo",
                                         obj_name,
                                         obj_name+"-annotate",
                                         "masked",)
        
        # self.qry_mask_path = osp.join("/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo",
        #                                  obj_name,
        #                                  obj_name+"-test",
        #                                  "boxes",)
        self.qry_mask_path = osp.join("/mnt/data2/interns/gid-baiyan/cnos/outputs",
                                                obj_name+"_masks",
                                                str(query_idx),
                                                )

        self.ref_mask_path = osp.join("/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo",
                                         obj_name,
                                         obj_name+"-annotate",
                                         "boxes",)
        
        self.bbox_vis_path = osp.join("/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo",
                                         obj_name,
                                         obj_name+"-test",
                                         "bbox_vis",)
        
        # Path(self.bbox_vis_path).mkdir(exist_ok=True)
        if self.mask_qry:
            self.query_fname = osp.join(self.masked_query_frame_path, str(top_idx) + ".png")
            self.wis3d_pth = osp.join('/mnt/data2/interns/gid-baiyan/test/loftr',
                    'wis3d',
                    "masked_"+ obj_name +"_masked",
                    str(query_idx),
                    )
        else:
            self.query_fname = osp.join(self.query_frame_path, str(query_idx) + ".png")
            self.wis3d_pth = osp.join('/mnt/data2/interns/gid-baiyan/test/loftr',
                    'wis3d',
                    obj_name + "_masked",
                    str(query_idx),
                    )

        
        self.ref_fname = osp.join(self.ref_frame_path, str(ref_idx) + ".png")
        self.qry_mask = osp.join(self.qry_mask_path, str(top_idx) + ".txt")
        self.ref_mask = osp.join(self.ref_mask_path, str(ref_idx) + ".txt")

        # dump_dir, name = self.wis3d_pth.rsplit('/',1)
        # print("dump_dir:",dump_dir)
        # print("name:",name)
        self.vis3d = Vis3D(self.wis3d_pth, str(query_idx)+"_"+str(ref_idx))

        self.corners_homo = []
        

    def load_torch_image(self, fname):
        img_np = cv2.imread(fname)
        #print("origin img shape:",img.shape)
        mask = torch.zeros_like(self.transf(img_np))[0,:,:]
        img = self.np_to_ts(img_np)
        return img, mask 
    
    def np_to_ts(self, img):
        img = K.image_to_tensor(img, False).float() /255.
        img = K.color.bgr_to_rgb(img)
        img.permute(1,0,2,3)
        return img
    # def crop_resize_img(ref_corners,original_img,resize_shape):
    #     # Crop image by 2D visible bbox, and change K
    #     box = np.array([ref_corners[0], ref_corners[1], ref_corners[2], ref_corners[3]])
    #     resize_shape = np.array([ref_corners[3] - ref_corners[1], ref_corners[2] - ref_corners[0]])
    #     K_crop, K_crop_homo = get_K_crop_resize(box, K, resize_shape)
    #     image_crop, _ = get_image_crop_resize(original_img, box, resize_shape)
    #     image_masked = get_masked_image(original_img , box)

    #     box_new = np.array([0, 0, ref_corners[2] - ref_corners[0], ref_corners[3] - ref_corners[1]])
    #     resize_shape = np.array([resize_shape, resize_shape])
    #     K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
    #     image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)
        
        # return image_crop, K_crop
    def crop_resize_img(x0,y0,x1,y1,original_img):
        # Crop image by 2D visible bbox, and change K
        box = np.array([x0, y0, x1, y1])
        resize_shape = np.array([y1 - y0, x1 - x0])
        K_crop, K_crop_homo = get_K_crop_resize(box, K, resize_shape)
        image_crop, _ = get_image_crop_resize(original_img, box, resize_shape)

        box_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape = np.array([256, 256])
        K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
        image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)
    
    def load_img_mask(self):
        ref_corners=[]#x0,y0,x1,y1
        qry_corners=[640,640,0,0]

        with open(self.ref_mask, 'r', encoding='utf-8') as f:
            for content in f.readlines():
                ref_corners.append(int(float(content.strip('\n'))))
        
        print("query frame path:",self.query_fname)
        print("ref frame path:",self.ref_fname)

        img0, mask0 = self.load_torch_image(self.query_fname)
        img1, mask1 = self.load_torch_image(self.ref_fname)

        qry_img = cv2.imread(self.query_fname)
        ref_img = cv2.imread(self.ref_fname)
        print(ref_corners)
        #ref_img_crop = self.crop_resize_img((ref_corners[0],ref_corners[1],ref_corners[2],ref_corners[3]),ref_img)
        # Crop image by 2D visible bbox, and change K
        original_K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
        box = np.array([ref_corners[0], ref_corners[1], ref_corners[2], ref_corners[3]])
        resize_shape = np.array([ref_corners[3] - ref_corners[1], ref_corners[2] - ref_corners[0]])
        K_crop, K_crop_homo = get_K_crop_resize(box, original_K, resize_shape)
        ref_image_crop, _ = get_image_crop_resize(ref_img, box, resize_shape)

        box_new = np.array([0, 0, ref_corners[2] - ref_corners[0], ref_corners[3] - ref_corners[1]])
        resize_shape = np.array([256, 256])
        K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
        ref_image_crop, _ = get_image_crop_resize(ref_image_crop, box_new, resize_shape)
        cv2.imwrite(osp.join('/mnt/data2/interns/gid-baiyan/cnos/test/crop','ref' + str(self.ref_idx)+'.png'),ref_image_crop)
        print("crop frame path:",osp.join('/mnt/data2/interns/gid-baiyan/cnos/test/crop',str(self.ref_idx)+'.png'))
        H, W = ref_img.shape[-2:]
        self.corners_homo.append(
            np.array(
                [
                    [0, 0, 1],
                    [W, 0, 1], # w, 0
                    [0, H, 1], # 0, h
                    [W, H, 1],
                ]
            ).T  # 3*4
        )
        print("len of homo",len(self.corners_homo))
        # for sparse mask
        # if self.mask_qry:
        #     with open(self.qry_mask, 'r', encoding='utf-8') as f:
        #         for content in f.readlines():
        #             qry_corners.append(int(float(content.strip('\n'))))
        #     mask0[qry_corners[1]:qry_corners[3],qry_corners[0]:qry_corners[2]] = 1
        # else:
        #     mask0 = torch.ones_like(mask0)

        # for dense mask
        if self.mask_qry:
            mask0 = torch.from_numpy(np.loadtxt(self.qry_mask,dtype=np.int8, delimiter=' '))
        else:
            mask0 = torch.ones_like(mask0)
        for x in range(mask0.shape[1]):
            for y in range(mask0.shape[0]):
                if mask0[y,x] == 1:
                    qry_corners[0] = np.min([qry_corners[0],x])
                    qry_corners[1] = np.min([qry_corners[1],y])
                    qry_corners[2] = np.max([qry_corners[2],x])
                    qry_corners[3] = np.max([qry_corners[3],y])
        mask1[ref_corners[1]:ref_corners[3],ref_corners[0]:ref_corners[2]] = 1

        box = np.array([qry_corners[0], qry_corners[1], qry_corners[2], qry_corners[3]])
        resize_shape = np.array([qry_corners[3] - qry_corners[1], qry_corners[2] - qry_corners[0]])
        K_crop, K_crop_homo = get_K_crop_resize(box, original_K, resize_shape)
        qry_image_crop, _ = get_image_crop_resize(qry_img, box, resize_shape)

        box_new = np.array([0, 0, qry_corners[2] - qry_corners[0], qry_corners[3] - qry_corners[1]])
        resize_shape = np.array([256, 256])
        K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
        qry_image_crop, _ = get_image_crop_resize(qry_image_crop, box_new, resize_shape)
        cv2.imwrite(osp.join('/mnt/data2/interns/gid-baiyan/cnos/test/crop','qry' + str(self.ref_idx)+'.png'),qry_image_crop)
        
        qry_image_crop = self.np_to_ts(qry_image_crop)
        ref_image_crop = self.np_to_ts(ref_image_crop)

        print("ref_corners:",ref_corners)
        print("qry_corners:",qry_corners)
        print("qry img shape:",img0.shape)
        print("ref img shape:",img1.shape)
        print("qry mask shape:",mask0.shape)
        print("ref mask shape:",mask1.shape)

        [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                    scale_factor=0.125,
                                                    mode='nearest',
                                                    recompute_scale_factor=False)[0]

        ts_mask_0.unsqueeze_(0)
        ts_mask_1.unsqueeze_(0)
        print("down sampled mask shape:",ts_mask_1.shape)

        return qry_image_crop, ref_image_crop, ts_mask_0, ts_mask_1


    def get_matching_result(self, img0, img1, ts_mask_0, ts_mask_1):
        input_dict = {
            "image0": K.color.rgb_to_grayscale(img0),  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(img1),
            #"mask0": ts_mask_0,
            #"mask1": ts_mask_1,
        }

        # with torch.inference_mode():
        with torch.no_grad():
            correspondences = self.matcher(input_dict)
        # print(correspondences.keys())
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        self.confidence = correspondences["confidence"].cpu().numpy()
        print("number of matching pairs",mkpts0.shape)
        
        #if no matching pairs
        if mkpts0.shape[0] == 0:
            print("no matching pairs!!!!!!!!")
            return None, None, None, None
        
        Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 1.0, 0.999, 100000)
        affine, _= cv2.estimateAffine2D(
                mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=6
            )
        
        # Estimate box:
        four_corner = self.corners_homo
        bbox = (affine @ four_corner).T.astype(np.int32)  # 4*2
        #print("bbox:",bbox)

        left_top = np.min(bbox, axis=0)
        right_bottom = np.max(bbox, axis=0)
        print(left_top)
        print(right_bottom)

        # w ,h = right_bottom - left_top
        # offset_percent = 0.0
        # x0 = left_top[0] - int(w * offset_percent)
        # y0 = left_top[1] - int(h * offset_percent)
        # x1 = right_bottom[0] + int(w * offset_percent)
        # y1 = right_bottom[1] + int(h * offset_percent)
        
        print("number of inliers",inliers.sum())
        num_in = inliers.sum()
        mkpts0 = mkpts0 * inliers
        mkpts1 = mkpts1 * inliers

        inliers0 = mkpts0[~np.all(mkpts0 == 0, axis=1)]
        inliers1 = mkpts1[~np.all(mkpts1 == 0, axis=1)]

        # inliers = inliers > 0
        return mkpts0, mkpts1, inliers0, inliers1, num_in

    #use vis3d to visualize the keypoints
    def add_kpc_to_vis3d(self, img0, img1, kpts0, kpts1):
        img0 = np.asarray(img0.cpu().numpy()[0][0]*256).astype(np.uint8)
        img1 = np.asarray(img1.cpu().numpy()[0][0]*256).astype(np.uint8)
        #vis3d = Vis3D(dump_dir, save_name)
        self.vis3d.add_keypoint_correspondences(img0, img1, kpts0, kpts1, 
                                                unmatched_kpts0 = None, unmatched_kpts1 = None, metrics = None, 
                                                booleans = None, meta = None, name = None)
    def metrics(self, inliers):
        pass

if __name__ == "__main__":
    obj_name = "ape"
    qry_idx = 1
    top_idx = 0
    best_id = 0
    max_mnum = 0
    sum_mnum = 0
    for ref_idx in range(20):
        LT = LoFTR_Test(obj_name = obj_name, query_idx = qry_idx, 
                        ref_idx= ref_idx, top_idx = top_idx, mask_qry=True)
        img0, img1, ts_mask_0, ts_mask_1 = LT.load_img_mask()
        mkpts0, mkpts1, inliers0, inliers1, mnum = LT.get_matching_result(img0, img1, ts_mask_0,ts_mask_1)
        LT.add_kpc_to_vis3d(img0, img1, inliers0, inliers1)
        if max_mnum < mnum :
            max_mnum = mnum
            best_id = ref_idx
        sum_mnum = sum_mnum + mnum
        print("-----------------------------------")
    average_mnum = sum_mnum / 20
    print("obj_name", obj_name)
    print("qry_idx:",qry_idx)
    print("top_idx:",top_idx)
    print("average_mnum:",average_mnum)
    print("max_mnum:",max_mnum)
    print("best_id:",best_id)