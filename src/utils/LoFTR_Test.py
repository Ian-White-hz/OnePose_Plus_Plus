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
import os.path as osp

cfgs = {
    "model": {
        "method": "LoFTR",
        "weight_path": "weight/LoFTR_wsize9.ckpt",
        "seed": 666,
    },
}

class LoFTR_Test():
    def __init__(self,obj_name,query_idx, ref_idx , mask_qry):
        self.matcher = KF.LoFTR(pretrained="outdoor")
        #matcher = KF.LoFTR(pretrained="indoor")
        self.transf = transforms.ToTensor()
        self.mask_qry = mask_qry

        self.query_frame_path = osp.join("/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo",
                                         obj_name,
                                         obj_name+"-test",
                                         "color_full",)
        
        self.masked_query_frame_path = osp.join("/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo",
                                         obj_name,
                                         obj_name+"-test",
                                         "masked",)
        
        self.ref_frame_path = osp.join("/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo",
                                         obj_name,
                                         obj_name+"-annotate",
                                         "masked",)
        
        self.qry_mask_path = osp.join("/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo",
                                         obj_name,
                                         obj_name+"-test",
                                         "boxes",)
        
        self.ref_mask_path = osp.join("/mnt/data2/interns/gid-baiyan/OnePose_Plus_Plus/data/demo",
                                         obj_name,
                                         obj_name+"-annotate",
                                         "boxes",)
        
        if self.mask_qry:
            self.query_fname = osp.join(self.masked_query_frame_path, str(query_idx) + ".png")
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
        self.qry_mask = osp.join(self.qry_mask_path, str(query_idx) + ".txt")
        self.ref_mask = osp.join(self.ref_mask_path, str(ref_idx) + ".txt")

        # dump_dir, name = self.wis3d_pth.rsplit('/',1)
        # print("dump_dir:",dump_dir)
        # print("name:",name)
        self.vis3d = Vis3D(self.wis3d_pth, str(query_idx)+"_"+str(ref_idx))
        

    def load_torch_image(self, fname):
        img = cv2.imread(fname)
        #print("origin img shape:",img.shape)
        mask = torch.zeros_like(self.transf(img))[0,:,:]
        img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
        img = K.color.bgr_to_rgb(img)
        img.permute(1,0,2,3)
        return img, mask 
    
    def load_img_mask(self, ref_idx):
        ref_corners=[]#x0,y0,x1,y1
        qry_corners=[]

        with open(self.ref_mask, 'r', encoding='utf-8') as f:
            for content in f.readlines():
                ref_corners.append(int(float(content.strip('\n'))))
        
        print("query frame path:",self.query_fname)
        print("ref frame path:",self.ref_fname)

        img0, mask0 = self.load_torch_image(self.query_fname)
        img1, mask1 = self.load_torch_image(self.ref_fname)

        if self.mask_qry:
            with open(self.qry_mask, 'r', encoding='utf-8') as f:
                for content in f.readlines():
                    qry_corners.append(int(float(content.strip('\n'))))
            mask0[qry_corners[1]:qry_corners[3],qry_corners[0]:qry_corners[2]] = 1
        else:
            mask0 = torch.ones_like(mask0)

        mask1[ref_corners[1]:ref_corners[3],ref_corners[0]:ref_corners[2]] = 1

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

        # ts_mask_1 = F.interpolate(mask1,
        #                           scale_factor=0.125,
        #                           mode='nearest',
        #                           recompute_scale_factor=False)[0].bool()
        ts_mask_0.unsqueeze_(0)
        ts_mask_1.unsqueeze_(0)
        print("down sampled mask shape:",ts_mask_1.shape)
        return img0, img1, ts_mask_0, ts_mask_1


    def get_matching_result(self, img0, img1, ts_mask_0, ts_mask_1):
        input_dict = {
            "image0": K.color.rgb_to_grayscale(img0),  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(img1),
            "mask0": ts_mask_0,
            "mask1": ts_mask_1,
        }

        # with torch.inference_mode():
        with torch.no_grad():
            correspondences = self.matcher(input_dict)

        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        print("number of matching pairs",mkpts0.shape)
        
        #if no matching pairs
        if mkpts0.shape[0] == 0:
            print("no matching pairs!!!!!!!!")
            return None, None, None, None
        
        Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 1.0, 0.999, 100000)
        print("number of inliers",inliers.sum())
        mkpts0 = mkpts0 * inliers
        mkpts1 = mkpts1 * inliers

        inliers0 = mkpts0[~np.all(mkpts0 == 0, axis=1)]
        inliers1 = mkpts1[~np.all(mkpts1 == 0, axis=1)]

        # inliers = inliers > 0
        return mkpts0, mkpts1, inliers0, inliers1

    #use vis3d to visualize the keypoints
    def add_kpc_to_vis3d(self, img0, img1, kpts0, kpts1):
        img0 = np.asarray(img0.cpu().numpy()[0][0]*256).astype(np.uint8)
        img1 = np.asarray(img1.cpu().numpy()[0][0]*256).astype(np.uint8)
        #vis3d = Vis3D(dump_dir, save_name)
        self.vis3d.add_keypoint_correspondences(img0, img1, kpts0, kpts1, 
                                                unmatched_kpts0 = None, unmatched_kpts1 = None, metrics = None, 
                                                booleans = None, meta = None, name = None)


if __name__ == "__main__":
    for ref_idx in range(20):
        LT = LoFTR_Test(obj_name="lamp", query_idx=0, ref_idx= ref_idx, mask_qry=True)
        img0, img1, ts_mask_0, ts_mask_1 = LT.load_img_mask(ref_idx)
        mkpts0, mkpts1, inliers0, inliers1 = LT.get_matching_result(img0, img1, ts_mask_0,ts_mask_1)
        LT.add_kpc_to_vis3d(img0, img1, inliers0, inliers1)
        print("-----------------------------------")