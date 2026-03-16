"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from kornia_moons.viz import draw_LAF_matches


def iterative_refinement_ransac(pts1, pts2, disparities1, disparities2, threshold=3.0, max_iter=30):
    for i in range(max_iter):
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=threshold)
        inliers = mask.ravel() == 1
        pts1, pts2, disparities1, disparities2 = pts1[inliers], pts2[inliers], disparities1[inliers], disparities2[inliers]
    return F, pts1, pts2, disparities1, disparities2



def compute_feature_matching_loftr(matcher, left_videos, right_videos, debug_output=None, draw_matches=False,
                            cutoff_min_disparity=None,
                            subsample_percent=None,
                            top_k=None,
                            ransac=False):
    all_mkpts0 = []
    all_mkpts1 = []
    all_horizonal_disparities = []
    all_vertical_disparities = []

    if debug_output is not None:
        os.makedirs(debug_output, exist_ok=True)

    for n in range(len(left_videos)):
        img1 = left_videos[[n]] / 255.0
        img2 = right_videos[[n]] / 255.0

        input_dict = {
            "image0": K.color.rgb_to_grayscale(img1).cuda(),
            "image1": K.color.rgb_to_grayscale(img2).cuda(),
        }

        with torch.no_grad():
            correspondences = matcher(input_dict)

        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()

        if top_k is not None:
            confidence = correspondences["confidence"].cpu().numpy()

            # Sort by confidence and select the top 100 matches
            sorted_indices = np.argsort(confidence)[::-1]  # Sort in descending order
            top_indices = sorted_indices[:top_k]  # Get indices of the top 100 matches
            mkpts0 = mkpts0[top_indices]
            mkpts1 = mkpts1[top_indices]

        dist = (mkpts0 - mkpts1)
        horizontal_disparity = dist[:, 0]
        vertical_disparity = dist[:, 1]

        all_mkpts0.append(mkpts0)
        all_mkpts1.append(mkpts1)
        all_horizonal_disparities.extend(horizontal_disparity.tolist())
        all_vertical_disparities.extend(vertical_disparity.tolist())

        # Debugging output - histograms for disparities
        if debug_output and n < 10:
            plt.hist(horizontal_disparity, bins=100)
            plt.title(f"Frame {n}: Horizontal Disparity Histogram")
            plt.savefig(f"{debug_output}/frame_{n}_horizontal_disparity.png")
            plt.clf()

            plt.hist(vertical_disparity, bins=100)
            plt.title(f"Frame {n}: Vertical Disparity Histogram")
            plt.savefig(f"{debug_output}/frame_{n}_vertical_disparity.png")
            plt.clf()

            if draw_matches:
                Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
                inliers = inliers > 0

                plt.hist(horizontal_disparity[inliers.flatten()], bins=100)
                plt.title(f"Frame {n}: Horizontal Disparity Histogram")
                plt.savefig(f"{debug_output}/frame_{n}_horizontal_disparity_inliers.png")
                plt.clf()

                plt.hist(vertical_disparity[inliers.flatten()], bins=100)
                plt.title(f"Frame {n}: Vertical Disparity Histogram")
                plt.savefig(f"{debug_output}/frame_{n}_vertical_disparity_inliers.png")
                plt.clf()

                draw_LAF_matches(
                    KF.laf_from_center_scale_ori(
                        torch.from_numpy(mkpts0).view(1, -1, 2),
                        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
                    ),
                    KF.laf_from_center_scale_ori(
                        torch.from_numpy(mkpts1).view(1, -1, 2),
                        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
                    ),
                    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
                    K.tensor_to_image(img1),
                    K.tensor_to_image(img2),
                    inliers,  # No inliers at this point
                    draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
                )
                plt.savefig(f"{debug_output}/frame_{n}_matches.png")
                plt.clf()

                draw_LAF_matches(
                    KF.laf_from_center_scale_ori(
                        torch.from_numpy(mkpts0).view(1, -1, 2),
                        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
                    ),
                    KF.laf_from_center_scale_ori(
                        torch.from_numpy(mkpts1).view(1, -1, 2),
                        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
                    ),
                    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
                    K.tensor_to_image(img1),
                    K.tensor_to_image(img2),
                    inliers,
                    draw_dict={"inlier_color": None, "tentative_color": None, "feature_color": None, "vertical": False},
                )
                plt.savefig(f"{debug_output}/frame_{n}.png")
                plt.clf()
                
    # Aggregate matches across all frames
    all_mkpts0 = np.concatenate(all_mkpts0, axis=0)
    all_mkpts1 = np.concatenate(all_mkpts1, axis=0)
    all_horizonal_disparities = np.array(all_horizonal_disparities)
    all_vertical_disparities = np.array(all_vertical_disparities)

    if subsample_percent is not None:
        total_points = all_mkpts0.shape[0]
        num_samples = int(total_points * subsample_percent)
        subsample_indices = np.random.choice(total_points, size=num_samples, replace=False)
        all_mkpts0 = all_mkpts0[subsample_indices]
        all_mkpts1 = all_mkpts1[subsample_indices]
        all_horizonal_disparities = all_horizonal_disparities[subsample_indices]
        all_vertical_disparities = all_vertical_disparities[subsample_indices]

    print("Matches across all frames", all_horizonal_disparities.shape)


    if ransac:
        Fm, all_mkpts0, all_mkpts1, all_horizonal_disparities, all_vertical_disparities = \
            iterative_refinement_ransac(all_mkpts0, all_mkpts1, all_horizonal_disparities, all_vertical_disparities)
    else:
        Fm, inliers = cv2.findFundamentalMat(all_mkpts0, all_mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        inliers = inliers > 0 if inliers is not None else np.zeros((len(all_mkpts0),), dtype=bool)
        inliers = inliers.flatten()

        all_mkpts0 = all_mkpts0[inliers]
        all_mkpts1 = all_mkpts1[inliers]
        all_horizonal_disparities = all_horizonal_disparities[inliers]
        all_vertical_disparities = all_vertical_disparities[inliers]
    print("Inliers", all_horizonal_disparities.shape)

    min_horizontal_disparity = all_horizonal_disparities.min()
    max_abs_vertical_disparity = np.abs(all_vertical_disparities).max()
    
    if cutoff_min_disparity is not None:
        # Randomly select 10,000 values to speed up sorting
        k = 10000
        if len(all_horizonal_disparities) > k:
            all_horizonal_disparities = np.random.choice(all_horizonal_disparities, size=k, replace=False)
            all_vertical_disparities = np.random.choice(all_vertical_disparities, size=k, replace=False)

        # Exclude the lowest cutoff_min_disparity of disparities
        cutoff_idx = int(len(all_horizonal_disparities) * cutoff_min_disparity)

        robust_min_horizontal_disparity = np.sort(all_horizonal_disparities)[cutoff_idx]
        robust_max_abs_vertical_disparity = np.sort(np.abs(all_vertical_disparities))[-cutoff_idx]
    else:
        robust_min_horizontal_disparity = None
        robust_max_abs_vertical_disparity = None

    return (
        all_mkpts0, all_mkpts1, Fm, 
        min_horizontal_disparity, robust_min_horizontal_disparity,
        max_abs_vertical_disparity, robust_max_abs_vertical_disparity,
    )
