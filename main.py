import numpy as np
import cv2 as cv
import random
import glob
from homography_estimation_implement import getPerspectiveTransform

def cylindrical_warp(img, K):
    h, w = img.shape[:2]
    f = K[0, 0]
    
    # Create grid of coordinates
    y_indices, x_indices = np.indices((h, w))
    x_c = x_indices - w / 2
    y_c = y_indices - h / 2
    
    # Cylindrical to Cartesian coordinates mapping (backward mapping)
    theta = x_c / f
    h_cyl = y_c / f
    
    X = np.sin(theta)
    Y = h_cyl
    Z = np.cos(theta)
    
    # Project back to 2D image coordinates
    x_img = f * X / Z + w / 2
    y_img = f * Y / Z + h / 2
    
    # Remap
    warped_img = cv.remap(img, x_img.astype(np.float32), y_img.astype(np.float32), cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    return warped_img

def evaluate_homography(H, src, dst):
    # Vectorized evaluation
    src_h = np.hstack((src, np.ones((len(src), 1))))
    p2q = (H @ src_h.T).T
    p2q = p2q[:, :2] / (p2q[:, 2:] + 1e-8)
    errors = np.linalg.norm(p2q - dst, axis=1)
    return errors

def findHomography(src, dst, n_sample, ransac_trial, ransac_threshold):
    best_score = -1
    best_model = None
    
    for _ in range(ransac_trial):
        # Step 1: Hypothesis generation
        sample_idx = random.choices(range(len(src)), k=n_sample)
        model = getPerspectiveTransform(src[sample_idx], dst[sample_idx])
        
        # Step 2: Hypothesis evaluation (Vectorized)
        errors = evaluate_homography(model, src, dst)
        score = np.sum(errors < ransac_threshold)
        
        if score > best_score:
            best_score = score
            best_model = model

    # [핵심 개선] Least Squares Refinement (전체 Inlier를 사용하여 Homography 재계산)
    errors = evaluate_homography(best_model, src, dst)
    best_inlier_mask = (errors < ransac_threshold)
    
    inlier_src = src[best_inlier_mask]
    inlier_dst = dst[best_inlier_mask]
    
    if len(inlier_src) >= 4:
        # 모든 정상 매칭점(Inlier)을 사용하여 행렬을 재계산 (노이즈 및 왜곡 극소화)
        refined_model = getPerspectiveTransform(inlier_src, inlier_dst)
        refined_errors = evaluate_homography(refined_model, src, dst)
        final_inlier_mask = (refined_errors < ransac_threshold).astype(np.uint8)
        return refined_model, final_inlier_mask

    return best_model, best_inlier_mask.astype(np.uint8)

def vectorized_warp(src, src_alpha, H, dst_size):
    h_dst, w_dst = dst_size
    y_indices, x_indices = np.indices((h_dst, w_dst))
    coords = np.stack([x_indices.ravel(), y_indices.ravel(), np.ones_like(x_indices).ravel()])
    
    H_inv = np.linalg.inv(H)
    p = H_inv @ coords
    p = p[:2] / (p[2:] + 1e-8)
    
    px = np.round(p[0]).astype(np.int32)
    py = np.round(p[1]).astype(np.int32)
    
    valid_mask = (px >= 0) & (py >= 0) & (px < src.shape[1]) & (py < src.shape[0])
    
    dst_img = np.zeros((h_dst, w_dst, 3), dtype=src.dtype)
    dst_alpha = np.zeros((h_dst, w_dst, 1), dtype=src_alpha.dtype)
    
    dst_y = y_indices.ravel()[valid_mask]
    dst_x = x_indices.ravel()[valid_mask]
    src_y = py[valid_mask]
    src_x = px[valid_mask]
    
    dst_img[dst_y, dst_x] = src[src_y, src_x]
    dst_alpha[dst_y, dst_x] = src_alpha[src_y, src_x]
    
    return dst_img, dst_alpha

if __name__ == '__main__':
    # Load all images in data folder
    img_paths = sorted(glob.glob('./data/IMG_*.JPG'))
    images = [cv.imread(p) for p in img_paths if cv.imread(p) is not None]
    
    if len(images) < 2:
        print("Not enough images to stitch.")
        exit()

    print(f"Loaded {len(images)} images.")

    # Cylindrical Projection
    f = (images[0].shape[1] + images[0].shape[0]) / 2 
    K = np.array([[f, 0, images[0].shape[1]/2], [0, f, images[0].shape[0]/2], [0, 0, 1]])
    
    cyl_images = []
    cyl_alphas = []
    
    # 1D Linear Alpha Mask for Feather Blending
    for img in images:
        cyl_img = cylindrical_warp(img, K)
        cyl_images.append(cyl_img)
        
        h, w = cyl_img.shape[:2]
        X = np.linspace(-1, 1, w)
        # X는 -1 ~ 1. 중심으로부터의 거리 역산. 중심에서 1, 가장자리에서 0
        mask_1d = 1.0 - np.abs(X)
        mask_2d = np.tile(mask_1d, (h, 1)).astype(np.float32)
        
        # 원통형 투영 시 생긴 검은 빈 공간(Black Padding)을 마스크에서 제외하여 검은 선이 생기지 않도록 방지
        valid_mask = (cyl_img.sum(axis=2) > 0).astype(np.float32)
        alpha_mask = mask_2d[..., np.newaxis] * valid_mask[..., np.newaxis]
        
        cyl_alphas.append(alpha_mask)

    fdetector = cv.BRISK_create()
    fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
    
    # Prepare Large Canvas
    h, w = cyl_images[0].shape[:2]
    canvas_w = w * len(cyl_images)
    canvas_h = h * 2  # 여유 공간
    
    pano = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    pano_alpha = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    # 초기 변환 행렬: 첫 이미지를 캔버스의 적당한 여백(약 1/3 지점)에 배치
    H_global = np.array([[1.0, 0.0, canvas_w // 3],
                         [0.0, 1.0, canvas_h // 2 - h // 2],
                         [0.0, 0.0, 1.0]])

    print("Stitching image 1...")
    pano, pano_alpha = vectorized_warp(cyl_images[0], cyl_alphas[0], H_global, (canvas_h, canvas_w))

    for i in range(1, len(cyl_images)):
        print(f"Stitching image {i+1}/{len(cyl_images)}...")
        img_prev = cyl_images[i-1]
        img_curr = cyl_images[i]
        
        # Retrieve matching points
        kp1, des1 = fdetector.detectAndCompute(img_prev, None)
        kp2, des2 = fdetector.detectAndCompute(img_curr, None)
        
        knn_matches = fmatcher.knnMatch(des1, des2, k=2)
        
        pts1, pts2 = [], []
        good_matches = []
        for m, n in knn_matches:
            if m.distance < 0.75 * n.distance:
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
                good_matches.append(m)
                
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)

        # Calculate local homography
        H_local, inlier_mask = findHomography(pts2, pts1, 4, 200, 2)
        
        # Update global homography
        H_global = H_global @ H_local
        
        # Warp current image
        warped_curr, warped_alpha = vectorized_warp(img_curr, cyl_alphas[i], H_global, (canvas_h, canvas_w))
        
        # ----------------------------------------------------
        # [핵심 개선] Center-weighted Voronoi Seam Blending
        # ----------------------------------------------------
        # 1. 두 이미지가 겹치는 구역(Overlap) 마스크 생성
        overlap_mask = (pano_alpha > 0) & (warped_alpha > 0)
        
        # 2. Voronoi Seam 마스크 생성 (새 이미지가 중심에 더 가까운 부분은 1.0)
        seam_mask = np.zeros_like(pano_alpha)
        seam_mask[warped_alpha > pano_alpha] = 1.0
        
        # 3. 경계선을 부드럽게 만들기 위해 가우시안 블러(Gaussian Blur) 적용 (약 31픽셀 너비의 부드러운 전환)
        blend_mask = cv.GaussianBlur(seam_mask, (31, 31), 0)
        if blend_mask.ndim == 2:
            blend_mask = blend_mask[..., np.newaxis]
            
        # 4. 최종 합성용 가중치 마스크(M) 생성
        M = np.zeros_like(pano_alpha)
        
        # (1) 새 이미지만 있는 구역
        M[(pano_alpha == 0) & (warped_alpha > 0)] = 1.0
        # (2) 겹치는 구역은 부드러운 Seam(blend_mask) 적용
        M[overlap_mask] = blend_mask[overlap_mask]
        # (3) 이전 이미지만 있는 구역은 M=0 (초기값 그대로)
        
        pano_float = pano.astype(np.float32) * (1.0 - M) + warped_curr.astype(np.float32) * M
        pano = pano_float.astype(np.uint8)
        
        # 거리 맵(Alpha)은 항상 최대값(중심에 가까운 값)으로 업데이트
        pano_alpha = np.maximum(pano_alpha, warped_alpha)
        # ----------------------------------------------------

        # ---------------------------------------------
        # 3단계 시각화 및 저장 (Original -> Matches -> Merged)
        # ---------------------------------------------
        row1 = np.hstack((img_prev, img_curr))
        
        img_matched = cv.drawMatches(img_prev, kp1, img_curr, kp2, good_matches, None, None, None, matchesMask=inlier_mask.tolist())
        row2 = img_matched
        
        # 현재까지의 파노라마 크롭
        non_black_step = np.argwhere(pano_alpha > 0)
        (y_min_s, x_min_s, _) = non_black_step.min(axis=0)
        (y_max_s, x_max_s, _) = non_black_step.max(axis=0)
        pano_step = pano[y_min_s:y_max_s+1, x_min_s:x_max_s+1]
        
        # 3단계를 가로 크기에 맞춰 합치기
        target_w = row1.shape[1]
        scale_ratio = target_w / pano_step.shape[1]
        row3_resized = cv.resize(pano_step, (target_w, int(pano_step.shape[0] * scale_ratio)))
        
        step_vis = np.vstack((row1, row2, row3_resized))
        
        # 각각의 사진 및 통합 사진 저장
        cv.imwrite(f'./data/step{i}_1_original.jpg', row1)
        cv.imwrite(f'./data/step{i}_2_matches.jpg', row2)
        cv.imwrite(f'./data/step{i}_3_merged.jpg', pano_step)
        cv.imwrite(f'./data/step{i}_all_visualization.jpg', step_vis)
        
        # 화면 출력 (크기 조정)
        disp_vis = cv.resize(step_vis, (0,0), fx=1200/target_w, fy=1200/target_w)
        wnd_name = f'Step {i} Stitching Process (Press any key for next)'
        cv.imshow(wnd_name, disp_vis)
        cv.waitKey(0)
        cv.destroyWindow(wnd_name)
        cv.waitKey(1)

    # Crop black margins (Outer Bounding Box)
    non_black = np.argwhere(pano_alpha > 0)
    if len(non_black) > 0:
        (y_min, x_min, _) = non_black.min(axis=0)
        (y_max, x_max, _) = non_black.max(axis=0)
        pano_cropped = pano[y_min:y_max+1, x_min:x_max+1]
    else:
        pano_cropped = pano

    # Inner Bounding Box Crop: 상하좌우 지그재그로 어긋나서 생긴 검은 빈 공간을 완전히 잘라내기
    gray = cv.cvtColor(pano_cropped, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    top, bottom, left, right = 0, thresh.shape[0]-1, 0, thresh.shape[1]-1
    
    while top < bottom and left < right:
        t_black = np.any(thresh[top, left:right+1] == 0)
        b_black = np.any(thresh[bottom, left:right+1] == 0)
        l_black = np.any(thresh[top:bottom+1, left] == 0)
        r_black = np.any(thresh[top:bottom+1, right] == 0)
        
        if not (t_black or b_black or l_black or r_black):
            break
            
        if t_black: top += 1
        if b_black: bottom -= 1
        if l_black: left += 1
        if r_black: right -= 1
        
    pano_final = pano_cropped[top:bottom+1, left:right+1]

    # Save and Show the merged image
    cv.imwrite('./data/panorama_result.jpg', pano_final)
    
    disp_pano = cv.resize(pano_final, (0,0), fx=1200/pano_final.shape[1], fy=1200/pano_final.shape[1] * (pano_final.shape[0]/pano_final.shape[1]))
    cv.imshow(f'Multi-Image Panorama ({len(cyl_images)} images)', disp_pano)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1) 