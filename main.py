import numpy as np
import cv2 as cv
import random
import glob
from homography_estimation_implement import getPerspectiveTransform

def auto_order_images(images):
    n = len(images)
    if n <= 2:
        return list(range(n))
        
    fdetector = cv.BRISK_create()
    fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
    
    dess = []
    for img in images:
        scale = 800.0 / max(img.shape[0], img.shape[1])
        small_img = cv.resize(img, (0, 0), fx=scale, fy=scale)
        _, des = fdetector.detectAndCompute(small_img, None)
        dess.append(des)
        
    match_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if dess[i] is None or dess[j] is None:
                continue
            knn_matches = fmatcher.knnMatch(dess[i], dess[j], k=2)
            good = 0
            for m_obj, n_obj in knn_matches:
                if m_obj.distance < 0.75 * n_obj.distance:
                    good += 1
            match_matrix[i, j] = good
            match_matrix[j, i] = good
            
    visited = set()
    i, j = np.unravel_index(np.argmax(match_matrix), match_matrix.shape)
    path = [int(i), int(j)]
    visited.add(int(i))
    visited.add(int(j))
    
    while len(visited) < n:
        left, right = path[0], path[-1]
        max_l_val, max_l_node = -1, -1
        max_r_val, max_r_node = -1, -1
        
        for v in range(n):
            if v not in visited:
                if match_matrix[v, left] > max_l_val:
                    max_l_val = match_matrix[v, left]
                    max_l_node = v
                if match_matrix[right, v] > max_r_val:
                    max_r_val = match_matrix[right, v]
                    max_r_node = v
                    
        if max_l_val > max_r_val:
            path.insert(0, max_l_node)
            visited.add(max_l_node)
        else:
            path.append(max_r_node)
            visited.add(max_r_node)
            
    return path

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
    h_src, w_src = src.shape[:2]
    
    # 1. 원본 이미지의 4개 꼭지점을 H로 변환하여 대상 이미지에서의 Bounding Box를 구합니다.
    corners = np.array([[0, 0, 1],
                        [w_src-1, 0, 1],
                        [w_src-1, h_src-1, 1],
                        [0, h_src-1, 1]]).T
    warped_corners = H @ corners
    warped_corners = warped_corners[:2] / (warped_corners[2] + 1e-8)
    
    min_x = max(0, int(np.floor(np.min(warped_corners[0]))))
    max_x = min(w_dst - 1, int(np.ceil(np.max(warped_corners[0]))))
    min_y = max(0, int(np.floor(np.min(warped_corners[1]))))
    max_y = min(h_dst - 1, int(np.ceil(np.max(warped_corners[1]))))
    
    # Bounding Box가 화면 밖이면 빈 이미지 반환
    if min_x > max_x or min_y > max_y:
        return np.zeros((h_dst, w_dst, 3), dtype=src.dtype), np.zeros((h_dst, w_dst, 1), dtype=src_alpha.dtype)
        
    # 2. Bounding Box 영역에 대해서만 좌표 그리드 생성 (연산량 수백 배 감소)
    y_indices, x_indices = np.mgrid[min_y:max_y+1, min_x:max_x+1]
    
    coords = np.stack([x_indices.ravel(), y_indices.ravel(), np.ones_like(x_indices).ravel()])
    
    H_inv = np.linalg.inv(H)
    p = H_inv @ coords
    p = p[:2] / (p[2:] + 1e-8)
    
    safe_p = np.isfinite(p[0]) & np.isfinite(p[1]) & (p[0] > -1e6) & (p[0] < 1e6) & (p[1] > -1e6) & (p[1] < 1e6)
    
    px = np.full_like(p[0], -1, dtype=np.int32)
    py = np.full_like(p[1], -1, dtype=np.int32)
    
    px[safe_p] = np.round(p[0][safe_p]).astype(np.int32)
    py[safe_p] = np.round(p[1][safe_p]).astype(np.int32)
    
    valid_mask = safe_p & (px >= 0) & (py >= 0) & (px < w_src) & (py < h_src)
    
    dst_img = np.zeros((h_dst, w_dst, 3), dtype=src.dtype)
    dst_alpha = np.zeros((h_dst, w_dst, 1), dtype=src_alpha.dtype)
    
    # Bounding Box 내의 좌표를 전체 이미지 좌표로 변환
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
    import cv2 as cv
    raw_images = [cv.imread(p) for p in img_paths if cv.imread(p) is not None]
    
    images = []
    for img in raw_images:
        max_dim = 1600.0
        scale = max_dim / max(img.shape[:2])
        if scale < 1.0:
            images.append(cv.resize(img, (0, 0), fx=scale, fy=scale))
        else:
            images.append(img)
            
    if len(images) < 2:
        print("Not enough images to stitch.")
        exit()

    print("Checking and auto-ordering images by feature matching...")
    order = auto_order_images(images)
    images = [images[i] for i in order]
    img_paths = [img_paths[i] for i in order]
    print(f"Ordered indices: {order}")

    print(f"Loaded {len(images)} images.")

    # ----------------------------------------------------
    # Planar Projection (과제 요구사항에 맞춰 원본 평면 이미지 사용)
    # ----------------------------------------------------
    proj_images = []
    proj_alphas = []
    
    # 1D Linear Alpha Mask for Feather Blending
    for img in images:
        proj_images.append(img)
        
        h, w = img.shape[:2]
        import numpy as np
        X = np.linspace(-1, 1, w)
        # X는 -1 ~ 1. 중심으로부터의 거리 역산. 중심에서 1, 가장자리에서 0
        mask_1d = 1.0 - np.abs(X)
        mask_2d = np.tile(mask_1d, (h, 1)).astype(np.float32)
        
        valid_mask = (img.sum(axis=2) > 0).astype(np.float32)
        alpha_mask = mask_2d[..., np.newaxis] * valid_mask[..., np.newaxis]
        proj_alphas.append(alpha_mask)

    print("Pass 1: Estimating Homographies (Sequential)...")
    fdetector = cv.BRISK_create()
    fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
    
    def extract_features(img, detector, max_dim=1200.0):
        scale = max_dim / max(img.shape[:2])
        if scale < 1.0:
            small_img = cv.resize(img, (0, 0), fx=scale, fy=scale)
            kp, des = detector.detectAndCompute(small_img, None)
            for k in kp:
                k.pt = (k.pt[0] / scale, k.pt[1] / scale)
            return kp, des
        return detector.detectAndCompute(img, None)
        
    kps_list = []
    kp, des = extract_features(proj_images[0], fdetector)
    kps_list.append((kp, des))
    
    H_globals_0 = [np.eye(3, dtype=np.float64)]
    matches_list = []
    inlier_masks = []
    
    for i in range(1, len(proj_images)):
        print(f"Matching image {i} to {i-1}...")
        img_prev = proj_images[i-1]
        img_curr = proj_images[i]
        
        kp2, des2 = extract_features(img_curr, fdetector)
        kps_list.append((kp2, des2))
        
        kp1, des1 = kps_list[i-1]
        
        knn_matches = fmatcher.knnMatch(des1, des2, k=2)
        pts1, pts2, good_matches = [], [], []
        for m, n in knn_matches:
            if m.distance < 0.75 * n.distance:
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
                good_matches.append(m)
                
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)
        
        H_local, inlier_mask = findHomography(pts2, pts1, 4, 200, 2)
        matches_list.append(good_matches)
        inlier_masks.append(inlier_mask)
        H_global = H_globals_0[-1] @ H_local
        H_globals_0.append(H_global)

    # ----------------------------------------------------
    # Pass 2: Center Reference Adjustment
    # ----------------------------------------------------
    print("Pass 2: Adjusting matrices to Center Reference...")
    c = len(proj_images) // 2
    H_center_inv = np.linalg.inv(H_globals_0[c])
    
    # Prepare Large Canvas
    h, w = proj_images[0].shape[:2]
    canvas_w = int(w * len(proj_images) * 1.5)  # 평면 뷰는 좌우로 넓게 퍼지므로 여백 넉넉히
    canvas_h = int(h * 3.5)                     # 상하로도 늘어남
    
    T_canvas = np.array([[1.0, 0.0, canvas_w // 2 - w // 2],
                         [0.0, 1.0, canvas_h // 2 - h // 2],
                         [0.0, 0.0, 1.0]])
                         
    H_finals = []
    for H in H_globals_0:
        H_finals.append(T_canvas @ H_center_inv @ H)
        
    pano = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    pano_alpha = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    # ----------------------------------------------------
    # Pass 3: Warping and Blending
    # ----------------------------------------------------
    print("Pass 3: Warping and Blending...")
    print("Stitching image 1...")
    pano, pano_alpha = vectorized_warp(proj_images[0], proj_alphas[0], H_finals[0], (canvas_h, canvas_w))
    
    for i in range(1, len(proj_images)):
        print(f"Stitching image {i+1}/{len(proj_images)}...")
        img_prev = proj_images[i-1]
        img_curr = proj_images[i]
        
        warped_curr, warped_alpha = vectorized_warp(img_curr, proj_alphas[i], H_finals[i], (canvas_h, canvas_w))
        
        # [핵심 개선] Gain Compensation & Center-weighted Voronoi Seam Blending
        overlap_mask = (pano_alpha > 0) & (warped_alpha > 0)
        
        # Gain Compensation (밝기 보정)
        overlap_bool = overlap_mask.squeeze()
        if np.any(overlap_bool):
            mean_pano = pano[overlap_bool].mean(axis=0)
            mean_curr = warped_curr[overlap_bool].mean(axis=0)
            mean_curr[mean_curr == 0] = 1.0
            gain = mean_pano / mean_curr
            warped_curr_float = warped_curr.astype(np.float32) * gain
            warped_curr = np.clip(warped_curr_float, 0, 255).astype(np.uint8)
        
        seam_mask = np.zeros_like(pano_alpha)
        seam_mask[warped_alpha > pano_alpha] = 1.0
        
        # Bounding box blur optimization (수백 배 속도 향상)
        y_nz, x_nz = np.nonzero(warped_alpha.squeeze())
        blend_mask = np.zeros_like(seam_mask)
        
        if len(y_nz) > 0:
            pad = 200
            y_min = max(0, y_nz.min() - pad)
            y_max = min(canvas_h - 1, y_nz.max() + pad)
            x_min = max(0, x_nz.min() - pad)
            x_max = min(canvas_w - 1, x_nz.max() + pad)
            
            sub_mask = seam_mask[y_min:y_max+1, x_min:x_max+1]
            blurred = cv.GaussianBlur(sub_mask, (101, 101), 0)
            if blurred.ndim == 2: blurred = blurred[..., np.newaxis]
            blend_mask[y_min:y_max+1, x_min:x_max+1] = blurred
        else:
            blend_mask = seam_mask.copy()
            
        M = np.zeros_like(pano_alpha)
        M[(pano_alpha == 0) & (warped_alpha > 0)] = 1.0
        M[overlap_mask] = blend_mask[overlap_mask]
        
        pano_float = pano.astype(np.float32) * (1.0 - M) + warped_curr.astype(np.float32) * M
        pano = pano_float.astype(np.uint8)
        pano_alpha = np.maximum(pano_alpha, warped_alpha)
        
        # ---------------------------------------------
        # 시각화 및 저장
        # ---------------------------------------------
        row1 = np.hstack((img_prev, img_curr))
        kp1 = kps_list[i-1][0]
        kp2 = kps_list[i][0]
        good_matches = matches_list[i-1]
        inlier_mask = inlier_masks[i-1]
        img_matched = cv.drawMatches(img_prev, kp1, img_curr, kp2, good_matches, None, None, None, matchesMask=inlier_mask.tolist())
        row2 = img_matched
        
        non_black = np.argwhere(pano_alpha > 0)
        if len(non_black) > 0:
            (y_min, x_min, _) = non_black.min(axis=0)
            (y_max, x_max, _) = non_black.max(axis=0)
            pano_step = pano[y_min:y_max+1, x_min:x_max+1]
        else:
            pano_step = pano.copy()
            
        target_w = row2.shape[1]
        if pano_step.shape[1] > 0:
            scale = target_w / pano_step.shape[1]
            pano_step_resized = cv.resize(pano_step, (target_w, max(1, int(pano_step.shape[0] * scale))))
        else:
            pano_step_resized = np.zeros((100, target_w, 3), dtype=np.uint8)
            
        step_vis = np.vstack((row1, row2, pano_step_resized))
        
        cv.imwrite(f'./data/step{i}_1_original.jpg', row1)
        cv.imwrite(f'./data/step{i}_2_matches.jpg', row2)
        cv.imwrite(f'./data/step{i}_3_merged.jpg', pano_step)
        cv.imwrite(f'./data/step{i}_all_visualization.jpg', step_vis)
        
        disp_vis = cv.resize(step_vis, (0,0), fx=1200/target_w, fy=1200/target_w)
        cv.imshow(f'Step {i} Stitching Process', disp_vis)
        cv.waitKey(1)
        cv.destroyAllWindows()
        
    # ----------------------------------------------------
    # 최종 크롭 및 저장
    # ----------------------------------------------------
    non_black = np.argwhere(pano_alpha > 0)
    (y_min, x_min, _) = non_black.min(axis=0)
    (y_max, x_max, _) = non_black.max(axis=0)
    pano_cropped = pano[y_min:y_max+1, x_min:x_max+1]
    
    gray = cv.cvtColor(pano_cropped, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    x, y, w, h = cv.boundingRect(contours[0])
    
    top, bottom, left, right = y, y+h-1, x, x+w-1
    
    while top < bottom:
        if np.any(pano_cropped[top, left:right+1] == 0): top += 1
        else: break
    while bottom > top:
        if np.any(pano_cropped[bottom, left:right+1] == 0): bottom -= 1
        else: break
    while left < right:
        if np.any(pano_cropped[top:bottom+1, left] == 0): left += 1
        else: break
    while right > left:
        if np.any(pano_cropped[top:bottom+1, right] == 0): right -= 1
        else: break
        
    if top < bottom and left < right:
        pano_final = pano_cropped[top:bottom+1, left:right+1]
    else:
        print("Warning: Inner crop failed. Using outer bounding box.")
        pano_final = pano_cropped

    if pano_final.size == 0 or pano_final.shape[0] == 0 or pano_final.shape[1] == 0:
        print("Error: Final panorama is empty. Aborting.")
        exit()

    cv.imwrite('./data/panorama_result.jpg', pano_final)
    
    if pano_final.shape[1] > 0:
        disp_pano = cv.resize(pano_final, (1200, max(1, int(1200 * pano_final.shape[0] / pano_final.shape[1]))))
        cv.imshow(f'Multi-Image Panorama ({len(proj_images)} images)', disp_pano)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)
