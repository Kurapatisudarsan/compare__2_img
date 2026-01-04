import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModel
from timm.models.vision_transformer import VisionTransformer

# --- Model Loading (Cached to prevent reloading) ---
@st.cache_resource
def load_dino_base_model():
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    return processor, model

@st.cache_resource
def load_dino_large_model():
    st.info("Loading DINOv3 model... this is a large file and may take time on the first run.")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large")
    return processor, model

@st.cache_resource
def load_clip_model():
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model

# --- Matching Functions ---
def match_with_dino(img1, img2, processor, model):
    # ... (function is unchanged)
    inputs1 = processor(images=img1, return_tensors="pt"); inputs2 = processor(images=img2, return_tensors="pt")
    with torch.no_grad():
        outputs1 = model(**inputs1); outputs2 = model(**inputs2)
    features1 = outputs1.last_hidden_state[:, 1:, :].squeeze(0); features2 = outputs2.last_hidden_state[:, 1:, :].squeeze(0)
    h, w, _ = img1.shape; patch_size = model.config.patch_size
    num_patches_h = h // patch_size; num_patches_w = w // patch_size
    kp1 = [cv2.KeyPoint(float(j*patch_size + patch_size/2), float(i*patch_size + patch_size/2), patch_size) for i in range(num_patches_h) for j in range(num_patches_w)]
    kp2 = [cv2.KeyPoint(float(j*patch_size + patch_size/2), float(i*patch_size + patch_size/2), patch_size) for i in range(num_patches_h) for j in range(num_patches_w)]
    bf = cv2.BFMatcher(cv2.NORM_L2); matches = bf.knnMatch(features1.cpu().numpy(), features2.cpu().numpy(), k=2)
    good_initial_matches = [m for m, n in matches if m.distance < 0.9 * n.distance]
    ransac_matches_count = 0; inlier_ratio = 0.0; vis_img = None
    if len(good_initial_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_initial_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_initial_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist(); ransac_matches_count = np.sum(matches_mask)
        inlier_ratio = ransac_matches_count / len(good_initial_matches) if len(good_initial_matches) > 0 else 0
        vis_img = cv2.drawMatches(img1, kp1, img2, kp2, good_initial_matches, None, matchColor=(-1), matchesMask=matches_mask, flags=2)
    else: vis_img = cv2.drawMatches(img1, [], img2, [], [], None, flags=2)
    confidence = min(100.0, (ransac_matches_count / 200.0) * 100.0)
    return {"KP1": len(kp1), "KP2": len(kp2), "Initial Matches": len(good_initial_matches), "RANSAC Matches": ransac_matches_count, "Inlier Ratio": inlier_ratio, "Confidence": confidence}, vis_img

def get_clip_similarity(img1, img2, processor, model):
    # ... (function is unchanged)
    inputs1 = processor(images=img1, return_tensors="pt"); inputs2 = processor(images=img2, return_tensors="pt")
    with torch.no_grad():
        features1 = model.get_image_features(**inputs1); features2 = model.get_image_features(**inputs2)
    features1 /= features1.norm(p=2, dim=-1, keepdim=True); features2 /= features2.norm(p=2, dim=-1, keepdim=True)
    similarity = (features1 @ features2.T).item()
    return {"KP1": "N/A", "KP2": "N/A", "Initial Matches": "N/A", "RANSAC Matches": "N/A", "Inlier Ratio": similarity, "Confidence": similarity * 100}, None

def get_matches_with_ransac(detector, img1, img2, custom_descriptor=None):
    """Finds matches using classical detectors and RANSAC. Now handles separate detector/descriptor."""
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY); img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if custom_descriptor:
        kp1 = detector.detect(img1_gray, None); kp2 = detector.detect(img2_gray, None)
        kp1, des1 = custom_descriptor.compute(img1_gray, kp1); kp2, des2 = custom_descriptor.compute(img2_gray, kp2)
        matcher_norm = cv2.NORM_HAMMING
    else:
        kp1, des1 = detector.detectAndCompute(img1_gray, None); kp2, des2 = detector.detectAndCompute(img2_gray, None)
        matcher_norm = cv2.NORM_L2 if des1 is not None and des1.dtype == np.float32 else cv2.NORM_HAMMING

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return {"KP1": len(kp1) if 'kp1' in locals() else 0, "KP2": len(kp2) if 'kp2' in locals() else 0, "Initial Matches": 0, "RANSAC Matches": 0, "Inlier Ratio": 0, "Confidence": 0}, None
    
    bf = cv2.BFMatcher(matcher_norm, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_initial_matches = []
    if matches and all(isinstance(m, tuple) and len(m) == 2 for m in matches):
        for m, n in matches:
            if m.distance < 0.75 * n.distance: good_initial_matches.append(m)
    initial_matches_count = len(good_initial_matches); ransac_matches_count = 0; inlier_ratio = 0.0; vis_img = None
    if initial_matches_count > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_initial_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_initial_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist(); ransac_matches_count = np.sum(matches_mask)
        inlier_ratio = ransac_matches_count / initial_matches_count if initial_matches_count > 0 else 0
        vis_img = cv2.drawMatches(img1, kp1, img2, kp2, good_initial_matches, None, matchColor=(-1), singlePointColor=None, matchesMask=matches_mask, flags=2)
    else: vis_img = cv2.drawMatches(img1, kp1, img2, kp2, [], None, flags=2)
    confidence = min(100.0, (ransac_matches_count / 200.0) * 100.0)
    return {"KP1": len(kp1), "KP2": len(kp2), "Initial Matches": initial_matches_count, "RANSAC Matches": ransac_matches_count, "Inlier Ratio": inlier_ratio, "Confidence": confidence}, vis_img

# --- Streamlit User Interface ---
st.set_page_config(page_title="Ultimate Object Verifier", page_icon="✨", layout="wide")
st.title("✨ Ultimate Object Verification Tool")
st.write("Comparing classical and modern AI algorithms for image matching.")

# Load models
dino_base_processor, dino_base_model = load_dino_base_model()
dino_large_processor, dino_large_model = load_dino_large_model()
clip_processor, clip_model = load_clip_model()

# --- CHANGE: Removed FREAK from this dictionary ---
classical_detectors = { "ORB": cv2.ORB_create(nfeatures=1000), "SIFT": cv2.SIFT_create(nfeatures=1000),
                        "BRISK": cv2.BRISK_create(), "AKAZE": cv2.AKAZE_create(), "KAZE": cv2.KAZE_create() }
# --- CHANGE: Created separate detectors and descriptors for special cases ---
gftt_detector = cv2.GFTTDetector_create()
brief_descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
fast_detector = cv2.FastFeatureDetector_create()
freak_descriptor = cv2.xfeatures2d.FREAK_create()


col1, col2 = st.columns(2)
with col1: st.header("First Image"); uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "jpeg", "png"], key="img1")
with col2: st.header("Second Image"); uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "jpeg", "png"], key="img2")

if uploaded_file1 and uploaded_file2:
    image1 = Image.open(uploaded_file1).convert('RGB'); image2 = Image.open(uploaded_file2).convert('RGB')
    img1_cv = np.array(image1); img2_cv = np.array(image2)
    st.image([image1, image2], caption=["First Image", "Second Image"], width=400)

    if st.button("Run Full Comparison", type="primary"):
        all_results = []; visualization_images = {}
        with st.spinner("Analyzing with all algorithms... this can take a moment."):
            # Run classical detectors
            for name, detector in classical_detectors.items():
                results, vis_img = get_matches_with_ransac(detector, img1_cv.copy(), img2_cv.copy()); results["Method"] = name
                all_results.append(results); visualization_images[name] = vis_img

            # Run GFTT+BRIEF
            gftt_results, gftt_vis = get_matches_with_ransac(gftt_detector, img1_cv.copy(), img2_cv.copy(), custom_descriptor=brief_descriptor)
            gftt_results["Method"] = "GFTT+BRIEF"; all_results.append(gftt_results); visualization_images["GFTT+BRIEF"] = gftt_vis

            # --- NEW: Run FAST+FREAK ---
            freak_results, freak_vis = get_matches_with_ransac(fast_detector, img1_cv.copy(), img2_cv.copy(), custom_descriptor=freak_descriptor)
            freak_results["Method"] = "FAST+FREAK"; all_results.append(freak_results); visualization_images["FAST+FREAK"] = freak_vis
            
            # Run DINO models
            dino_base_results, dino_base_vis = match_with_dino(img1_cv.copy(), img2_cv.copy(), dino_base_processor, dino_base_model)
            dino_base_results["Method"] = "DINOv2"; all_results.append(dino_base_results); visualization_images["DINOv2"] = dino_base_vis
            dino_large_results, dino_large_vis = match_with_dino(img1_cv.copy(), img2_cv.copy(), dino_large_processor, dino_large_model)
            dino_large_results["Method"] = "DINOv3"; all_results.append(dino_large_results); visualization_images["DINOv3"] = dino_large_vis

            # Run CLIP
            clip_results, _ = get_clip_similarity(img1_cv.copy(), img2_cv.copy(), clip_processor, clip_model); clip_results["Method"] = "CLIP"
            all_results.append(clip_results); visualization_images["CLIP"] = None

        keypoint_methods_results = [res for res in all_results if res["RANSAC Matches"] != "N/A"]
        if keypoint_methods_results:
             best_result = max(keypoint_methods_results, key=lambda x: x['RANSAC Matches'])
             st.subheader("🏆 Final Verdict")
             st.info(f"The most reliable match was found by the **{best_result['Method']}** algorithm.")
             final_col1, final_col2 = st.columns(2)
             with final_col1: st.metric("Final Similarity Score", f"{best_result['Inlier Ratio']:.2%}")
             with final_col2: st.metric("Reliable Matches Found", f"{best_result['RANSAC Matches']}")
        else:
            best_result = None
            st.warning("No reliable keypoint matches found by any method.")

        df = pd.DataFrame(all_results)
        df['Inlier Ratio'] = df['Inlier Ratio'].apply(lambda x: f"{x:.2%}" if isinstance(x, float) else "N/A")
        df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.1f}%" if isinstance(x, float) else "N/A")
        
        df = df[["Method", "KP1", "KP2", "Initial Matches", "RANSAC Matches", "Inlier Ratio", "Confidence"]]
        st.subheader("Detailed Comparison Results"); st.dataframe(df.set_index('Method'))
        
        if best_result:
            st.subheader("Best Match Visualization")
            best_vis_img = visualization_images[best_result['Method']]
            if best_vis_img is not None:
                st.image(best_vis_img, caption=f"Most reliable matches as found by {best_result['Method']}", use_column_width=True)
            else:
                st.warning("Visualization is not available for the selected method.")