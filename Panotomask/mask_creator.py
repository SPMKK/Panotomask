import math
import os
import cv2
import argparse
import numpy as np
from PIL import Image, ImageOps
import torch
from ultralytics import YOLO
from sklearn.cluster import KMeans


class DominantColorExtractor:
    def __init__(self, rgb):
        self.rgb = rgb

    # def extract_from_box(self, image, box):
    #     """
    #     Trích xuất màu chủ đạo từ bounding box.
    #     Lọc bỏ:
    #     - Màu đen (R, G, B < 30)
    #     - Màu trắng (R, G, B > 225)
    #     - Màu xám nhạt (R, G, B > 125 tất cả)
    #     """
    #     W, H = image.size
    #     x, y, w, h = box
    #     left = int((x - w / 2) * W)
    #     top = int((y - h / 2) * H)
    #     right = int((x + w / 2) * W)
    #     bottom = int((y + h / 2) * H)

    #     cropped = image.crop((left, top, right, bottom))
    #     img_np = np.array(cropped).reshape(-1, 3)

    #     if len(img_np) < self.n_colors:
    #         return [(0, 0, 0)]

    #     kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init='auto')
    #     kmeans.fit(img_np)
    #     colors = kmeans.cluster_centers_.astype(int)

    #     def is_invalid(rgb):
    #         r, g, b = rgb
    #         return (
    #             (r < 30 and g < 30 and b < 30) or       # đen
    #             (r > 225 and g > 225 and b > 225) or    # trắng
    #             (r > 100 and g > 100 and b > 100)       # xám nhạt
    #         )

    #     filtered = [tuple(c) for c in colors if not is_invalid(c)]

    #     if not filtered:
    #         return [tuple(colors[0])]  # fallback: vẫn trả về 1 màu nếu bị loại hết

    #     return filtered



    def get_masked_image(self,image, tolerance=50, blur_ksize=9):
        """
        Dùng dominant color từ bbox để lọc toàn ảnh:
        Giữ lại các pixel toàn ảnh có màu gần với dominant color, các pixel còn lại chuyển thành trắng.
        """
        img_np = np.array(image).copy()

        # # Lấy dominant color từ bbox
        # dominant_color = self.extract_from_box(image, box)[0]

        # Tính khoảng cách màu cho toàn ảnh
        dist = np.linalg.norm(img_np - np.array(self.rgb), axis=2)
        mask = dist < tolerance  # giữ pixel gần màu chủ đạo

        mask = mask.astype(np.uint8) * 255
        mask_blur = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
        mask_blur = mask_blur.astype(np.float32) / 255.0  # đưa về [0,1] để blend

        # Tạo ảnh trắng và gán pixel gốc vào vùng mask
        new_img = np.ones_like(img_np, dtype=np.float32) * 255
        new_img = img_np * mask_blur[..., None] + new_img * (1 - mask_blur[..., None])
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)

        return Image.fromarray(new_img)

class PanoramaProcessor:
    def __init__(self, device='cuda'):
        self.device = device

    def _panorama_to_plane(self, panorama_tensor, FOV, output_size, yaw, pitch):
        pano_tensor = panorama_tensor.to(self.device)
        pano_tensor = pano_tensor.permute(2, 0, 1).float() / 255  # [3, H, W]
        pano_h, pano_w = pano_tensor.shape[1:]

        W, H = output_size
        f = (0.5 * W) / np.tan(np.radians(FOV) / 2)
        yaw_r, pitch_r = np.radians(yaw), np.radians(pitch)

        u, v = torch.meshgrid(torch.arange(W, device=self.device), torch.arange(H, device=self.device), indexing='xy')
        x = u - W / 2
        y = H / 2 - v
        z = torch.full_like(x, f)

        norm = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        x, y, z = x / norm, y / norm, z / norm

        dirs = torch.stack([x, y, z], dim=0)  # [3, H, W]
        dirs = dirs.reshape(3, -1).contiguous()  # [3, H*W]

        # Rotation matrices
        sin_pitch, cos_pitch = np.sin(pitch_r), np.cos(pitch_r)
        sin_yaw, cos_yaw = np.sin(yaw_r), np.cos(yaw_r)

        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_pitch, -sin_pitch],
            [0, sin_pitch, cos_pitch]
        ], dtype=torch.float32, device=self.device)

        Rz = torch.tensor([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        R = Rz @ Rx
        rotated_dirs = R @ dirs  # [3, H*W]
        x3, y3, z3 = rotated_dirs[0], rotated_dirs[1], rotated_dirs[2]

        # Convert to spherical
        theta = torch.acos(z3.clamp(-1, 1))
        phi = torch.atan2(y3, x3) % (2 * np.pi)

        U = phi * pano_w / (2 * np.pi)
        V = theta * pano_h / np.pi

        # Normalize to [-1, 1]
        U_norm = 2 * (U / pano_w) - 1
        V_norm = 2 * (V / pano_h) - 1
        grid = torch.stack((U_norm, V_norm), dim=-1).reshape(H, W, 2).unsqueeze(0)

        pano_tensor = pano_tensor.unsqueeze(0)
        sampled = torch.nn.functional.grid_sample(pano_tensor, grid, mode='bilinear', align_corners=True)

        sampled_img = (sampled.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        return sampled_img

    def generate_all_views(self, image_path):
        pano = Image.open(image_path).convert('RGB')
        pano_tensor = torch.from_numpy(np.array(pano)).to(self.device)

        result_images = []
        yaws, pitches = [], []

        for yaw in np.linspace(0, 360, 16):
            for pitch in [30,60,90,120,150]:
                view = self._panorama_to_plane(pano_tensor, 115, (512, 512), yaw, pitch)
                result_images.append(view)
                yaws.append(yaw)
                pitches.append(pitch)

        return result_images


class ObjectDetector:
    def __init__(self, model_path, important_classes=None):
        self.model = YOLO(model_path)
        self.important_classes = important_classes  # list of int

    def get_center_distance(self, box, img_w, img_h):
        x_center, y_center, w, h = box
        dx = abs(x_center - img_w / 2)
        dy = abs(y_center - img_h / 2)
        return math.sqrt(dx**2 + dy**2)

    def get_top5_by_center_distance(self, images):
        scored_images = []

        for i, img in enumerate(images):
            result = self.model(img, verbose=False)[0]

            if hasattr(result, "boxes") and result.boxes and result.boxes.xywh is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                img_h, img_w = img.shape[:2]

                min_dist = float('inf')
                for box in boxes:
                    dist = self.get_center_distance(box, img_w, img_h)
                    if dist < min_dist:
                        min_dist = dist

                scored_images.append((min_dist, img))
            #     print(f"View {i}: Min distance to center = {min_dist:.2f}")
            # else:
            #     print(f"View {i}: No boxes found.")
                scored_images.append((float('inf'), img))
        scored_images.sort(key=lambda x: x[0])
        top5_imgs = [img for _, img in scored_images[:5]]
        return top5_imgs


    def select_by_dominant_color(self, images, rgb_color, tolerance=30, max_deviation=10):
        """
        Chọn ảnh trong danh sách `images` (dạng BGR) có nhiều pixel gần với `rgb_color` nhất.
        `rgb_color`: tuple/list (R, G, B)
        `tolerance`: ngưỡng để tính pixel khớp màu (khoảng cách Euclidean)
        `max_deviation`: cho phép lệch tối đa 5 bbp cho mỗi kênh màu
        """
        # Chuyển RGB sang BGR vì ảnh đang ở BGR (do OpenCV)
        target_rgb = np.array(rgb_color)

        def color_match_score(img_rgb):
            # Tính độ lệch tối đa cho mỗi kênh
            diff = np.abs(img_rgb - target_rgb)
            # Kiểm tra điều kiện lệch tối đa cho mỗi kênh (R, G, B)
            match_mask = np.all(diff <= max_deviation, axis=2)
            return np.sum(match_mask)

        scores = []
        for i, img in enumerate(images):
            score = color_match_score(img)
            scores.append((score, img))
            # print(f"Image {i}: Matching pixels = {score}")

        best_score, best_img = max(scores, key=lambda x: x[0])
        return best_img, best_score

class Maskcreation:
    def __init__(self, model_path='model/best.pt', dominant_colors=3, device='cuda'):
        self.model_path = self._resolve_model_path(model_path)
        self.dominant_colors = dominant_colors
        self.device = device

        self.processor = PanoramaProcessor(device=self.device)
        self.detector = ObjectDetector(model_path=self.model_path)
        self.color_extractor = DominantColorExtractor(rgb = (135, 206, 235))

        self.output_dir = self._create_output_folder('output_masks')  # <-- Bỏ comment để bật lưu file
        # # self.all_view = self._create_output_folder('all_view')    # Bật để tạo folder
        # self.top5 = self._create_output_folder('top5')   # Bật để tạo folder
    def _resolve_model_path(self, relative_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.normpath(os.path.join(base_dir, '..', relative_path))
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"✘ Model file not found: {abs_path}")
        return abs_path

    def _create_output_folder(self, folder_name='output'):
        folder_path = os.path.abspath(folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        #     print(f"✓ Created output folder at: {folder_path}")
        # else:
        #     print(f"• Output folder already exists: {folder_path}")
        return folder_path

    def __call__(self, image_path):
        pano_name = os.path.splitext(os.path.basename(image_path))[0]
        views = self.processor.generate_all_views(image_path)

        # Lưu tất cả các góc nhìn
        # for idx, view_np in enumerate(views):
        #     view_img = Image.fromarray(view_np)
        #     view_path = os.path.join(self.all_view, f"{pano_name}_view_{idx}.png")
        #     view_img.save(view_path)

        # Lấy top 5 ảnh có bbox gần tâm nhất
        top5_imgs = self.detector.get_top5_by_center_distance(views)
        # -----Luư top 5 --------
        # for inx, rank in enumerate(top5_imgs):
        #     rank_img = Image.fromarray(rank)
        #     rank_path = os.path.join(self.top5, f"{pano_name}_top{inx+1}.png")
        #     rank_img.save(rank_path)
        if not top5_imgs:
            print("✘ No valid views found.")
            return None

        # Dùng ảnh đầu tiên trong top 5 để lấy dominant color từ box đầu tiên
        # first_img = top5_imgs[0]
        # results = self.detector.model(first_img, verbose=False)[0]

        # if results.boxes is None or len(results.boxes) == 0:
        #     print("✘ No object detected in first view of top 5.")
        #     return None, None

        # Lấy dominant color từ box đầu tiên
        # box = results.boxes.xywhn.cpu().numpy()[0]
        # first_img_pil = Image.fromarray(first_img)
        # dominant_color = self.color_extractor.extract_from_box(first_img_pil, box)[0]
        rgb = (135, 206, 235)

        # Tìm ảnh trong top 5 có nhiều pixel trùng màu nhất với dominant color
        best_img_np, best_score = self.detector.select_by_dominant_color(top5_imgs,rgb)
        if best_score < 100000 and best_score > 2200:
            return None, None

        if best_img_np is None:
            print("✘ No best image found by dominant color.")
            return None, None

        best_img_pil = Image.fromarray(best_img_np)

        # --- Lưu best view ---
        # best_view_path = os.path.join(self.output_dir, f"{pano_name}_best_view.png")
        # best_img_pil.save(best_view_path)
        # print(f"✓ Saved best view image to {best_view_path}")

        # Chạy YOLO để detect lại object, lấy bbox đầu tiên tạo mask
        results = self.detector.model(best_img_pil, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            print("✘ No object detected in final best view.")
            return None, None
        box = results.boxes.xywhn.cpu().numpy()[0]
        mask_img = self.color_extractor.get_masked_image(best_img_pil, tolerance=50)
        mask_img = crop_and_resize(mask_img, box, best_score)
        # --- Lưu mask và dominant color ---
        mask_path = os.path.join(self.output_dir, f"{pano_name}_mask.png")
        txt_path = os.path.join(self.output_dir, f"{pano_name}_color.txt")
        mask_img.save(mask_path)
        # with open(txt_path, 'w') as f:
        #     f.write(str(rgb))

        print(f"✓ Saved mask to {mask_path}")
        # print(f"✓ Saved dominant color to {txt_path}")
        print("Mask create successfully", pano_name, best_score)
        if best_score <=2200:
            return mask_img, 1 
        elif best_score >=100000:
            return mask_img, 2
        return mask_img, 3



def crop_and_resize(mask_img, box, best_score, output_size=(512, 512), zoom_scale=2):
    """
    Xử lý mask theo best_score:
    - Nếu best_score < 5000: zoom vào object, rồi đặt vào ảnh 512x512, tô trắng xung quanh.
    - Nếu 5000 <= best_score < 11000: giữ nguyên ảnh, chỉ tô trắng ngoài bbox.
    - Nếu >= 11000: trả lại ảnh gốc.

    Args:
        mask_img (PIL.Image): ảnh mask gốc.
        box (tuple): bbox YOLO (x_center, y_center, w, h), chuẩn hóa (0-1).
        best_score (float): điểm số.
        output_size (tuple): kích thước đích.
        zoom_scale (float): tỉ lệ zoom vào vật.

    Returns:
        PIL.Image: ảnh đã xử lý.
    """
    W, H = mask_img.size
    x, y, w, h = box

    # bbox gốc (tính theo pixel)
    x_center = x * W
    y_center = y * H
    box_w = w * W
    box_h = h * H

    if best_score < 5000:
        # Crop vùng bbox nhỏ trước
        left = int(x_center - box_w / 2)
        top = int(y_center - box_h / 2)
        right = int(x_center + box_w / 2)
        bottom = int(y_center + box_h / 2)

        # Clamp
        left = max(0, left)
        top = max(0, top)
        right = min(W, right)
        bottom = min(H, bottom)

        cropped = mask_img.crop((left, top, right, bottom))

        # Zoom vào (resize object lên)
        zoomed_w = int(cropped.width * zoom_scale)
        zoomed_h = int(cropped.height * zoom_scale)
        zoomed = cropped.resize((zoomed_w, zoomed_h), Image.LANCZOS)

        # Tạo canvas trắng 512x512 và paste sao cho tâm object đúng tâm bbox
        canvas = Image.new(mask_img.mode, output_size, 255 if mask_img.mode == "L" else (255, 255, 255))

        paste_x = int(output_size[0] / 2 - zoomed_w / 2)
        paste_y = int(output_size[1] / 2 - zoomed_h / 2)

        # Cắt nếu object quá to
        zoomed = zoomed.crop((
            max(0, -paste_x),
            max(0, -paste_y),
            min(zoomed_w, output_size[0] - paste_x),
            min(zoomed_h, output_size[1] - paste_y)
        ))

        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)
        canvas.paste(zoomed, (paste_x, paste_y))
        return canvas

    elif best_score < 14000:
        # Tô trắng vùng ngoài bbox
        white = Image.new(mask_img.mode, (W, H), 255 if mask_img.mode == "L" else (255, 255, 255))
        left = int((x - w / 2) * W)
        top = int((y - h / 2) * H)
        right = int((x + w / 2) * W)
        bottom = int((y + h / 2) * H)
        left, top, right, bottom = max(0, left), max(0, top), min(W, right), min(H, bottom)

        region = mask_img.crop((left, top, right, bottom))
        white.paste(region, (left, top))
        return white

    return mask_img
