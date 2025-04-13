import os
import argparse
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from sklearn.cluster import KMeans


class DominantColorExtractor:
    def __init__(self, n_colors=1):
        self.n_colors = n_colors

    def extract_from_box(self, image, box):
        """
        Trích xuất màu chủ đạo từ một bounding box.
        box: [x_center, y_center, width, height] (normalized YOLO format)
        """
        W, H = image.size
        x, y, w, h = box
        left = int((x - w / 2) * W)
        top = int((y - h / 2) * H)
        right = int((x + w / 2) * W)
        bottom = int((y + h / 2) * H)

        cropped = image.crop((left, top, right, bottom))
        cropped = cropped.resize((100, 100))  # Resize cho nhanh
        img_np = np.array(cropped).reshape(-1, 3)

        if len(img_np) < self.n_colors:
            return [(0, 0, 0)]

        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init='auto')
        kmeans.fit(img_np)
        dominant = kmeans.cluster_centers_.astype(int)
        return [tuple(c) for c in dominant]

    def get_masked_image(self, image, box, tolerance=30):
        """
        Giữ lại vùng trong bbox có màu gần giống màu chủ đạo, các vùng còn lại chuyển thành trắng.
        """
        W, H = image.size
        x, y, w, h = box
        left = int((x - w / 2) * W)
        top = int((y - h / 2) * H)
        right = int((x + w / 2) * W)
        bottom = int((y + h / 2) * H)

        img_np = np.array(image).copy()
        cropped = img_np[top:bottom, left:right]

        if cropped.size == 0:
            return image  # tránh lỗi nếu bbox lệch

        dominant_color = self.extract_from_box(image, box)[0]

        # Tính khoảng cách đến dominant color
        dist = np.linalg.norm(cropped - np.array(dominant_color), axis=2)
        mask = dist < tolerance  # giữ pixel gần dominant

        # Tạo ảnh kết quả (nền trắng)
        new_img = np.ones_like(img_np) * 255

        # Chỉ giữ lại pixel gốc trong bbox nếu gần dominant
        new_img_crop = new_img[top:bottom, left:right]
        new_img_crop[mask] = cropped[mask]

        new_img[top:bottom, left:right] = new_img_crop

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

        norm = torch.sqrt(x * 2 + y * 2 + z ** 2)
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
            for pitch in [30, 60, 90, 120, 150]:
                view = self._panorama_to_plane(pano_tensor, 100, (512, 512), yaw, pitch)
                result_images.append(view)
                yaws.append(yaw)
                pitches.append(pitch)

        return result_images

class ObjectDetector:
    def __init__(self, model_path, important_classes=None):
        self.model = YOLO(model_path)
        self.important_classes = important_classes  # list of int

    # def object_alignment_score(self, box, img_w, img_h, conf):
    #     x_center, y_center, w, h = box

    #     dx = (x_center - img_w / 2) / (img_w / 2)
    #     dy = (y_center - img_h / 2) / (img_h / 2)
    #     distance = np.sqrt(dx**2 + dy**2)

    #     center_bonus = np.exp(-5 * distance)  # càng gần trung tâm càng tốt
    #     size_score = w * h / (img_w * img_h)  # phần trăm diện tích
    #     return conf * center_bonus * size_score
    
    def compute_box_score(self, box, img_w, img_h, conf, cls_id):
        x_center, y_center, w, h = box

        dx = (x_center - img_w / 2) / (img_w / 2)
        dy = (y_center - img_h / 2) / (img_h / 2)
        distance_sq = dx * 2 + dy * 2
        center_weight = np.exp(-4 * distance_sq)  # Gaussian drop-off

        class_weight = 1.5 if self.important_classes and int(cls_id) in self.important_classes else 1.0

        return w * h * conf * center_weight * class_weight

    def get_best_view(self, images):
        best_img = None
        best_box_score = -1

        for i, img in enumerate(images):
            result = self.model(img, verbose=False)[0]
            if hasattr(result, "boxes") and result.boxes and result.boxes.xywh is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clses = result.boxes.cls.cpu().numpy()
                img_h, img_w = img.shape[:2]

                for j, box in enumerate(boxes):
                    x_center, y_center, w, h = box
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2
                    x_max = x_center + w / 2
                    y_max = y_center + h / 2

                    if x_min <= 0 or y_min <= 0 or x_max >= img_w or y_max >= img_h:
                        continue  # Skip box bị cắt

                    conf = confs[j]
                    cls_id = clses[j]
                    box_score = self.compute_box_score(box, img_w, img_h, conf, cls_id)

                    if box_score > best_box_score:
                        best_box_score = box_score
                        best_img = img
                        print(f"👉 View {i}: New best frontal box score = {box_score:.2f}")

        print(f"\n🏆 Best frontal view score: {best_box_score:.2f}")
        return best_img

class Maskcreation:
    def __init__(self, model_path='model/best.pt', dominant_colors=3, device='cuda'):
        self.model_path = self._resolve_model_path(model_path)
        self.dominant_colors = dominant_colors
        self.device = device

        self.processor = PanoramaProcessor(device=self.device)
        self.detector = ObjectDetector(model_path=self.model_path)
        self.color_extractor = DominantColorExtractor(n_colors=self.dominant_colors)

        self.output_dir = self._create_output_folder('output_masks')  # <-- Bỏ comment để bật lưu file

    def _resolve_model_path(self, relative_path):
        base_dir = os.path.dirname(os.path.abspath(_file_))
        abs_path = os.path.normpath(os.path.join(base_dir, '..', relative_path))
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"✘ Model file not found: {abs_path}")
        return abs_path

    def _create_output_folder(self, folder_name='output'):
        folder_path = os.path.abspath(folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"✓ Created output folder at: {folder_path}")
        else:
            print(f"• Output folder already exists: {folder_path}")
        return folder_path

    def __call__(self, image_path):
        pano_name = os.path.splitext(os.path.basename(image_path))[0]
        views = self.processor.generate_all_views(image_path)
        best_img_np = self.detector.get_best_view(views)

        if best_img_np is None:
            print("✘ No object detected in any view.")
            return None, None

        best_img_pil = Image.fromarray(best_img_np)

        results = self.detector.model(best_img_np, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            print("✘ No object detected in best view.")
            return None, None

        box = results.boxes.xywhn.cpu().numpy()[0]
        dominant_color = self.color_extractor.extract_from_box(best_img_pil, box)[0]

        mask_img = self.color_extractor.get_masked_image(best_img_pil, box, tolerance=30)
        bgr_str = ','.join(str(int(c)) for c in dominant_color[::-1])

        # --- Bật chức năng lưu nếu cần ---
        mask_path = os.path.join(self.output_dir, f"{pano_name}_mask.png")
        txt_path = os.path.join(self.output_dir, f"{pano_name}_color.txt")
        mask_img.save(mask_path)
        with open(txt_path, 'w') as f:
            f.write(bgr_str)
        print(f"✓ Saved mask to {mask_path}")
        print(f"✓ Saved dominant color to {txt_path}")

        return mask_img, bgr_str
