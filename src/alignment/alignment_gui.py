import os
import argparse
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import Label, Button
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from skimage.transform import estimate_transform

# ========= Loaders ========= #
def load_keyframes(path):
    kf = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split()
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            kf.append([tx, ty, tz])
    return np.array(kf)

def load_map_points(path):
    points = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip() == "" or line.startswith("pos_x"):
                continue
            parts = line.strip().replace(",", " ").split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append([x, y, z])
    return np.array(points)

def project_to_plane(trajectory, pointcloud):
    all_points = np.vstack([trajectory, pointcloud])
    pca = PCA(n_components=2)
    pca.fit(all_points)
    traj_2d = pca.transform(trajectory)
    pc_2d = pca.transform(pointcloud)
    return traj_2d, pc_2d

# ========= Main GUI Class ========= #
class AlignmentGUI:
    def __init__(self, root, floorplan_img, keyframe_positions, point_cloud=None):
        self.root = root
        self.root.title("Alignment Tool")
        self.kf_3d = keyframe_positions
        self.pc_3d = point_cloud if point_cloud is not None else np.empty((0, 3))
        self.kf, self.pc = project_to_plane(self.kf_3d, self.pc_3d)

        self.kf_pts = []
        self.img_pts = []
        self.img_array = floorplan_img

        # === Floorplan Panel ===
        img_rgb = (floorplan_img * 255).astype(np.uint8) if floorplan_img.max() <= 1.0 else floorplan_img
        img_pil = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(image=img_pil)

        img_label = Label(root, image=self.tk_image)
        img_label.grid(row=0, column=0, rowspan=4, padx=10, pady=10)

        # === Controls Panel ===
        Button(root, text="Show SLAM Trajectory", command=self.show_kf_plot, width=25).grid(row=0, column=1, pady=10)
        Button(root, text="Show Floorplan", command=self.show_img_plot, width=25).grid(row=1, column=1, pady=10)

        self.info_label = Label(root, text=f"Correspondences: 0")
        self.info_label.grid(row=2, column=1, pady=10)

        Button(root, text="Compute Alignment", command=self.compute_alignment, width=25).grid(row=3, column=1, pady=10)

    def update_info_label(self):
        self.info_label.config(text=f"Correspondences: {len(self.kf_pts)}")

    def show_kf_plot(self):
        fig, ax = plt.subplots()
        ax.set_title("SLAM Trajectory + Map Points (Projected 2D)")
        ax.plot(self.kf[:, 0], self.kf[:, 1], 'r.-', label="Keyframes")
        if self.pc is not None:
            ax.scatter(self.pc[:, 0], self.pc[:, 1], s=1, c='blue', alpha=0.3, label="Map Points")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def show_img_plot(self):
        fig, ax = plt.subplots()
        ax.set_title("Floorplan Image")
        ax.imshow(self.img_array, cmap='gray')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def compute_alignment(self):
        if len(self.kf_pts) < 3:
            messagebox.showerror("Error", "Need at least 3 correspondences to compute alignment.")
            return

        tform = estimate_transform('similarity', np.array(self.kf_pts), np.array(self.img_pts))
        print("✅ Alignment complete.")

        transformed = tform(self.kf)
        fig, ax = plt.subplots()
        ax.set_title("Aligned Trajectory")
        ax.imshow(self.img_array, cmap='gray')
        ax.plot(transformed[:, 0], transformed[:, 1], 'r--', label='Aligned Trajectory')

        if self.pc is not None:
            pc2d = tform(self.pc)
            ax.scatter(pc2d[:, 0], pc2d[:, 1], s=1, c='blue', alpha=0.3, label='Map Points')

        ax.legend()
        ax.set_aspect('equal')
        plt.show()

# ========= Main ========= #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--floor', type=str, required=True, help='Floor name (e.g., FRB2)')
    args = parser.parse_args()

    floorplan_path = os.path.expanduser(f"/home/kevin-zhou/Desktop/UMich/WeilandLab/Indoor-Map-GUIs/data/floorplans")
    img_path = os.path.join(floorplan_path, f"{args.floor}.jpg")
    print(f"Loading floorplan image from {img_path}")
    slam_path = os.path.expanduser(f"/home/kevin-zhou/Desktop/UMich/WeilandLab/Indoor-Map-GUIs/data/slam_result/{args.floor}")
    kf_path = os.path.join(slam_path, f"kf_{args.floor}.txt")
    map_path = os.path.join(slam_path, "map_points.txt")

    if not os.path.exists(img_path) or not os.path.exists(kf_path) or not os.path.exists(map_path):
        print("❌ Missing required files.")
        return

    img = mpimg.imread(img_path)
    kf = load_keyframes(kf_path)
    pc = load_map_points(map_path) if os.path.exists(map_path) else None

    root = tk.Tk()
    app = AlignmentGUI(root, img, kf, pc)
    root.mainloop()

if __name__ == "__main__":
    main()