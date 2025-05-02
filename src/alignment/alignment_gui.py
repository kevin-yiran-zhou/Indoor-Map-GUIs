import os
import argparse
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from scipy.interpolate import Rbf

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
    traj_2d[:, 1] *= -1
    pc_2d[:, 1] *= -1
    return traj_2d, pc_2d

class PointCloudAligner:
    def __init__(self, root, kf, pc, floor_img, slam_path):
        self.root = root
        self.kf = kf
        self.pc = pc
        self.floor_img = floor_img
        self.slam_path = slam_path

        self.kf_points = []
        self.floor_points = []

        self.selected_kf_point = None
        self.selected_floor_point = None

        self.status_var = tk.StringVar()

        self.fig, (self.ax_top, self.ax_bottom) = plt.subplots(2, 1, figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)
        tk.Button(button_frame, text="Add Correspondence", command=self.add_correspondence, font=("Helvetica", 14), width=20, height=2).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Remove Last Correspondence", command=self.remove_last_correspondence, font=("Helvetica", 14), width=25, height=2).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Save Correspondences", command=self.save_correspondences, font=("Helvetica", 14), width=25, height=2).pack(side=tk.LEFT, padx=10)

        tk.Label(root, textvariable=self.status_var, fg="blue", font=("Helvetica", 12)).pack(pady=4)
        
        self.load_correspondences()
        self.redraw()

    def on_click(self, event):
        if event.inaxes == self.ax_top:
            self.selected_kf_point = np.array([event.xdata, event.ydata])
        elif event.inaxes == self.ax_bottom:
            self.selected_floor_point = np.array([event.xdata, event.ydata])
        self.redraw()

    def load_correspondences(self):
        path = os.path.join(self.slam_path, f"correspondences.txt")
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
            for line in f:
                kx, ky, fx, fy = map(float, line.strip().split())
                self.kf_points.append(np.array([kx, ky]))
                self.floor_points.append(np.array([fx, fy]))
        self.status_var.set(f"Loaded {len(self.kf_points)} correspondences.")

    def add_correspondence(self):
        if self.selected_kf_point is not None and self.selected_floor_point is not None:
            self.kf_points.append(self.selected_kf_point)
            self.floor_points.append(self.selected_floor_point)
            self.selected_kf_point = None
            self.selected_floor_point = None
            self.status_var.set(f"Added correspondence. Total: {len(self.kf_points)}")
            self.redraw()

    def remove_last_correspondence(self):
        if self.kf_points:
            self.kf_points.pop()
            self.floor_points.pop()
            self.status_var.set(f"Removed last correspondence. Total: {len(self.kf_points)}")
            self.redraw()
    
    def save_correspondences(self):
        if not self.kf_points or not self.floor_points:
            self.status_var.set("Nothing to save.")
            return
        save_path = os.path.join(self.slam_path, f"correspondences.txt")
        with open(save_path, 'w') as f:
            for kf_pt, fl_pt in zip(self.kf_points, self.floor_points):
                f.write(f"{kf_pt[0]} {kf_pt[1]} {fl_pt[0]} {fl_pt[1]}\n")
        self.status_var.set(f"Saved {len(self.kf_points)} correspondences.")

    def redraw(self):
        self.ax_top.clear()
        self.ax_top.set_title("Point Cloud and Trajectory")
        self.ax_top.plot(self.kf[:, 0], self.kf[:, 1], 'r.-', label='Keyframes')
        if self.pc is not None:
            self.ax_top.scatter(self.pc[:, 0], self.pc[:, 1], s=1, c='blue', alpha=0.3, label='Map Points')
        if self.selected_kf_point is not None:
            self.ax_top.plot(self.selected_kf_point[0], self.selected_kf_point[1], 'ko', markersize=10, label='Selected')
        self.ax_top.set_aspect('equal')
        self.ax_top.legend()

        self.ax_bottom.clear()
        self.ax_bottom.set_title("Floor Plan + TPS Aligned")
        self.ax_bottom.imshow(self.floor_img, cmap='gray', origin='upper')
        self.ax_bottom.set_xlim([0, self.floor_img.shape[1]])
        self.ax_bottom.set_ylim([self.floor_img.shape[0], 0])

        for pt in self.floor_points:
            self.ax_bottom.plot(pt[0], pt[1], 'go', markersize=5)
        if len(self.kf_points) >= 4:
            src = np.array(self.kf_points)
            dst = np.array(self.floor_points)
            fx = Rbf(src[:, 0], src[:, 1], dst[:, 0], function='thin_plate')
            fy = Rbf(src[:, 0], src[:, 1], dst[:, 1], function='thin_plate')
            aligned_kf = np.stack([fx(self.kf[:, 0], self.kf[:, 1]), fy(self.kf[:, 0], self.kf[:, 1])], axis=1)
            self.ax_bottom.plot(aligned_kf[:, 0], aligned_kf[:, 1], 'r.-', label='Aligned Keyframes')
            if self.pc is not None:
                aligned_pc = np.stack([fx(self.pc[:, 0], self.pc[:, 1]), fy(self.pc[:, 0], self.pc[:, 1])], axis=1)
                self.ax_bottom.scatter(aligned_pc[:, 0], aligned_pc[:, 1], s=1, c='blue', alpha=0.3, label='Aligned Map Points')
            self.ax_bottom.legend()
        if self.selected_floor_point is not None:
            self.ax_bottom.plot(self.selected_floor_point[0], self.selected_floor_point[1], 'ko', markersize=10, label='Selected')
        self.ax_bottom.set_aspect('equal')
        self.canvas.draw()

# ======= Main launcher using real files ======= #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--floor', type=str, required=True, help='Floor name (e.g., FRB2)')
    args = parser.parse_args()

    floorplan_path = os.path.expanduser("~/Desktop/UMich/WeilandLab/Indoor-Map-GUIs/data/floorplans")
    img_path = os.path.join(floorplan_path, f"{args.floor}.jpg")
    slam_path = os.path.expanduser(f"~/Desktop/UMich/WeilandLab/Indoor-Map-GUIs/data/slam_result/{args.floor}")
    kf_path = os.path.join(slam_path, f"kf_{args.floor}.txt")
    map_path = os.path.join(slam_path, "map_points.txt")

    if not os.path.exists(img_path) or not os.path.exists(kf_path):
        print("Missing required files.")
        return

    img = mpimg.imread(img_path)
    kf = load_keyframes(kf_path)
    pc = load_map_points(map_path) if os.path.exists(map_path) else None
    kf_2d, pc_2d = project_to_plane(kf, pc)

    root = tk.Tk()
    root.title("Point Cloud TPS Alignment GUI")
    app = PointCloudAligner(root, kf_2d, pc_2d, img, slam_path)

    root.mainloop()

if __name__ == '__main__':
    main()
