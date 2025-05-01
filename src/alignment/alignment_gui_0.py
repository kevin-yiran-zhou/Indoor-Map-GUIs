import os
import argparse
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
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
    return traj_2d, pc_2d

# ========= Main GUI Class ========= #
class AlignmentGUI:
    def __init__(self, root, floorplan_img, keyframe_positions, point_cloud=None):
        self.root = root
        self.root.title("Alignment Tool")

        self.kf_3d = keyframe_positions
        self.pc_3d = point_cloud if point_cloud is not None else np.empty((0, 3))
        self.kf, self.pc = project_to_plane(self.kf_3d, self.pc_3d)

        self.mode = None
        self.last_mouse_pos = None
        self.deform_start = None
        self.control_src = []
        self.control_dst = []

        self.img_rgb = (floorplan_img * 255).astype(np.uint8) if floorplan_img.max() <= 1.0 else floorplan_img
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)
        tk.Button(control_frame, text="Move", command=lambda: self.set_mode('move')).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Zoom", command=lambda: self.set_mode('zoom')).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Rotate", command=lambda: self.set_mode('rotate')).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Deform", command=lambda: self.set_mode('deform')).pack(side=tk.LEFT, padx=5)

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_drag)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.redraw()

    def set_mode(self, mode):
        self.mode = mode
        print(f"🟢 {mode.capitalize()} mode enabled")

    def on_mouse_press(self, event):
        if self.mode in ['move', 'rotate', 'deform'] and event.xdata is not None and event.ydata is not None:
            self.last_mouse_pos = np.array([event.xdata, event.ydata])
            if self.mode == 'deform':
                self.deform_start = self.last_mouse_pos.copy()

    def on_mouse_release(self, event):
        if self.mode == 'deform' and self.deform_start is not None and event.xdata is not None and event.ydata is not None:
            deform_end = np.array([event.xdata, event.ydata])
            self.control_src.append(self.deform_start)
            self.control_dst.append(deform_end)
            print(f"Added deform pair: {self.deform_start} → {deform_end}")
            if len(self.control_src) >= 3:
                self.apply_tps_deformation()
            self.deform_start = None
        self.last_mouse_pos = None

    def on_mouse_drag(self, event):
        if self.last_mouse_pos is not None and event.xdata is not None and event.ydata is not None:
            curr_mouse_pos = np.array([event.xdata, event.ydata])
            delta = curr_mouse_pos - self.last_mouse_pos

            if self.mode == 'move':
                self.kf += delta
                if self.pc is not None:
                    self.pc += delta
            elif self.mode == 'rotate':
                center = np.mean(self.kf, axis=0)
                v1 = self.last_mouse_pos - center
                v2 = curr_mouse_pos - center
                angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                R = np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle),  np.cos(angle)]])
                self.kf = (self.kf - center) @ R.T + center
                if self.pc is not None:
                    self.pc = (self.pc - center) @ R.T + center

            self.last_mouse_pos = curr_mouse_pos
            self.redraw()

    def on_scroll(self, event):
        if self.mode == 'zoom':
            zoom = 1.1 if event.step > 0 else 0.9
            center = np.mean(self.kf, axis=0)
            self.kf = (self.kf - center) * zoom + center
            if self.pc is not None:
                self.pc = (self.pc - center) * zoom + center
            self.redraw()

    def apply_tps_deformation(self):
        print("Applying TPS with", len(self.control_src), "pairs")
        src = np.array(self.control_src)
        dst = np.array(self.control_dst)

        def warp(points):
            x, y = points[:, 0], points[:, 1]
            fx = Rbf(src[:, 0], src[:, 1], dst[:, 0], function='thin_plate')
            fy = Rbf(src[:, 0], src[:, 1], dst[:, 1], function='thin_plate')
            return np.stack([fx(x, y), fy(x, y)], axis=1)

        self.kf = warp(self.kf)
        if self.pc is not None:
            self.pc = warp(self.pc)
        self.redraw()

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.img_rgb, cmap='gray')
        self.ax.plot(self.kf[:, 0], self.kf[:, 1], 'r.-', label="Keyframes")
        if self.pc is not None:
            self.ax.scatter(self.pc[:, 0], self.pc[:, 1], s=1, c='blue', alpha=0.3, label="Map Points")
        self.ax.set_aspect('equal')
        self.ax.legend()
        self.canvas.draw()
        self.canvas.flush_events()

# ========= Main ========= #
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