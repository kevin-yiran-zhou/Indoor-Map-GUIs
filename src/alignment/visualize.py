import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_trajectory(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split()
            tx, ty, tz = map(float, parts[1:4])
            data.append([tx, ty, tz])
    return np.array(data)

def load_point_cloud(file_path):
    if not os.path.exists(file_path):
        print(f"[Warning] Point cloud file not found: {file_path}")
        return None
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == "" or line.startswith("pos_x"):
                continue
            parts = line.strip().replace(',', ' ').split()
            x, y, z = map(float, parts)
            data.append([x, y, z])
    return np.array(data)

def main():
    parser = argparse.ArgumentParser(description="Visualize ORB-SLAM trajectory and point cloud.")
    parser.add_argument("--floor", type=str, required=True, help="Floor name (e.g., FRB2)")
    parser.add_argument("--ref", action='store_true', help="Use reference map points instead of raw map points")
    args = parser.parse_args()

    floor = args.floor
    base_path = os.path.expanduser(f"/home/kevin-zhou/Desktop/UMich/WeilandLab/Indoor-Map-GUIs/data/slam_result/{floor}")
    
    kf_file = os.path.join(base_path, f"kf_{floor}.txt")
    f_file = os.path.join(base_path, f"f_{floor}.txt")

    # Choose between ref_map_points.txt and map_points.txt
    pc_file = os.path.join(base_path, "ref_map_points.txt" if args.ref else "map_points.txt")

    if not os.path.exists(kf_file) or not os.path.exists(f_file):
        print(f"[Error] Missing trajectory files for floor '{floor}'. Expected at:")
        print(f"  {kf_file}")
        print(f"  {f_file}")
        return

    kf_traj = load_trajectory(kf_file)
    f_traj = load_trajectory(f_file)
    point_cloud = load_point_cloud(pc_file)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(f_traj[:, 0], f_traj[:, 1], f_traj[:, 2], label="All Frames", linewidth=0.5, color='gray')
    ax.plot(kf_traj[:, 0], kf_traj[:, 1], kf_traj[:, 2], 'r.-', label="Keyframes")

    if point_cloud is not None:
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                   s=1, c='blue', label="Map Points", alpha=0.6)

    ax.set_title(f"ORB-SLAM Trajectory + Map Points ({'Ref' if args.ref else 'Raw'}) - {floor}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
