import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

# ================== 参数设置 ==================
n_total = 30           # 收集点数量（不含处理厂）
m_candidates = 5       # 中转站候选点数
Tf = 100000            # 中转站建设成本
S_k = [50, 60, 40, 55]  # 四类垃圾存储容量（假设）
operating_time = [6, 18]  # 运营时间窗口
k_types = 4            # 垃圾类型数量

# 车辆参数（来自附件2）
# [载重, 容积, 单位距离成本, 碳排放系数1, 碳排放系数2]
vehicle_params = np.array([
    [8, 20, 2.5, 0.8, 0.3],   # 厨余垃圾
    [6, 25, 2.0, 0.6, 0.2],   # 可回收物
    [3, 10, 5.0, 1.2, 0.5],   # 有害垃圾
    [10, 18, 1.8, 0.7, 0.25],  # 其他垃圾
])

# PSO 参数
swarm_size = 50
max_iter = 50
w_pso = 0.729
c1 = c2 = 1.49445

# 收集点坐标和垃圾量（处理厂 + 30个收集点）
points = np.array([
    [0, 0, 0, 0, 0, 0],  # 处理厂（索引0）
    [12, 8, 0.72, 0.12, 0.06, 0.3],
    [5, 15, 1.38, 0.23, 0.05, 0.64],
    [20, 30, 1.08, 0.18, 0.04, 0.5],
    [25, 10, 1.55, 0.31, 0.06, 1.18],
    [35, 22, 1.62, 0.27, 0.05, 0.76],
    [18, 5, 1.76, 0.384, 0.096, 0.96],
    [30, 35, 0.77, 0.168, 0.042, 0.42],
    [10, 25, 1.02, 0.238, 0.068, 0.374],
    [22, 18, 1.32, 0.176, 0.044, 0.66],
    [38, 15, 1.45, 0.3, 0.075, 0.675],
    [5, 8, 1.35, 0.27, 0.108, 0.972],
    [15, 32, 1.87, 0.51, 0.068, 0.952],
    [28, 5, 2.58, 0.516, 0.129, 1.075],
    [30, 12, 1.134, 0.21, 0.063, 0.693],
    [10, 10, 0.78, 0.13, 0.065, 0.325],
    [20, 20, 0.768, 0.192, 0.08, 0.56],
    [35, 30, 0.72, 0.27, 0.09, 0.72],
    [8, 22, 1.595, 0.348, 0.087, 0.87],
    [25, 25, 1.5, 0.36, 0.09, 1.05],
    [32, 8, 1.08, 0.18, 0.09, 0.45],
    [15, 5, 0.912, 0.19, 0.038, 0.76],
    [28, 20, 0.9, 0.195, 0.075, 0.33],
    [38, 25, 0.99, 0.27, 0.072, 0.468],
    [10, 30, 1.44, 0.24, 0.048, 0.672],
    [20, 10, 1.74, 0.319, 0.116, 0.725],
    [30, 18, 1.17, 0.39, 0.13, 0.91],
    [5, 25, 1.7, 0.34, 0.17, 1.19],
    [18, 30, 2.64, 0.66, 0.044, 1.056],
    [35, 10, 0.864, 0.216, 0.072, 0.648],
    [22, 35, 0.986, 0.204, 0.085, 0.425],
])

# 计算距离矩阵（向量化优化）
def calculate_distance_matrix(coords):
    coords = np.asarray(coords)
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

# 阶段一：中转站聚类选址
def find_transfer_stations(points_data, m_candidates):
    coords = points_data[1:, :2]  # 忽略处理厂
    kmeans = KMeans(n_clusters=m_candidates, random_state=0).fit(coords)
    cluster_idx = kmeans.labels_
    centroids = kmeans.cluster_centers_

    valid_stations = []
    for i in range(m_candidates):
        members = np.where(cluster_idx == i)[0]
        total_load = np.sum(points_data[members + 1, 2:6], axis=0)
        if all(load <= S_k[idx] for idx, load in enumerate(total_load)):
            valid_stations.append({
                'id': i,
                'coords': centroids[i],
                'members': members
            })
        else:
            print(f"Candidate station {i+1} overloaded. Max capacity {S_k}, actual {total_load}")
    print("Valid transfer station locations:")
    for vs in valid_stations:
        print(vs['coords'])
    return valid_stations

# 阶段二：路径优化子函数
@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    cost: float
    best_position: np.ndarray
    best_cost: float

def decode_particle(position, points, D, vehicle_params):
    routes = [[] for _ in range(k_types)]
    total_cost = 0
    total_emission = 0

    for k in range(k_types):
        idx = np.where(position[:, k] == 1)[0]
        if len(idx) == 0:
            continue

        type_routes, type_cost, type_emission = decode_type_routes(
            idx, points, D, vehicle_params[k]
        )
        routes[k] = type_routes
        total_cost += type_cost
        total_emission += type_emission

    return routes, total_cost, total_emission

def decode_type_routes(idx, points, D, params):
    Q = params[0]
    C = params[2]
    alpha = params[3]
    beta = params[4]

    current_path = [0]
    current_load = 0
    routes = []
    type_cost = 0
    type_emission = 0

    if len(idx) == 0:
        return [], 0, 0

    for point_index in idx:
        point_id = point_index + 1
        w = points[point_id, 2]

        if current_load + w > Q:
            current_path.append(0)
            path_array = np.array(current_path)
            dist = np.sum(D[path_array[:-1], path_array[1:]])
            cost = dist * C
            emission = dist * (alpha + beta * current_load)

            routes.append({'path': current_path, 'load': current_load, 'distance': dist, 'cost': cost, 'emission': emission})
            type_cost += cost
            type_emission += emission

            current_path = [0, point_id]
            current_load = w
        else:
            current_path.append(point_id)
            current_load += w

    current_path.append(0)
    path_array = np.array(current_path)
    dist = np.sum(D[path_array[:-1], path_array[1:]])
    cost = dist * C
    emission = dist * (alpha + beta * current_load)
    routes.append({'path': current_path, 'load': current_load, 'distance': dist, 'cost': cost, 'emission': emission})
    type_cost += cost
    type_emission += emission

    return routes, type_cost, type_emission

def optimize_routes_sub(points, D, vehicle_params):
    n = len(points) - 1
    particles = []

    for _ in range(swarm_size):
        pos = np.random.randint(0, 2, size=(n, k_types))
        for j in range(n):
            if np.sum(pos[j]) == 0:
                type_idx = j % k_types
                pos[j, type_idx] = 1
        particles.append(Particle(
            position=pos.astype(int),
            velocity=np.zeros((n, k_types)),
            cost=float('inf'),
            best_position=pos.copy().astype(int),
            best_cost=float('inf')
        ))

    global_best = particles[0]
    cost_history = []  # 用于记录收敛曲线

    for iter_num in range(max_iter):
        for p in particles:
            _, cost, _ = decode_particle(p.position, points, D, vehicle_params)
            p.cost = cost
            if cost < p.best_cost:
                p.best_position = p.position.copy()
                p.best_cost = cost
            if cost < global_best.cost:
                global_best = p

        for p in particles:
            r1 = np.random.rand(n, k_types)
            r2 = np.random.rand(n, k_types)
            p.velocity = w_pso * p.velocity + \
                         c1 * r1 * (p.best_position - p.position) + \
                         c2 * r2 * (global_best.position - p.position)
            p.position = (1 / (1 + np.exp(-p.velocity)) > 0.5).astype(int)
            for j in range(n):
                if np.sum(p.position[j]) == 0:
                    type_idx = j % k_types
                    p.position[j, type_idx] = 1

        cost_history.append(global_best.cost)
        print(f"Iteration {iter_num+1}/{max_iter}, Best Cost: {global_best.cost:.2f} CNY")

    # 返回结果及历史记录
    routes, total_cost, total_emission = decode_particle(global_best.position, points, D, vehicle_params)
    return routes, total_cost, total_emission, cost_history

# ================== 可视化扩展 ==================

def plot_path_heatmap(points, valid_stations, all_routes, grid_size=20):
    """
    绘制路径热力图，展示路径重合度
    """
    path_map = np.zeros((grid_size, grid_size))

    def map_to_grid(x, y, max_x, max_y):
        gx = int(x / max_x * (grid_size - 1))
        gy = int(y / max_y * (grid_size - 1))
        return min(gx, grid_size - 1), min(gy, grid_size - 1)

    max_x = np.max(points[:, 0]) + 1
    max_y = np.max(points[:, 1]) + 1

    for station_routes in all_routes:
        for k_type in range(k_types):
            for route_info in station_routes[k_type]:
                path = route_info['path']
                for i in range(len(path) - 1):
                    x1, y1 = points[path[i]][0], points[path[i]][1]
                    x2, y2 = points[path[i+1]][0], points[path[i+1]][1]
                    line_points = np.linspace((x1, y1), (x2, y2), 10)
                    for x, y in line_points:
                        gx, gy = map_to_grid(x, y, max_x, max_y)
                        path_map[gy, gx] += 1

    plt.figure(figsize=(8, 8))
    plt.imshow(path_map, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(label='Path Overlap Times')
    plt.title("Path Heatmap (Overlap Analysis)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(False)
    plt.show()

def plot_convergence_curve(cost_history, title="PSO Convergence Curve"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cost_history)+1), cost_history, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost (CNY)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_all_routes(points, valid_stations):
    plt.figure(figsize=(10, 8))
    plt.title("Transfer Station Location and Collection Point Assignment", fontsize=12)

    # Plot all collection points
    for i in range(1, len(points)):
        plt.plot(points[i, 0], points[i, 1], "ko", markersize=6)
        plt.text(points[i, 0], points[i, 1], str(i), fontsize=8)

    # Plot treatment plant and transfer stations
    plt.plot(points[0, 0], points[0, 1], "gs", markersize=10, label="Treatment Plant")
    for vs in valid_stations:
        plt.plot(vs['coords'][0], vs['coords'][1], "r^", markersize=10, label="Transfer Station")

    colors = ['C{}'.format(i % 10) for i in range(len(valid_stations))]
    for idx, vs in enumerate(valid_stations):
        members = vs['members']
        for pt_idx in members:
            x, y = points[pt_idx+1, 0], points[pt_idx+1, 1]
            plt.plot(x, y, '.', color=colors[idx], label=f"Cluster {idx+1}")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

# 主程序入口
def main():
    valid_stations = find_transfer_stations(points, m_candidates)
    total_cost = Tf * len(valid_stations)
    total_emission = 0

    all_routes = []  # 存储所有路径信息，用于热力图
    all_costs = []   # 存储每次子问题的成本历史

    for station in valid_stations:
        members = station['members']
        station_coord = station['coords']

        sub_points = np.vstack((
            np.hstack((station_coord, [0]*4)),  # 中转站
            points[members + 1]  # 收集点
        ))
        D_sub = calculate_distance_matrix(sub_points[:, :2])

        routes, cost, emission, cost_history = optimize_routes_sub(sub_points, D_sub, vehicle_params)
        total_cost += cost
        total_emission += emission
        all_routes.append(routes)
        all_costs.append(cost_history)

        print(f"\nTransfer Station at {station_coord}\n")
        for k in range(k_types):
            print(f"Vehicle Type {k+1}:")
            for r in routes[k]:
                path = r['path']
                nodes = " → ".join(map(str, path[1:-1]))
                print(f"  Route: {nodes}, Load: {r['load']:.1f} tons, Distance: {r['distance']:.1f} km, Cost: {r['cost']:.1f} CNY, Emission: {r['emission']:.1f} kg")

    # 可视化：收敛曲线（取最后一次优化的历史）
    if all_costs:
        plot_convergence_curve(all_costs[-1], "PSO Convergence Curve")

    # 可视化：路径热力图
    plot_path_heatmap(points, valid_stations, all_routes)

    # 可视化：站点与收集点分布
    visualize_all_routes(points, valid_stations)

    # 输出最终结果
    print("\n======= Final Optimization Results =======")
    print(f"Total Cost: {total_cost:.2f} CNY (Construction: {len(valid_stations)*Tf:.2f} CNY + Transportation: {total_cost - len(valid_stations)*Tf:.2f} CNY)")
    print(f"Total Carbon Emission: {total_emission:.2f} kg")

if __name__ == "__main__":
    main()