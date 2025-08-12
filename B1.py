import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from IPython.display import display, Javascript

# 设置随机种子以保证可重复性
np.random.seed(42)

# 允许无限滚动输出
display(Javascript('IPython.OutputArea.prototype._should_scroll = function(lines) { return false; }'))

# ================== 参数设置 ==================
n = 30  # 收集点数量
Q = 5   # 车辆最大载重
max_iter = 150  # 增加迭代次数
swarm_size = 60  # 增加粒子数量
num_runs = 5  # 运行次数取最优解

# 可视化配置常量
VISUALIZATION_CONFIG = {
    'point_color': 'ko',
    'point_size': 6,
    'plant_marker': 'gs',
    'plant_size': 10,
    'line_width': 1.5,
    'title_fontsize': 12,
    'label_fontsize': 8,
    'grid': True,
    'legend_loc': 'best',
    'color_map': 'tab20'
}

# ================== 坐标和数据 ==================
points = np.array([
    [0, 0, 0],  # 处理厂（索引0）
    [12, 8, 1.2], [5, 15, 2.3], [20, 30, 1.8], [25, 10, 3.1],
    [35, 22, 2.7], [18, 5, 1.5], [30, 35, 2.9], [10, 25, 1.1],
    [22, 18, 2.4], [38, 15, 3.0], [5, 8, 1.7], [15, 32, 2.1],
    [28, 5, 3.2], [30, 12, 2.6], [10, 10, 1.9], [20, 20, 2.5],
    [35, 30, 3.3], [8, 22, 1.3], [25, 25, 2.8], [32, 8, 3.4],
    [15, 5, 1.6], [28, 20, 2.2], [38, 25, 3.5], [10, 30, 1.4],
    [20, 10, 2.0], [30, 18, 3.6], [5, 25, 1.0], [18, 30, 2.3],
    [35, 10, 3.7], [22, 35, 1.9]
])

num_points = points.shape[0]
coords = points[:, :2]
D = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))

# ================== 数据结构 ==================
@dataclass
class Particle:
    position: np.ndarray  # 当前位置（收集点排列顺序）
    velocity: np.ndarray  # 速度
    cost: float  # 适应度值（总距离）
    best_position: np.ndarray  # 个体最优位置
    best_cost: float  # 个体最优适应度值

def calculate_route_distance(path: List[int], D: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return sum(D[path[i]][path[i+1]] for i in range(len(path)-1))

def insert_point_optimally(path, point_id, D):
    min_increase = float('inf')
    best_idx = 0
    for i in range(len(path) - 1):
        before = path[i]
        after = path[i + 1]
        increase = D[before][point_id] + D[point_id][after] - D[before][after]
        if increase < min_increase:
            min_increase = increase
            best_idx = i + 1
    path.insert(best_idx, point_id)
    return path

def decode_particle(position: np.ndarray, points: np.ndarray, D: np.ndarray, Q: float) -> Tuple[List[Dict], float]:
    if len(position) != n:
        raise ValueError("Position length must match number of points")

    routes = []
    total_distance = 0
    current_path = [0]
    current_load = 0

    for idx in range(len(position)):
        point_id = position[idx]
        if point_id < 1 or point_id >= num_points:
            raise ValueError(f"Invalid point ID: {point_id}")

        weight = points[point_id, 2]

        if current_load + weight > Q:
            current_path.append(0)
            dist = calculate_route_distance(current_path, D)
            routes.append({'path': current_path, 'load': current_load, 'distance': dist})
            total_distance += dist
            current_path = [0, point_id]
            current_load = weight
        else:
            current_path = insert_point_optimally(current_path, point_id, D)
            current_load += weight

    if current_path:
        current_path.append(0)
        dist = calculate_route_distance(current_path, D)
        routes.append({'path': current_path, 'load': current_load, 'distance': dist})
        total_distance += dist

    return routes, total_distance

def get_swap_sequence(pos1: np.ndarray, pos2: np.ndarray) -> List[Tuple[int, int]]:
    if len(pos1) != len(pos2):
        raise ValueError("Positions must have the same length")
    pos1_indices = {val: idx for idx, val in enumerate(pos1)}
    swaps = []
    tmp = pos1.copy()
    for i in range(len(tmp)):
        if tmp[i] != pos2[i]:
            j = pos1_indices[pos2[i]]
            swaps.append((i, j))
            pos1_indices[tmp[i]] = j
            pos1_indices[tmp[j]] = i
            tmp[i], tmp[j] = tmp[j], tmp[i]
    return swaps

def apply_swaps(pos: np.ndarray, swaps: List[Tuple[int, int]], prob: float) -> np.ndarray:
    new_pos = pos.copy()
    for i, j in swaps:
        if np.random.rand() < prob:
            new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
    return new_pos

def mutate(position: np.ndarray, mutation_rate=0.05) -> np.ndarray:
    new_pos = position.copy()
    for i in range(len(new_pos)):
        if np.random.rand() < mutation_rate:
            j = np.random.randint(0, len(new_pos))
            new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
    return new_pos

def get_pso_params(iter_num, max_iter):
    w_max, w_min = 0.9, 0.4
    w = w_max - (w_max - w_min) * iter_num / max_iter
    c1_base, c2_base = 1.2, 1.2
    return w, c1_base, c2_base

def run_pso():
    particles = []
    for _ in range(swarm_size):
        pos = np.random.permutation(n) + 1
        particles.append(Particle(
            position=pos,
            velocity=np.zeros(n),
            cost=float('inf'),
            best_position=pos.copy(),
            best_cost=float('inf')
        ))

    global_best_cost = particles[0].best_cost
    global_best_position = particles[0].best_position.copy()

    for iter_num in range(max_iter):
        w, c1, c2 = get_pso_params(iter_num, max_iter)
        for p in particles:
            try:
                routes, cost = decode_particle(p.position, points, D, Q)
                p.cost = cost
                if cost < p.best_cost:
                    p.best_position = p.position.copy()
                    p.best_cost = cost
                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_position = p.position.copy()
            except Exception as e:
                print(f"Error processing particle: {e}")
                continue

        for p in particles:
            delta_p = get_swap_sequence(p.position, p.best_position)
            delta_g = get_swap_sequence(p.position, global_best_position)
            new_pos = apply_swaps(p.position, delta_p, w)
            new_pos = apply_swaps(new_pos, delta_g, c2 * np.random.rand())
            p.position = mutate(new_pos)  # 添加变异

    return global_best_cost, global_best_position

# ================== 多次运行取最优 ==================
best_cost = float('inf')
best_position = None
for run in range(num_runs):
    print(f"\n=== 运行第 {run+1} 次 ===")
    cost, pos = run_pso()
    if cost < best_cost:
        best_cost = cost
        best_position = pos

print(f"\n最终最短距离: {best_cost:.2f}km")

# ================== 输出结果 ==================
try:
    best_routes, _ = decode_particle(best_position, points, D, Q)
    print("\n===== 最优路径方案 =====")
    for k, route in enumerate(best_routes):
        path = route['path']
        nodes = ' → '.join(map(str, path[1:-1]))
        print(f"车辆{k+1}: 0 → {nodes} → 0，载重{route['load']:.2f}吨，距离{route['distance']:.2f}km")
except Exception as e:
    print(f"Error decoding best particle: {e}")

# ================== 可视化路径 ==================
def visualize_routes(routes: List[Dict], points: np.ndarray):
    try:
        plt.figure(figsize=(10, 8))
        plt.title("Optimization of Garbage Collection Path", fontsize=VISUALIZATION_CONFIG['title_fontsize'])

        # 绘制所有收集点
        for i in range(1, len(points)):
            plt.plot(points[i, 0], points[i, 1],
                     VISUALIZATION_CONFIG['point_color'],
                     markersize=VISUALIZATION_CONFIG['point_size'])
            plt.text(points[i, 0], points[i, 1], str(i),
                     fontsize=VISUALIZATION_CONFIG['label_fontsize'])

        # 绘制处理厂
        plt.plot(points[0, 0], points[0, 1],
                 VISUALIZATION_CONFIG['plant_marker'],
                 markersize=VISUALIZATION_CONFIG['plant_size'])
        plt.text(points[0, 0], points[0, 1], "Treatment plant",
                 fontsize=VISUALIZATION_CONFIG['label_fontsize'], ha="right")

        # 使用颜色区分不同车辆路径
        colors = plt.cm.get_cmap(VISUALIZATION_CONFIG['color_map'])(np.linspace(0, 1, len(routes)))
        for k, route in enumerate(routes):
            path = route['path']
            x = points[path, 0]
            y = points[path, 1]
            plt.plot(x, y, '-', color=colors[k], linewidth=VISUALIZATION_CONFIG['line_width'], label=f'Route{k+1}')

        plt.legend(loc=VISUALIZATION_CONFIG['legend_loc'])
        if VISUALIZATION_CONFIG['grid']:
            plt.grid(True)
        plt.axis("equal")
        plt.show()
    except Exception as e:
        print(f"Visualization error: {e}")
visualize_routes(best_routes, points)