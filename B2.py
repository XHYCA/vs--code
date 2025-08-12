import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

# ================== 参数设置 ==================
n = 30           # 收集点数量
k_types = 4      # 垃圾类型数量
max_iter = 100   # 最大迭代次数
swarm_size = 50  # 粒子数量
w_pso = 0.729    # 惯性权重
c1 = c2 = 1.49445  # 学习因子

# ================== 数据初始化 ==================
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

# 车辆参数（来自附件2）
# [载重, 容积, 单位距离成本]
vehicle_params = np.array([
    [8, 20, 2.5],   # 厨余垃圾
    [6, 25, 2.0],   # 可回收物
    [3, 10, 5.0],   # 有害垃圾
    [10, 18, 1.8],  # 其他垃圾
])

num_points = points.shape[0]

# 计算距离矩阵
D = np.zeros((num_points, num_points))
for i in range(num_points):
    for j in range(num_points):
        D[i, j] = np.linalg.norm(points[i, :2] - points[j, :2])

# ================== 数据结构 ==================
@dataclass
class Particle:
    position: np.ndarray  # (n, k_types) 分配矩阵
    velocity: np.ndarray
    cost: float
    best_position: np.ndarray
    best_cost: float

particles = []

# 初始化粒子群
for _ in range(swarm_size):
    pos = np.zeros((n, k_types), dtype=int)
    
    # 每个收集点只分配到一个垃圾类型
    for j in range(n):
        type_idx = np.random.randint(k_types)
        pos[j, type_idx] = 1
    
    particles.append(Particle(
        position=pos,
        velocity=np.zeros((n, k_types)),
        cost=float('inf'),
        best_position=pos.copy(),
        best_cost=float('inf')
    ))

global_best_cost = float('inf')
global_best_position = None

# ================== 核心函数实现 ==================

def decode_particle(position: np.ndarray, points: np.ndarray, D: np.ndarray, vehicle_params: np.ndarray):
    routes = [[] for _ in range(k_types)]
    total_cost = 0

    for k in range(k_types):
        idx = np.where(position[:, k] == 1)[0]
        
        if len(idx) == 0:
            continue

        type_routes, type_cost = decode_type_routes(k, idx, points, D, vehicle_params[k])
        routes[k] = type_routes
        total_cost += type_cost

    return routes, total_cost


def decode_type_routes(k: int, idx: np.ndarray, points: np.ndarray, D: np.ndarray, params: np.ndarray):
    Q = params[0]  # 载重限制
    V = params[1]  # 容积限制
    C = params[2]  # 单位距离成本

    current_load = 0
    current_volume = 0
    current_path = [0]
    routes = []
    total_cost = 0

    remaining_idx = list(idx)
    current_point = 0  # 处理厂

    while remaining_idx:
        feasible_points = []
        for point_index in remaining_idx:
            point_id = point_index + 1
            
            # 根据垃圾类型获取正确的垃圾量
            if k == 0:  # 厨余垃圾
                w = points[point_id, 2]
            elif k == 1:  # 可回收物
                w = points[point_id, 3]
            elif k == 2:  # 有害垃圾
                w = points[point_id, 4]
            else:  # 其他垃圾
                w = points[point_id, 5]
                
            v = w / 0.5  # 假设体积是重量除以密度
            
            if current_load + w <= Q and current_volume + v <= V:
                feasible_points.append((point_index, point_id, w, v, D[current_point][point_id]))
        
        if not feasible_points:
            # 如果没有可行点，则结束当前路径并开始新路径
            current_path.append(0)
            dist = calculate_route_distance(current_path, D)
            cost = dist * C
            routes.append({'path': current_path, 'load': current_load, 'volume': current_volume, 'distance': dist, 'cost': cost})
            
            current_path = [0]
            current_load = 0
            current_volume = 0
            current_point = 0
            continue
        
        # 选择最近的可行点
        next_point = min(feasible_points, key=lambda x: x[4])  # 按照距离排序
        point_index, point_id, w, v, _ = next_point
        
        current_path.append(point_id)
        current_load += w
        current_volume += v
        current_point = point_id
        remaining_idx.remove(point_index)

    # 添加最后一条路径
    if len(current_path) > 1:
        current_path.append(0)
        dist = calculate_route_distance(current_path, D)
        cost = dist * C
        routes.append({'path': current_path, 'load': current_load, 'volume': current_volume, 'distance': dist, 'cost': cost})
        total_cost += cost

    return routes, total_cost


def calculate_route_distance(path: List[int], D: np.ndarray) -> float:
    return sum(D[path[i]][path[i+1]] for i in range(len(path)-1))


# ================== PSO 主循环 ==================
for iter_num in range(max_iter):
    for p in particles:
        routes, cost = decode_particle(p.position, points, D, vehicle_params)
        p.cost = cost

        if cost < p.best_cost:
            p.best_position = p.position.copy()
            p.best_cost = cost

        if cost < global_best_cost:
            global_best_cost = cost
            global_best_position = p.position.copy()

    # 更新粒子位置
    for p in particles:
        r1 = np.random.rand(n, k_types)
        r2 = np.random.rand(n, k_types)

        # Sigmoid 映射
        personal_best = 1 / (1 + np.exp(- (p.best_position.astype(float) - p.position.astype(float))))
        global_best = 1 / (1 + np.exp(- (global_best_position.astype(float) - p.position.astype(float))))

        p.velocity = w_pso * p.velocity + \
                     c1 * r1 * personal_best + \
                     c2 * r2 * global_best

        # 二值化
        p.position = 1 / (1 + np.exp(-p.velocity)) > 0.5
        p.position = p.position.astype(int)

        # 确保每个点至少分配到一个垃圾类型
        for j in range(n):
            if np.sum(p.position[j]) == 0:
                type_idx = np.random.randint(k_types)
                p.position[j, type_idx] = 1

    print(f"迭代 {iter_num+1}/{max_iter}, 最低成本: {global_best_cost:.2f}元")

# ================== 输出结果 ==================
best_routes, _ = decode_particle(global_best_position, points, D, vehicle_params)

print("\n===== 最优路径方案 =====")
for k, routes in enumerate(best_routes):
    print(f"垃圾类型 {k+1}:")
    for r, route in enumerate(routes):
        path = route['path']
        nodes = ' → '.join(map(str, path[1:-1]))
        print(f"  车辆{r+1}: 0 → {nodes} → 0，载重{route['load']:.2f}吨，体积{route['volume']:.2f}m³，距离{route['distance']:.2f}km，成本{route['cost']:.2f}元")


# ================== 可视化路径 ==================
def visualize_routes(routes: List[List[Dict]], points: np.ndarray, vehicle_params: np.ndarray):
    plt.figure(figsize=(10, 8))
    plt.title("Collaborative planning diagram for vehicle classification", fontsize=14)
    
    # 绘制所有收集点
    for i in range(1, len(points)):
        plt.plot(points[i, 0], points[i, 1], "ko", markersize=6)
        plt.text(points[i, 0], points[i, 1], str(i), fontsize=8)
    
    # 绘制处理厂
    plt.plot(points[0, 0], points[0, 1], "gs", markersize=10)
    plt.text(points[0, 0], points[0, 1], "Processing Plant", fontsize=10, ha="right")

    # 定义颜色和对应标签
    colors = ['blue', 'red', 'magenta', 'cyan']  # 对应四种垃圾类型
    labels = [
        'kitchen waste',
        'recyclable waste',
        'hazardous waste',
        'other waste'
    ]
    
    legend_added = set()  # 避免重复添加图例
    
    for k, type_routes in enumerate(routes):
        for r, route in enumerate(type_routes):
            path = route['path']
            x = points[path, 0]
            y = points[path, 1]
            label = labels[k]  # 使用对应的中文或英文标签
            
            if label not in legend_added:
                plt.plot(x, y, '-', color=colors[k], linewidth=1.5, label=label)
                legend_added.add(label)
            else:
                plt.plot(x, y, '-', color=colors[k], linewidth=1.5)
                
    plt.legend(loc='best')
    plt.grid(True)
    plt.axis("equal")
    plt.show()


visualize_routes(best_routes, points, vehicle_params)