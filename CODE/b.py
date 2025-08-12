import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
# 读取上传的 Excel 文件
file_path = r'C:\Users\XHYCA\Desktop\DGB\附件1.xlsx'
data = pd.read_excel(file_path)

# 清洗数据，提取有效信息
data_clean = data.iloc[2:, [0, 1, 2, 3]].reset_index(drop=True)
data_clean.columns = ['编号', 'x', 'y', '垃圾量']

# 将x, y坐标转换为数值型数据，垃圾量也转换为数值
data_clean['x'] = pd.to_numeric(data_clean['x'], errors='coerce')
data_clean['y'] = pd.to_numeric(data_clean['y'], errors='coerce')
data_clean['垃圾量'] = pd.to_numeric(data_clean['垃圾量'], errors='coerce')

# 提取收集点数据
coordinates = data_clean[['x', 'y']].values
demands = data_clean['垃圾量'].values
num_locations = len(coordinates)


# 计算欧几里得距离
def compute_distance_matrix(coords):
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                distance_matrix[i][j] = np.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            else:
                distance_matrix[i][j] = 0
    return distance_matrix


# 贪心算法
def greedy_vrp():
    remaining_points = list(range(1, num_locations))  # 剩余的收集点
    vehicle_routes = []
    total_distance = 0
    current_location = 0  # 初始位置为垃圾处理厂（编号为0）

    # 车辆最多为5辆，按需分配任务
    vehicle_capacity = 5
    demands_remaining = demands.copy()

    while remaining_points:
        current_route = [current_location]
        current_load = 0

        # 当前车辆的路径
        while remaining_points:
            # 选择距离当前点最近的点
            distances_to_remaining = [(i, distance_matrix[current_location][i]) for i in remaining_points]
            distances_to_remaining.sort(key=lambda x: x[1])

            next_point, _ = distances_to_remaining[0]
            # 判断是否超载
            if current_load + demands[next_point - 1] <= vehicle_capacity:
                current_route.append(next_point)
                current_load += demands[next_point - 1]
                remaining_points.remove(next_point)
                current_location = next_point
            else:
                break

        # 完成当前路径并返回垃圾处理厂
        current_route.append(0)
        vehicle_routes.append(current_route)
        # 计算总距离
        route_distance = sum(
            distance_matrix[current_route[i]][current_route[i + 1]] for i in range(len(current_route) - 1))
        total_distance += route_distance

        # 重置起始位置为垃圾处理厂
        current_location = 0

    return vehicle_routes, total_distance


# 执行贪心算法
distance_matrix = compute_distance_matrix(coordinates)
routes, total_distance = greedy_vrp()

# 输出解决方案
print("Routes:", routes)
print("Total distance:", total_distance)


# 增加更多的可视化内容：每辆车的总行驶距离，任务量展示和透明度处理
def plot_routes_with_more_info(routes, coordinates, num_locations, demands, total_distance):
    plt.figure(figsize=(12, 12))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    total_route_distance = 0

    for vehicle_id, route in enumerate(routes):
        route_coords = coordinates[route]
        route_distance = sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        total_route_distance += route_distance

        # 使用不同颜色区分每辆车的路径，设置透明度
        plt.plot(route_coords[:, 0], route_coords[:, 1], marker='o', markersize=6,
                 label=f'Vehicle {vehicle_id + 1} (Distance: {route_distance:.2f} km)',
                 color=colors[vehicle_id % len(colors)], alpha=0.7)

        # 标注每个路径的起点和终点
        plt.text(route_coords[0, 0], route_coords[0, 1], f"Start", color="black", fontsize=12)
        plt.text(route_coords[-1, 0], route_coords[-1, 1], f"End", color="blue", fontsize=12)

        # 标注每个收集点编号
        for i in route:
            plt.text(coordinates[i, 0], coordinates[i, 1], f'{i}', fontsize=10, ha='right', color='black')

        # 显示每辆车运输的垃圾量（任务量）
        for i in route[1:-1]:  # 从第一个点到倒数第二个点
            plt.annotate(f"{demands[i - 1]:.2f} tons", (coordinates[i, 0], coordinates[i, 1]),
                         textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)

    # 标记垃圾处理厂位置
    plt.scatter(coordinates[0, 0], coordinates[0, 1], color='black', label='Depot', s=100)
    plt.title(f"Vehicle Routes (Greedy Algorithm with More Info)\nTotal Distance: {total_route_distance:.2f} km")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend()
    plt.show()
# 绘制包含更多信息的路径图
plot_routes_with_more_info(routes, coordinates, num_locations, demands, total_distance)


# 进一步增加可视化内容：收集点分布图，距离矩阵热图，车辆任务量分布
def plot_additional_visualizations(coordinates, distance_matrix, demands, num_locations, routes):
    # 1. 收集点分布图（包括垃圾处理厂）
    plt.figure(figsize=(8, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='blue', label='Collection Points')
    plt.scatter(coordinates[0, 0], coordinates[0, 1], color='red', label='Depot', s=100)  # 垃圾处理厂
    plt.title("Collection Points and Depot Distribution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. 距离矩阵热图
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Distance')
    plt.title("Distance Matrix Heatmap")
    plt.xlabel("Destination")
    plt.ylabel("Source")
    plt.show()

    # 3. 每辆车的运输任务量分布
    vehicle_demands = [sum(demands[route[1:-1]]) for route in routes]  # 每辆车运输的任务量（垃圾量）

    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(vehicle_demands) + 1), vehicle_demands, color='skyblue')
    plt.xlabel("Vehicle ID")
    plt.ylabel("Total Garbage Transported (tons)")
    plt.title("Total Garbage Transported by Each Vehicle")
    plt.xticks(range(1, len(vehicle_demands) + 1))
    plt.grid(True)
    plt.show()


# 绘制附加的可视化图像
plot_additional_visualizations(coordinates, distance_matrix, demands, num_locations, routes)

# 完整的时间复杂度分析：计算距离矩阵需要O(n^2)，贪心算法的主要操作是每次选择最近的点，时间复杂度约为O(n^2)