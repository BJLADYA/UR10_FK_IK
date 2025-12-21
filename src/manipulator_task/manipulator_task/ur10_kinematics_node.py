import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

import numpy as np
from math import sin, cos, acos, atan2, sqrt, pi

import threading
import tkinter as tk
from tkinter import ttk, messagebox


# DH-параметры
DH = np.array([
#      a      alpha     d      theta
    [ 0.0,    pi/2,   0.1273,   0.0],   # joint 1
    [-0.612,  0.0,    0.0,      0.0],   # joint 2
    [-0.5723, 0.0,    0.0,      0.0],   # joint 3
    [ 0.0,    pi/2,   0.1639,   0.0],   # joint 4
    [ 0.0,   -pi/2,   0.1157,   0.0],   # joint 5
    [ 0.0,    0.0,    0.0922,   0.0],   # joint 6 (TCP)
])


a2 = DH[1, 0]   # -0.612
a3 = DH[2, 0]   # -0.5723
d1 = DH[0, 2]   # 0.1273
d4 = DH[3, 2]   # 0.1639
d5 = DH[4, 2]   # 0.1157
d6 = DH[5, 2]   # 0.0922


def dh_transform(a, alpha, d, theta):
    """Матрица перехода между звеньями"""
    ct, st = cos(theta), sin(theta)
    ca, sa = cos(alpha), sin(alpha)
    return np.array([
        [ct,       -st*ca,      st*sa,      a*ct],
        [st,        ct*ca,     -ct*sa,      a*st],
        [0.0,       sa,         ca,         d   ],
        [0.0,       0.0,        0.0,        1.0 ]
    ])


def dh_inverse_transform(a, alpha, d, theta):
    """Обратная матрица перехода между звеньями"""
    ct, st = cos(theta), sin(theta)
    ca, sa = cos(alpha), sin(alpha)
    return np.array([
        [ct,        st,        0.0,        -a   ],
        [-st*ca,    ct*ca,     sa,         -d*sa],
        [st*sa,    -ct*sa,     ca,         -d*ca],
        [0.0,       0.0,       0.0,         1.0 ]
    ])


def rpy_to_rotation_matrix(roll, pitch, yaw):
    """RPY => матрица вращения (Z-Y-X)"""
    cr, sr = cos(roll),  sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw),   sin(yaw)
    Rx = np.array([ [1,   0,   0 ], 
                    [0,   cr, -sr], 
                    [0,   sr,  cr]])
    Ry = np.array([ [cp,  0,   sp], 
                    [0,   1,   0 ], 
                    [-sp, 0,   cp]])
    Rz = np.array([ [cy, -sy,  0 ], 
                    [sy,  cy,  0 ], 
                    [0,   0,   1 ]])
    return Rz @ Ry @ Rx


def analytical_inverse_kinematics(target_pos, rpy):
    """
    Аналитическая обратная кинематика UR10 (одно фиксированное решение)
    Использует последовательное умножение на инверсные матрицы
    """
    # Из заданного положения и ориентации составляем матрицу
    # перехода от базы к фланцу
    # A06 = T01*T12*T23*T34*T45*T56
    R06 = rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])
    A06 = np.eye(4)
    A06[:3, :3] = R06
    A06[:3, 3] = target_pos # P06

    # === 1. Находим theta1 (T01^{-1}*A06 = T12*T23*T34*T45*T56) ===
    # T01^{-1} зависит от theta1, поэтому аналитически:
    P05 = target_pos - d6 * R06[:, 2] # центр joint 5

    r = sqrt(P05[0]**2 + P05[1]**2)
    if r < abs(d4):
        print('theta1 недостижима')
        return None

    phi = atan2(P05[1], P05[0])
    psi = acos(d4 / r)
    theta1 = phi - psi + pi/2

    # === 2. Находим theta5 (T54^{-1}*T01^{-1}*A06 = T12*T23*T34*T56) ===
    proj_check = P05[0]*sin(theta1) - P05[1]*cos(theta1)
    if abs(proj_check) > d4 + 1e-6:
        print('theta5 недостижима')
        return None

    proj = target_pos[0]*sin(theta1) - target_pos[1]*cos(theta1)
    cos_theta5 = (proj - d4) / d6
    theta5_plus = acos(cos_theta5)      # non-flip
    theta5_minus = -acos(cos_theta5)    # flip
    theta5 = theta5_plus  # выбираем non-flip запястье

    if abs(sin(theta5)) < 1e-8:
        print('theta5 сингулярность')
        return None

    # === 3. Находим theta6 (T65^{-1}*T54^{-1}*T01^{-1}*A06 = T34*T45) ===
    s1, c1 = sin(theta1), cos(theta1)
    # Проекции осей X и Y целевой ориентации на плоскость XY frame1
    num1 = (-R06[1,0]*s1 + R06[1,1]*c1) / sin(theta5)
    num2 = ( R06[0,0]*s1 - R06[0,1]*c1) / sin(theta5)
    if sqrt(num1**2 + num2**2) < 1e-4:
        theta6 = 0.0
    else:
        theta6 = atan2(num1, num2)

    # === 4. Решаем позиционную задачу для theta2, theta3, theta4 ===
    T01 = dh_transform(DH[0,0], DH[0,1], DH[0,2], theta1)
    T45 = dh_transform(DH[4,0], DH[4,1], DH[4,2], theta5)
    T56 = dh_transform(DH[5,0], DH[5,1], DH[5,2], theta6)
    T46 = T45 @ T56
    T04_target = np.linalg.inv(T01) @ A06 @ np.linalg.inv(T46)
    # Теперь T04_target — это желаемая трансформация для frame4
    px, py, pz = T04_target[:3, 3]

    # Геометрическое решение для theta3
    dist_sq = px**2 + py**2 + pz**2
    cos_theta3 = (dist_sq - a2**2 - a3**2) / (2 * abs(a2) * abs(a3))
    if abs(cos_theta3) > 1.0:
        print('theta3 недостижима')
        return None
    theta3 = -acos(cos_theta3)  # положительный (elbow up)

    # theta2
    alpha = atan2(py, px)
    beta = atan2(a3 * sin(theta3), a2 + a3 * cos(theta3))
    theta2 = alpha - beta

    # theta4 из ориентации
    T02 = T01 @ dh_transform(DH[1,0], DH[1,1], DH[1,2], theta2)
    T03 = T02 @ dh_transform(DH[2,0], DH[2,1], DH[2,2], theta3)
    R03 = T03[:3, :3]
    R34_target = R03.T @ T04_target[:3, :3]
    theta4 = atan2(R34_target[1,0], R34_target[0,0]) + pi/2

    joints = np.array([theta1, theta2, theta3, theta4, theta5, theta6])

    return joints


def forward_kinematics(joints):
    """Прямая кинематика: возвращает список T0i для каждого сустава"""
    T_list = []
    T = np.eye(4)
    for i, theta in enumerate(joints):
        T = T @ dh_transform(DH[i, 0], DH[i, 1], DH[i, 2], theta)
        T_list.append(T.copy())
    return T_list


def are_points_collinear(p1, p2, p3, eps=1e-6):
    """
    Проверка, лежат ли три точки на одной прямой
    """
    v1 = p2 - p1
    v2 = p3 - p1
    # Скалярное произведение двух векторов
    return np.linalg.norm(np.cross(v1, v2)) < eps


def circle_from_three_points(p1, p2, p3):
    """
    Поиск окружности, проходящей через три точки в 3D

    Предполагается, что точки не коллинеарны.
    Окружность строится в плоскости, заданной этими точками.

    return:
        center (3, 1)
        radius (float)
        normal (3, 1) — нормаль плоскости окружности
    """

    # Векторы в плоскости
    v1 = p2 - p1
    v2 = p3 - p1

    # Нормаль плоскости
    normal = np.cross(v1, v2)
    norm_n = np.linalg.norm(normal)
    normal = normal / norm_n

    # Середины отрезков
    mid1 = (p1 + p2) / 2.0
    mid2 = (p1 + p3) / 2.0

    # Перпендикуляры в плоскости
    perp1 = np.cross(normal, v1)
    perp2 = np.cross(normal, v2)

    # Решение системы: mid1 + t*perp1 = mid2 + s*perp2
    A = np.vstack([perp1, -perp2]).T
    b = mid2 - mid1

    t_s = np.linalg.lstsq(A, b, rcond=None)[0]
    center = mid1 + t_s[0] * perp1

    radius = np.linalg.norm(center - p1)

    return center, radius, normal


def generate_full_circle(center, radius, normal, num_points=50):
    """
    Генерация полной окружности в 3D

    center     — центр окружности
    radius     — радиус
    normal     — нормаль плоскости
    num_points — количество точек траектории
    """

    # Первый вектор — любой, не параллельный нормали
    if abs(normal[2]) < 0.9:
        ref = np.array([0, 0, 1])
    else:
        ref = np.array([1, 0, 0])

    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)

    v = np.cross(normal, u)

    points = []

    for angle in np.linspace(0, 2*pi, num_points, endpoint=False):
        point = center + radius * (cos(angle) * u + sin(angle) * v)
        points.append(point)

    return np.array(points)


class UR10KinematicsNode(Node):

    def __init__(self):
        super().__init__('ur10_kinematics_node')
        self.current_joints = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        self.get_logger().info('Нода запущена. Открывается GUI...')

        gui_thread = threading.Thread(target=self.run_gui, daemon=True)
        gui_thread.start()


    def joint_state_callback(self, msg):
        state_dict = dict(zip(msg.name, msg.position))

        ordered_joints = []
        try:
            for name in self.joint_names:
                ordered_joints.append(state_dict[name])
            self.current_joints = np.array(ordered_joints)
        except KeyError as e:
            self.get_logger().error(f"Сустав {e} не найден в сообщении /joint_states")


    def process_trajectory(self, p1, p2, p3, rpy_deg, tcp_speed):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        rpy_rad = np.array(rpy_deg) * np.pi / 180.0

        if are_points_collinear(p1, p2, p3):
            self.get_logger().error('Точки коллинеарны')
            return False, "Точки коллинеарны - невозможно построить окружность"
        
        for p in (p1, p2, p3):
            joints = analytical_inverse_kinematics(p, rpy_rad)
            if joints is None:
                self.get_logger().error(f'Точка {p} недостижима, невозможно обработать траекторию')
                return False, f'Точка {p} недостижима, невозможно обработать траекторию'

        center, radius, normal = circle_from_three_points(p1, p2, p3)
        traj_points = generate_full_circle(center, radius, normal)

        joint_points = []
        for i, p in enumerate(traj_points):         
            joints = analytical_inverse_kinematics(p, rpy_rad)
            
            if joints is not None:
                joint_points.append(joints)
            else:
                self.get_logger().error(f'Точка {i+1} недостижима, невозможно обработать траекторию')
                return False, f'Точка траектории недостижима, невозможно обработать траекторию'
        
        self.get_logger().info(f"Успешно обработано {len(traj_points)} точек")
        
        # Ждём текущие углы (максимум 5 сек)
        self.current_joints = None
        timeout = self.get_clock().now() + rclpy.duration.Duration(seconds=5)
        while self.current_joints is None and rclpy.ok() and self.get_clock().now() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.current_joints is None:
            self.get_logger().error('Не удалось получить текущие углы робота из /joint_states')
            return False, 'Не удалось получить текущие углы робота из /joint_states'

        # Добавляем первую точку — текущую позу робота
        current_T_list = forward_kinematics(self.current_joints)
        current_pos = current_T_list[-1][:3, 3]
        # Вставляем в начало траектории
        traj_points = np.vstack([current_pos, traj_points])
        joint_points = [self.current_joints.tolist()] + joint_points

        self.publish_trajectory(traj_points, joint_points, tcp_speed)

        return True, "Траектория отправлена на манипулятор"


    def publish_trajectory(self, traj_points, joint_points, tcp_speed):
        """
        Публикация траектории с заданной линейной скоростью TCP

        tcp_speed — м/с
        """
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        time = 0.0

        for i in range(len(joint_points)):
            point = JointTrajectoryPoint()
            point.positions = joint_points[i]

            if i > 0:
                if i == 1:
                    dt = 1
                else:
                    ds = np.linalg.norm(traj_points[i] - traj_points[i-1])
                    dt = ds / tcp_speed
            else:
                dt = 0.0

            time += dt

            point.time_from_start.sec = int(time)
            point.time_from_start.nanosec = int((time % 1.0) * 1e9)

            traj.points.append(point)

        self.publisher.publish(traj)


    def run_gui(self):
        root = tk.Tk()
        root.title("UR10 — Траектория по окружности")
        root.geometry("500x600")
        root.resizable(False, False)

        ttk.Label(root, text="Три точки для окружности (x, y, z в метрах)", font=("Helvetica", 12, "bold")).pack(pady=10)

        entries = []
        for i in range(1, 4):
            frame = ttk.Frame(root)
            frame.pack(pady=5)
            ttk.Label(frame, text=f"Точка {i}:").pack(side=tk.LEFT)
            point_entries = []
            for coord in ["X", "Y", "Z"]:
                entry = ttk.Entry(frame, width=10)
                entry.pack(side=tk.LEFT, padx=5)
                entry.insert(0, "0.8" if i==1 and coord=="X" else 
                                "0.7" if i==2 and coord=="X" else 
                                "0.6" if i==3 and coord=="X" else
                                "0.1" if i==2 and coord=="Y" else
                                "0.0" if coord=="Y" else "0.2")
                point_entries.append(entry)
            entries.append(point_entries)

        # Ориентация
        ttk.Label(root, text="Ориентация фланца (RPY в градусах)", font=("Helvetica", 10, "bold")).pack(pady=(20,5))
        rpy_frame = ttk.Frame(root)
        rpy_frame.pack(pady=5)
        rpy_entries = []
        for label in ["Roll", "Pitch", "Yaw"]:
            subframe = ttk.Frame(rpy_frame)
            subframe.pack(side=tk.LEFT, padx=10)
            ttk.Label(subframe, text=label).pack()
            entry = ttk.Entry(subframe, width=10)
            entry.pack()
            entry.insert(0, "0.0")
            rpy_entries.append(entry)

        # Скорость
        ttk.Label(root, text="Скорость TCP (м/с)", font=("Helvetica", 10, "bold")).pack(pady=(20,5))
        speed_entry = ttk.Entry(root, width=10)
        speed_entry.pack(pady=5)
        speed_entry.insert(0, "0.1")

        status_frame = ttk.Frame(root)
        status_frame.pack(pady=20, fill=tk.X, padx=20)

        ttk.Label(status_frame, text="Статус:", font=("Helvetica", 11, "bold")).pack(anchor="w")

        text_widget = tk.Text(status_frame, height=6, width=60, wrap=tk.WORD, font=("Helvetica", 10))
        text_widget.pack(side=tk.LEFT, fill=tk.X, expand=True)

        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)

        # Метка для статуса/ошибки
        text_widget.insert(tk.END, "Готов к запуску\n")
        text_widget.config(state=tk.DISABLED, fg="green")

        def on_start():
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, "Обработка...\n")
            text_widget.config(fg="orange")
            text_widget.see(tk.END)
            text_widget.config(state=tk.DISABLED)
            root.update()

            try:
                points = []
                for point_entries in entries:
                    vals = [float(e.get()) for e in point_entries]
                    points.append(vals)

                rpy_deg = [float(e.get()) for e in rpy_entries]
                speed = float(speed_entry.get())

                if speed <= 0:
                    raise ValueError("Скорость должна быть > 0")

                success, message = self.process_trajectory(points[0], points[1], points[2], rpy_deg, speed)

                text_widget.config(state=tk.NORMAL)
                text_widget.delete(1.0, tk.END)
                text_widget.insert(tk.END, message + "\n")
                text_widget.config(fg="green" if success else "red")
                text_widget.see(tk.END)
                text_widget.config(state=tk.DISABLED)

            except ValueError as ve:
                text_widget.config(state=tk.NORMAL)
                text_widget.delete(1.0, tk.END)
                text_widget.insert(tk.END, f"Ошибка ввода: {ve}\n")
                text_widget.config(fg="red")
                text_widget.see(tk.END)
                text_widget.config(state=tk.DISABLED)

            except Exception as e:
                text_widget.config(state=tk.NORMAL)
                text_widget.delete(1.0, tk.END)
                text_widget.insert(tk.END, f"Неожиданная ошибка: {e}\n")
                text_widget.config(fg="red")
                text_widget.see(tk.END)
                text_widget.config(state=tk.DISABLED)

        ttk.Button(root, text="START", command=on_start, style="Accent.TButton").pack(pady=30)

        root.mainloop()


def main():
    rclpy.init()
    node = UR10KinematicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
