import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

import numpy as np
from math import sin, cos, acos, atan2, sqrt, pi
from scipy.optimize import least_squares


# DH-параметры
# a, alpha, d, theta_offset
DH = np.array([
    [ 0.0,    pi/2,   0.1273,   0.0],   # joint 1
    [-0.612,  0.0,    0.0,      0.0],   # joint 2
    [-0.5723, 0.0,    0.0,      0.0],   # joint 3
    [ 0.0,    pi/2,   0.1639,   0.0],   # joint 4
    [ 0.0,   -pi/2,   0.1157,   0.0],   # joint 5
    [ 0.0,    0.0,    0.0922,   0.0],   # joint 6 (TCP)
])

def dh_transform(a, alpha, d, theta):
    """
    Однородная матрица преобразования по DH-параметрам
    """
    return np.array([
        [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
        [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
        [0.0,         sin(alpha),             cos(alpha),            d],
        [0.0,         0.0,                     0.0,                  1.0]
    ])


def forward_kinematics(joints):
    """Прямая кинематика: возвращает список T0i для каждого сустава"""
    T_list = []
    T = np.eye(4)
    for i, theta in enumerate(joints):
        T = T @ dh_transform(DH[i, 0], DH[i, 1], DH[i, 2], theta)
        T_list.append(T.copy())
    return T_list


def jacobian(q):
    """Якобиан для численного IK"""
    TF_lists = forward_kinematics(q)
    z = [np.array([0, 0, 1])]  # z0 в базе
    o = [np.array([0, 0, 0])]  # o0
    for i in range(6):
        T = TF_lists[i]
        z.append(T[:3, 2])  # ось Zi
        o.append(T[:3, 3])  # центр сустава i+1
    
    J = np.zeros((6, 6))
    for i in range(6):
        J[0:3, i] = np.cross(z[i], o[-1] - o[i])  # линейная часть
        J[3:6, i] = z[i]                          # угловая часть
    return J


def numerical_inverse_kinematics(target_pos, target_rpy, q_init=None, max_iter=500, tol=1e-3):
    """
    Полная численная IK (позиция + ориентация)
    """
    if q_init is None:
        q_init = np.array([0.0, -pi/2, 0.0, -pi/2, 0.0, 0.0])  # home позиция
    
    # Целевая матрица T06
    target_R = rpy_to_rotation_matrix(target_rpy[0], target_rpy[1], target_rpy[2])
    target_T = np.eye(4)
    target_T[:3, :3] = target_R
    target_T[:3, 3] = target_pos
    
    def error_function(q):
        TF = forward_kinematics(q)[-1]  # T06
        pos_err = target_T[:3, 3] - TF[:3, 3]
        
        # Ошибка ориентации (axis-angle в base frame)
        R_err = target_T[:3, :3] @ TF[:3, :3].T  # R_target * R_current^T
        theta = acos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        if theta < 1e-6:
            rot_err = np.zeros(3)
        else:
            multi = theta / (2 * sin(theta))
            rot_err = multi * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ])
        
        return np.concatenate((pos_err * 10, rot_err))  # вес позиции выше
    
    q = q_init.copy()
    for i in range(max_iter):
        error = error_function(q)
        error_norm = np.linalg.norm(error)
        
        if error_norm < tol:
            print(f"IK сошлось за {i} итераций, ошибка: {error_norm:.6f}")
            q = (q + pi) % (2 * pi) - pi  # нормализация
            return q
        
        J = jacobian(q)
        # Полный 6x6 якобиан
        try:
            dq = 0.1 * np.linalg.pinv(J, rcond=1e-3) @ error  # уменьшенный шаг для стабильности
        except np.linalg.LinAlgError:
            print("Сингулярность якобиана")
            return None
        
        q += dq
        q = np.clip(q, -pi*2, pi*2)  # ограничение углов
    
    print(f"IK не сошлось, финальная ошибка: {error_norm:.6f}")
    return None


def rpy_to_rotation_matrix(roll, pitch, yaw):
    """RPY в матрицу ротации"""
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


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

        self.get_logger().info('UR10 kinematics node started')

        self.demo()
        self.get_logger().info('Работа завершена')


    def joint_state_callback(self, msg):
        if self.current_joints is None:  # берём только один раз
            # Порядок joint_names должен совпадать!
            positions = []
            for name in self.joint_names:
                try:
                    idx = msg.name.index(name)
                    positions.append(msg.position[idx])
                except ValueError:
                    self.get_logger().warn(f'Joint {name} not found in joint_states')
                    return
            self.current_joints = np.array(positions)
            self.get_logger().info(f'Текущие углы получены: {positions}')


    def demo(self):
        # Три произвольные точки
        p1 = np.array([0.6, 0.0, 0.3])
        p3 = np.array([0.5, -0.1, 0.3])
        p2 = np.array([0.4, 0.0, 0.3])

        if are_points_collinear(p1, p2, p3):
            self.get_logger().error('Точки коллинеарны')
            return

        center, radius, normal = circle_from_three_points(p1, p2, p3)
        traj_points = generate_full_circle(center, radius, normal)

        joint_points = []
        rpy = np.array([pi, 0.0, 0.0])  # фиксированная ориентация
        
        # Начальное приближение для IK (важно для конфигурации!)
        q_init = np.array([0.0, -pi/2, 0.0, -pi/2, 0.0, 0.0])  # home
        
        success_count = 0
        for i, p in enumerate(traj_points):         
            joints = numerical_inverse_kinematics(p, rpy, q_init)
            
            if joints is not None:
                joint_points.append(joints)
                success_count += 1
                # Обновляем начальное приближение для следующей точки
                q_init = joints.copy()
            else:
                self.get_logger().error(f'Точка {i+1} вне рабочей зоны или не сошлось IK')
                # return
        
        if success_count == 0:
            self.get_logger().error('Ни одна точка не была обработана успешно')
            return
        
        print(f"Успешно обработано {success_count}/{len(traj_points)} точек")
        
        # Ждём текущие углы (максимум 5 сек)
        self.current_joints = None
        timeout = self.get_clock().now() + rclpy.duration.Duration(seconds=5)
        while self.current_joints is None and rclpy.ok() and self.get_clock().now() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.current_joints is None:
            self.get_logger().error('Не удалось получить текущие углы робота из /joint_states')
            return

        # Добавляем первую точку — текущую позу робота
        # Для этого нужно вычислить позицию/ориентацию текущей позы (FK)
        current_T_list = forward_kinematics(self.current_joints)
        current_pos = current_T_list[-1][:3, 3]

        # Вставляем в начало траектории
        traj_points = np.vstack([current_pos, traj_points])
        joint_points = [self.current_joints.tolist()] + joint_points

        self.get_logger().info(f'Добавлена стартовая точка. Всего точек: {len(traj_points)}')

        tcp_speed = 0.1
        self.publish_trajectory(traj_points[:success_count + 1], joint_points, tcp_speed)


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
                ds = np.linalg.norm(traj_points[i] - traj_points[i-1])
                dt = ds / tcp_speed
            else:
                dt = 0.0

            time += dt

            point.time_from_start.sec = int(time)
            point.time_from_start.nanosec = int((time % 1.0) * 1e9)

            traj.points.append(point)

        self.publisher.publish(traj)


def main():
    rclpy.init()
    node = UR10KinematicsNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
