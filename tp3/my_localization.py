#!/usr/bin/env python3

import math

import numpy as np

# Compatibilidad con dependencias que todavia usan alias viejos de numpy.
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

import rclpy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from tp3.robot_functions import RobotFunctions


def yaw_to_quaternion(yaw):
    """Convertir un angulo yaw a un mensaje Quaternion."""
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q


class Odom3Node(Node):
    """Nodo principal de localizacion por filtro de particulas."""

    def __init__(self):
        super().__init__("lidar_tf")

        self.declare_parameter("num_particles", 1000)
        num_particles = (
            self.get_parameter("num_particles")
            .get_parameter_value()
            .integer_value
        )

        self.subscription_odom = self.create_subscription(
            Odometry,
            "/calc_odom",
            self.odom_callback,
            10
        )
        self.subscription_real_odom = self.create_subscription(
            Odometry,
            "/odom",
            self.real_odom_callback,
            10
        )
        self.subscription_scan = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10
        )

        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.likelihood_map = self.create_subscription(
            OccupancyGrid,
            "/likelihood_map",
            self.map_callback,
            map_qos
        )

        self.last_odom = (0, 0, 0)
        self.read_odom = False
        self.map_received = False
        self.robot = RobotFunctions(num_particles)

        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            "/particle_cloud",
            10
        )

        self.real_path_pub = self.create_publisher(
            Path,
            "/real_robot_path",
            10
        )
        self.real_path_msg = Path()
        self.real_path_msg.header.frame_id = "map"

        self.calc_path_pub = self.create_publisher(
            Path,
            "/calc_robot_path",
            10
        )
        self.calc_path_msg = Path()
        self.calc_path_msg.header.frame_id = "map"

        self.particle_path_pub = self.create_publisher(
            Path,
            "/particle_robot_path",
            10
        )
        self.particle_path_msg = Path()
        self.particle_path_msg.header.frame_id = "map"

    def map_callback(self, msg):
        """Guardar el campo de verosimilitud publicado por likelihood.py."""
        if self.map_received:
            return

        self.map_data = msg
        self.map_received = True
        self.grid = np.array(self.map_data.data).reshape(
            (self.map_data.info.height, self.map_data.info.width)
        )
        self.get_logger().info("Likelihood map received.")

    def create_pointcloud2(self, points):
        """Crear un PointCloud2 con intensidad proporcional al peso."""
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        if len(points.shape) == 3:
            points = points.reshape(-1, 3)

        intensities = np.array(self.robot.get_weights(), dtype=np.float32)
        if np.max(intensities) > 0:
            intensities /= np.max(intensities)

        intensities = intensities.reshape(-1, 1).astype(np.float32)
        points_with_intensity = np.hstack(
            [points.astype(np.float32), intensities]
        )

        msg.height = 1
        msg.width = points_with_intensity.shape[0]
        msg.fields = [
            PointField(
                name='x',
                offset=0,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='y',
                offset=4,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='z',
                offset=8,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='intensity',
                offset=12,
                datatype=PointField.FLOAT32,
                count=1
            )
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points_with_intensity.tobytes()

        return msg

    def plot_particles(self):
        """Publicar nube de particulas y trayectoria estimada."""
        samples = self.robot.get_particle_states()
        samples_3d = np.hstack(
            [samples[:, :2], np.zeros((samples.shape[0], 1))]
        )
        pointcloud_msg = self.create_pointcloud2(samples_3d)
        self.pointcloud_pub.publish(pointcloud_msg)

        selected_state = self.robot.get_selected_state()
        mean_theta = np.arctan2(
            np.sin(selected_state[2]),
            np.cos(selected_state[2])
        )

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = float(selected_state[0])
        pose_stamped.pose.position.y = float(selected_state[1])
        pose_stamped.pose.position.z = 0.0
        pose_stamped.pose.orientation = yaw_to_quaternion(mean_theta)

        self.particle_path_msg.poses.append(pose_stamped)
        self.particle_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.particle_path_pub.publish(self.particle_path_msg)

    def real_odom_callback(self, data: Odometry):
        """Publicar la trayectoria real del robot simulado."""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = data.pose.pose

        self.real_path_msg.poses.append(pose_stamped)
        self.real_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.real_path_pub.publish(self.real_path_msg)

    def odom_callback(self, data: Odometry):
        """Actualizar particulas usando incrementos de odometria."""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = data.pose.pose

        self.calc_path_msg.poses.append(pose_stamped)
        self.calc_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.calc_path_pub.publish(self.calc_path_msg)

        x = data.pose.pose.position.x
        y = data.pose.pose.position.y

        q_w = data.pose.pose.orientation.w
        q_x = data.pose.pose.orientation.x
        q_y = data.pose.pose.orientation.y
        q_z = data.pose.pose.orientation.z

        current_rotation = R.from_quat([q_x, q_y, q_z, q_w])
        theta = current_rotation.as_euler('xyz', degrees=False)[2]

        deltas = {'t': 0, 'r1': 0, 'r2': 0}

        if self.read_odom:
            dx = x - self.last_odom[0]
            dy = y - self.last_odom[1]
            delta_t = np.sqrt(dx ** 2 + dy ** 2)

            if delta_t > 1e-6:
                delta_rot1 = np.arctan2(dy, dx) - self.last_odom[2]
                delta_rot2 = theta - self.last_odom[2] - delta_rot1
            else:
                delta_rot1 = 0.0
                delta_rot2 = theta - self.last_odom[2]

            delta_rot1 = np.arctan2(np.sin(delta_rot1), np.cos(delta_rot1))
            delta_rot2 = np.arctan2(np.sin(delta_rot2), np.cos(delta_rot2))

            deltas['t'] = delta_t
            deltas['r1'] = delta_rot1
            deltas['r2'] = delta_rot2

            self.robot.move_particles(deltas)
            self.plot_particles()

        self.last_odom = (x, y, theta)
        if not self.read_odom:
            self.read_odom = True

    def scan_with_calc(self, data: LaserScan):
        """Aplicar el modelo de sensor cuando ya existe mapa."""
        if self.map_received:
            self.robot.update_particles(data, self.map_data, self.grid)

    def scan_callback(self, data):
        """Corregir la terna angular del laser y procesar el scan."""
        data.angle_min += np.pi
        data.angle_max += np.pi
        self.scan_with_calc(data)


def main(args=None):
    """Ejecutar el nodo de localizacion."""
    rclpy.init(args=args)
    node = Odom3Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
