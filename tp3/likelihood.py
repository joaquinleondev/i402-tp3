import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy.ndimage import distance_transform_edt


class LikelihoodMapPublisher(Node):
    def __init__(self):
        super().__init__('likelihood_map_publisher')
        qos = rclpy.qos.QoSProfile(depth=1)
        qos.durability = rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(OccupancyGrid, '/likelihood_map', qos)
        self.sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos
        )

    def map_callback(self, msg):
        prob_msg = OccupancyGrid()
        prob_msg.header = msg.header
        prob_msg.info = msg.info

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        grid = np.array(msg.data, dtype=np.int16).reshape((height, width))

        occupied = grid >= 50
        unknown = grid < 0

        if np.any(occupied):
            # For distance_transform_edt, zero cells are the target features.
            distance_to_obstacle = distance_transform_edt(
                ~occupied,
                sampling=resolution
            )

            sigma = 0.25
            likelihood = np.exp(
                -(distance_to_obstacle ** 2) / (2.0 * sigma ** 2)
            )
            likelihood[unknown] = 0.0
            likelihood_data = np.clip(
                likelihood * 100.0,
                0,
                100
            ).astype(np.int8)
        else:
            likelihood_data = np.zeros((height, width), dtype=np.int8)

        prob_msg.data = likelihood_data.flatten().tolist()

        self.pub.publish(prob_msg)
        self.get_logger().info("Published likelihood map")


def main(args=None):
    rclpy.init(args=args)
    node = LikelihoodMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
