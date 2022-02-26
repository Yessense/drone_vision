from object_detector import *
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image

rospy.init_node("drone_vision")

capture = cv2.VideoCapture(0)
# Correct RGB image from RealSense
# capture = cv2.VideoCapture(6)
capture.set(3, 640)
capture.set(4, 480)

# Create topic for debug images
# topic = rospy.Publisher("camera/color/drone_detection", Image, queue_size=10)
# bridge = CvBridge()

rate = rospy.Rate(5)
while True:
    success, img = capture.read()

    img, object_info = get_objects(img)

    # Convert cv image to ROS Image message
    # image_message = bridge.cv2_to_imgmsg(img, encoding="passthrough")
    # topic.publish(image_message)

    # Show images
    # cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('Camera', img)
    # cv2.waitKey(1)

    rate.sleep()
