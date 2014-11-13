from trep import MidpointVI
import rospy, tf
from sensor_msgs.msg import JointState

class ROSMidpointVI(MidpointVI):
    def __init__(self, system, timestep, tolerance=1e-10, num_threads=None):
        super(ROSMidpointVI, self).__init__(system, tolerance=tolerance, num_threads=num_threads)

        self.dt = timestep
        self.rate = rospy.Rate(1/self.dt)
        self.pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        self.joint_msg = JointState()

        if [config.name[-3:] for config in system.configs[:6]] == ['-TX', '-TY', '-TZ', '-RX', '-RY', '-RZ']:
            self.floating = True
            self.br = tf.TransformBroadcaster()
            for config in system.configs[6:]:
                self.joint_msg.name.append(config.name)
        else:
            self.floating = False
            for config in system.configs:
                self.joint_msg.name.append(config.name)

    def step(self, u1=tuple(), k2=tuple(), max_iterations=200, q2_hint=None, lambda1_hint=None):
        steps = super(ROSMidpointVI, self).step(self.t2+self.dt, u1=u1, k2=k2, max_iterations=max_iterations, q2_hint=q2_hint, lambda1_hint=lambda1_hint)

        self.joint_msg.header.stamp = rospy.Time.now()

        if self.floating is True:
            self.joint_msg.position = self.q2[6:]
            self.br.sendTransform(self.system.frames[6].g()[:3,3],
                                 tf.transformations.quaternion_from_matrix(self.system.frames[6].g()),
                                 self.joint_msg.header.stamp,
                                 self.system.frames[6].name,
                                 "world")
        else:
            self.joint_msg.position = self.q2

        self.pub.publish(self.joint_msg)
        return steps

    def sleep(self):
        return self.rate.sleep()

