from trep import MidpointVI
import rospy, tf
from sensor_msgs.msg import JointState

class ROSMidpointVI(MidpointVI):
    def __init__(self, system, timestep, tolerance=1e-10, num_threads=None):
        super(ROSMidpointVI, self).__init__(system, tolerance=tolerance, num_threads=num_threads)

        self.dt = timestep
        self.rate = rospy.Rate(1/self.dt)
        self.br = tf.TransformBroadcaster()

    def step(self, u1=tuple(), k2=tuple(), max_iterations=200, q2_hint=None, lambda1_hint=None):
        steps = super(ROSMidpointVI, self).step(self.t2+self.dt, u1=u1, k2=k2, max_iterations=max_iterations, q2_hint=q2_hint, lambda1_hint=lambda1_hint)

        self.stamp = rospy.Time.now()
        for frame in self.system.frames:
            self.br.sendTransform(frame.g()[:3,3],
                                 tf.transformations.quaternion_from_matrix(frame.g()),
                                 self.stamp,
                                 frame.name,
                                 "world")

        return steps

    def sleep(self):
        return self.rate.sleep()

