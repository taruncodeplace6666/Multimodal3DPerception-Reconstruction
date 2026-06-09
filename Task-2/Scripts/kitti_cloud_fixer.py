
"""
kitti_cloud_fixer.py  v4
Handles KITTI bags where intensity field is named 'i' (not 'intensity').
Adds ring + time fields required by FAST-LIVO2.

Subscribes:  /kitti/velo/pointcloud        (x, y, z, i)
Publishes:   /kitti/velo/pointcloud_fixed  (x, y, z, intensity, ring, time)
"""
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import math
import struct

class KittiCloudFixer:

    # HDL-64E vertical angles (degrees), laser 0 = bottom, laser 63 = top
    VERT_ANGLES = np.array([
        -24.9, -24.8, -24.0, -23.5, -22.9, -22.4, -21.9, -21.3,
        -20.8, -20.2, -19.7, -19.1, -18.6, -18.0, -17.5, -16.9,
        -16.4, -15.8, -15.3, -14.7, -14.2, -13.6, -13.1, -12.5,
        -12.0, -11.4, -10.9, -10.3,  -9.8,  -9.2,  -8.7,  -8.1,
         -7.6,  -7.0,  -6.5,  -5.9,  -5.4,  -4.8,  -4.3,  -3.7,
         -3.2,  -2.6,  -2.1,  -1.5,  -1.0,  -0.4,   0.1,   0.7,
          1.2,   1.8,   2.3,   2.9,   3.4,   4.0,   4.5,   5.1,
          5.6,   6.2,   6.7,   7.3,   7.8,   8.4,   8.9,   9.5
    ], dtype=np.float32)

    SCAN_PERIOD = 0.1   # HDL-64E at 10 Hz

    # Output fields expected by FAST-LIVO2
    OUT_FIELDS = [
        PointField('x',         0,  PointField.FLOAT32, 1),
        PointField('y',         4,  PointField.FLOAT32, 1),
        PointField('z',         8,  PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
        PointField('ring',      16, PointField.UINT16,  1),
        PointField('time',      18, PointField.FLOAT32, 1),
    ]
    POINT_STEP = 22  # 4+4+4+4+2+4 bytes

    def __init__(self):
        self._fields_checked = False
        self._has_intensity  = False
        self._intensity_name = 'i'   # KITTI uses 'i', some bags use 'intensity'

        self.pub = rospy.Publisher(
            '/kitti/velo/pointcloud_fixed',
            PointCloud2, queue_size=10)
        self.sub = rospy.Subscriber(
            '/kitti/velo/pointcloud',
            PointCloud2, self.callback, queue_size=10)
        rospy.loginfo("[kitti_cloud_fixer] Ready: "
                      "/kitti/velo/pointcloud -> /kitti/velo/pointcloud_fixed")

    def _check_fields(self, msg):
        """Auto-detect intensity field name on first message."""
        names = [f.name for f in msg.fields]
        if 'i' in names:
            self._has_intensity  = True
            self._intensity_name = 'i'
        elif 'intensity' in names:
            self._has_intensity  = True
            self._intensity_name = 'intensity'
        else:
            self._has_intensity  = False
            self._intensity_name = None
        rospy.loginfo(
            f"[kitti_cloud_fixer] Detected fields: {names} | "
            f"intensity field: '{self._intensity_name}' "
            f"({'found' if self._has_intensity else 'NOT FOUND - using zeros'})")
        self._fields_checked = True

    def callback(self, msg):
        if not self._fields_checked:
            self._check_fields(msg)

        # Read points — handle both 'i' and 'intensity' field names
        if self._has_intensity:
            raw = list(pc2.read_points(
                msg,
                field_names=('x', 'y', 'z', self._intensity_name),
                skip_nans=True))
            pts = np.array(raw, dtype=np.float32)
            if pts.size == 0:
                return
            x, y, z  = pts[:, 0], pts[:, 1], pts[:, 2]
            intensity = pts[:, 3]
        else:
            raw = list(pc2.read_points(
                msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True))
            pts = np.array(raw, dtype=np.float32)
            if pts.size == 0:
                return
            x, y, z  = pts[:, 0], pts[:, 1], pts[:, 2]
            intensity = np.zeros(len(x), dtype=np.float32)

        # Ring: nearest HDL-64E laser by vertical angle 
        r_xy     = np.hypot(x, y)
        vert_deg = np.degrees(np.arctan2(z, r_xy))
        diff     = np.abs(vert_deg[:, None] - self.VERT_ANGLES[None, :])
        ring     = diff.argmin(axis=1).astype(np.uint16)

        # Time: relative timestamp from azimuth angle 
        azimuth  = np.arctan2(y, x)              # [-pi, pi]
        azimuth[azimuth < 0] += 2 * math.pi      # [0, 2pi]
        rel_time = (self.SCAN_PERIOD * azimuth
                    / (2 * math.pi)).astype(np.float32)

        # Pack into binary buffer 
        N   = len(x)
        buf = bytearray(N * self.POINT_STEP)
        for i in range(N):
            struct.pack_into(
                '<ffffHf', buf, i * self.POINT_STEP,
                float(x[i]), float(y[i]), float(z[i]),
                float(intensity[i]),
                int(ring[i]),
                float(rel_time[i]))

        # Publish 
        out              = PointCloud2()
        out.header       = msg.header
        out.height       = 1
        out.width        = N
        out.fields       = self.OUT_FIELDS
        out.is_bigendian = False
        out.point_step   = self.POINT_STEP
        out.row_step     = self.POINT_STEP * N
        out.data         = bytes(buf)
        out.is_dense     = True
        self.pub.publish(out)


if __name__ == '__main__':
    rospy.init_node('kitti_cloud_fixer')
    fixer = KittiCloudFixer()
    rospy.spin()
