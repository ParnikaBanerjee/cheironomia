import numpy as np

ARM_JOINTS = [
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist"
]

def normalize_sequence(sequence):
    """
    sequence: list of dicts {joint_name: (x, y, z)}
    returns: np.ndarray of shape (T, 6, 3)
    """
    normalized = []

    for frame in sequence:
        ls = np.array(frame["left_shoulder"])
        rs = np.array(frame["right_shoulder"])
        origin = (ls + rs) / 2

        frame_vec = []
        for joint in ARM_JOINTS:
            joint_pos = np.array(frame[joint])
            frame_vec.append(joint_pos - origin)

        normalized.append(frame_vec)

    return np.array(normalized)
