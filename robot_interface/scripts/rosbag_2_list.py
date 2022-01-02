import numpy as np
from io import StringIO


def rosbag_2_list(path):
    step = 0
    record_file = open(path, 'r')
    low_state_parameters = ["q", "dq", "ddq", "tau", "footForce", "quaternion", "gyroscope", "accelerometer"]
    data = []
    all_data = record_file.readlines()
    while True:
        step_index = step * (len(low_state_parameters) + 1)
        step_data = all_data[step_index:step_index + (len(low_state_parameters) + 1)]
        if not step_data:
            break
        i = 0
        for para in low_state_parameters:
            to_delete = para + ": ["
            para_string = step_data[i].replace(to_delete, '')
            i += 1
            para_string = para_string.replace(']', '')
            para_string = para_string.replace(',', ' ')
            temp = StringIO(para_string)
            data.append(np.loadtxt(temp))
            # if line is empty
            # end of file is reached
        step += 1
    record_file.close()
    return data


if __name__ == '__main__':
    path = '/home/tal/catkin_ws/src/Steps/robot_interface/sim_result.txt'
    data = rosbag_2_list(path)
    data2save = np.array(data)
    np.savez("new_result", data2save)
