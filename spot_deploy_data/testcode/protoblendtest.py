def create_proto_blend(pos_command, init_pos, hold_mask):

    N_DOF = 19

    config_kp = [5] * 12 + [10] * 7
    config_kd = [2] * 12 + [7] * 7

    default_kp = [500] * 12 + [100] * 7
    default_kd = [200] * 12 + [70] * 7

    k_q_p = [0] * N_DOF
    k_qd_p = [0] * N_DOF

    for i in range (N_DOF):
        if hold_mask[i]:
            # use def gains
            k_q_p[i] = default_kp[i]
            k_qd_p[i] = default_kd[i]
        else:
            # use conf gains
            k_q_p[i] = config_kp[i]
            k_qd_p[i] = config_kd[i] 

    pos_cmd = [0] * N_DOF

    for joint_ind in range(N_DOF):
        if hold_mask[joint_ind]:
            # if true hold at init pos
            pos_cmd[joint_ind] = init_pos[joint_ind]
        else:
            # if false use policy output
            pos_cmd[joint_ind] = pos_command[joint_ind]

    print("Pos cmd: ", pos_cmd)
    print("Kp: ", k_q_p)
    print("Kd: ", k_qd_p)
    return 0


pos_command = [15] * 12 + [5] * 7

init_pos = [0] * 12 + [0] * 7

hold_arm_only = [
    False, False, False, 
    False, True, False, 
    False, False, False, 
    False, False, False, 
    True, True, True, True, True, True, True]

create_proto_blend(pos_command, init_pos, hold_arm_only)