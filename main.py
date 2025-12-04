import os

from helpers import np, plt, load_data, compute_phase_angle, rms, average, summary
from graph import plot_5bus_network, plot_5bus_PQ

FREQUENCY = 50  # Hz

# define data labels from the power view
data_labels_boxA = ['A_I0', 'A_V0', 'A_I1', 'A_V1', 'A_I2', 'A_V2', 'A_I3', 'A_V3']
data_labels_boxB = ['B_I0', 'B_V0', 'B_I1', 'B_V1', 'B_I2', 'B_V2', 'B_I3', 'B_V3']
data_labels = data_labels_boxA + data_labels_boxB

# define paths
main_folder = os.path.dirname(__file__)
data_folder = os.path.join(main_folder, 'data')
images_folder = os.path.join(main_folder, 'images')

def main(experiment, show=False):
    data_file = os.path.join(data_folder, f'{experiment}.txt')
    image_file = os.path.join(images_folder, f'{experiment}.png')

    # define data shape and create data dictionary
    ex = load_data(data_file)
    data_points = len(ex)
    data = {k: [] for k in data_labels}
    for n, sample in enumerate(ex):
        {data[key].append(sample[i]) for i, (key, value) in enumerate(data.items())}

    # time was 10s but with 10s the signal frequency would be 50/3 hz which is not right
    f_sample = 4000 # Hz
    t_duration = data_points / f_sample  # seconds
    data['time'] = np.linspace(0, t_duration, data_points, endpoint=False)

    # define arrays of interest
    time = data['time']
    i_12 = np.array(data['B_I0'])
    i_23 = np.array(data['B_I1'])
    i_25 = np.array(data['B_I2'])
    i_32 = np.array(data['B_I3'])
    i_35 = np.array(data['A_I0'])
    i_52 = np.array(data['A_I1'])
    i_53 = np.array(data['A_I2'])
    i_43 = np.array(data['A_I3'])
    v_1 = np.array(data['B_V0'])
    v_2 = np.array(data['B_V1'])
    v_3 = np.array(data['B_V3'])
    v_4 = np.array(data['A_V3'])
    v_5 = np.array(data['A_V1'])
    # extra_1 = np.array(data['A_V0'])  # not used
    # extra_2 = np.array(data['A_V2'])  # not used
    # extra_3 = np.array(data['B_V2'])  # not used

    # calculated arrays (not measured)
    i_21 = -(i_23 + i_25)
    i_34 = -(i_32 + i_35)
    i_load = -(i_52 + i_53)

    # # inspect waveforms
    # plt.plot(time, v_1, label='B_V0 (V_1)')
    # plt.plot(time, max(v_1)*np.sin(2*np.pi*FREQUENCY*time), label='Reference sinewave', linestyle='--')
    # # plt.plot(time, v_2, label='B_V1 (V_2)')
    # # plt.plot(time, v_3, label='B_V3 (V_3)')
    # # plt.plot(time, v_4, label='A_V3 (V_4)')
    # # plt.plot(time, v_5, label='A_V1 (V_5)')
    # plt.legend()
    # plt.title(f"Experiment: {experiment}")
    # plt.show()
    # plt.close()

    # # inspect phase shift
    # phi = compute_phase_angle(v_1, i_12)
    # plt.plot(time, v_1, label='Bus voltage')
    # plt.plot(time, i_12, label='Line current')
    # plt.axhline(phi, label='Phase angle', color='black', linestyle='--')
    # plt.legend()
    # plt.title(f"Experiment: {experiment}")
    # plt.show()

    # # figures of merit (average, rms, phase angle)
    # electrical_data = {
    #     'v1': (0, ),
    #     'v2': compute_phase_angle(v_1, v_2),
    #     'v3': compute_phase_angle(v_1, v_3),
    #     'v4': compute_phase_angle(v_1, v_4),
    #     'v5': compute_phase_angle(v_1, v_5),
    #     'i12': compute_phase_angle(v_1, i_12),
    #     'i21': compute_phase_angle(v_1, i_21),
    #     'i23': compute_phase_angle(v_1, i_23),
    #     'i32': compute_phase_angle(v_1, i_32),
    #     'i25': compute_phase_angle(v_1, i_25),
    #     'i52': compute_phase_angle(v_1, i_52),
    #     'i35': compute_phase_angle(v_1, i_35),
    #     'i53': compute_phase_angle(v_1, i_53),
    #     'i34': compute_phase_angle(v_1, i_34),
    #     'i43': compute_phase_angle(v_1, i_43),
    #     'iload': compute_phase_angle(v_1, i_load)
    # }

    phi_v1 = 0
    phi_v2 = compute_phase_angle(v_1, v_2)
    phi_v3 = compute_phase_angle(v_1, v_3)
    phi_v4 = compute_phase_angle(v_1, v_4)
    phi_v5 = compute_phase_angle(v_1, v_5)
    phi_i12 = compute_phase_angle(v_1, i_12)
    phi_i21 = compute_phase_angle(v_1, i_21)
    phi_i23 = compute_phase_angle(v_1, i_23)
    phi_i32 = compute_phase_angle(v_1, i_32)
    phi_i25 = compute_phase_angle(v_1, i_25)
    phi_i52 = compute_phase_angle(v_1, i_52)
    phi_i35 = compute_phase_angle(v_1, i_35)
    phi_i53 = compute_phase_angle(v_1, i_53)
    phi_i34 = compute_phase_angle(v_1, i_34)
    phi_i43 = compute_phase_angle(v_1, i_43)
    phi_iload = compute_phase_angle(v_1, i_load)

    v1_rms = rms(wave=v_1, time=time)
    v2_rms = rms(wave=v_2, time=time)
    v3_rms = rms(wave=v_3, time=time)
    v4_rms = rms(wave=v_4, time=time)
    v5_rms = rms(wave=v_5, time=time)
    i12_rms = rms(wave=i_12, time=time)
    i21_rms = rms(wave=i_21, time=time)
    i23_rms = rms(wave=i_23, time=time)
    i32_rms = rms(wave=i_32, time=time)
    i25_rms = rms(wave=i_25, time=time)
    i52_rms = rms(wave=i_52, time=time)
    i35_rms = rms(wave=i_35, time=time)
    i53_rms = rms(wave=i_53, time=time)
    i34_rms = rms(wave=i_34, time=time)
    i43_rms = rms(wave=i_43, time=time)
    iload_rms = rms(wave=i_load, time=time)

    # # inspect RMS voltage
    # plt.plot(time, v_1, label='Bus voltage')
    # plt.axhline(v1_rms, label='RMS voltage', color='black', linestyle='--')
    # plt.legend()
    # plt.show()

    # Prepare data for plotting the network
    V = {
        1: (v1_rms, phi_v1),
        2: (v2_rms, phi_v2),
        3: (v3_rms, phi_v3),
        4: (v4_rms, phi_v4),
        5: (v5_rms, phi_v5)
    }

    I = {
        (1,2): (i12_rms, phi_i12),
        (2,1): (i21_rms, phi_i21),
        (2,3): (i23_rms, phi_i23),
        (3,2): (i32_rms, phi_i32),
        (2,5): (i25_rms, phi_i25),
        (5,2): (i52_rms, phi_i52),
        (3,5): (i35_rms, phi_i35),
        (5,3): (i53_rms, phi_i53),
        (3,4): (i34_rms, phi_i34),
        (4,3): (i43_rms, phi_i43),
        (5,6): (iload_rms, phi_iload),
    }   
    graph_labels = {
        (5,6): 'I_load'
    }
    image_file = os.path.join(images_folder, f'{experiment}_currents.png')
    fig = plot_5bus_network(V, I, graph_labels=graph_labels)
    fig.savefig(image_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # compute power flow

    # calculate instantaneous power
    p_12 = v_1 * i_12
    p_21 = v_2 * i_21
    p_23 = v_2 * i_23
    p_32 = v_3 * i_32
    p_25 = v_2 * i_25
    p_52 = v_5 * i_52
    p_35 = v_3 * i_35
    p_53 = v_5 * i_53
    p_34 = v_3 * i_34
    p_43 = v_4 * i_43
    p_load = v_5 * i_load

    # calculate average power
    P_12 = 3 * average(p_12, time)
    P_21 = 3 * average(p_21, time)
    P_23 = 3 * average(p_23, time)
    P_32 = 3 * average(p_32, time)
    P_25 = 3 * average(p_25, time)
    P_52 = 3 * average(p_52, time)
    P_35 = 3 * average(p_35, time)
    P_53 = 3 * average(p_53, time)
    P_34 = 3 * average(p_34, time)
    P_43 = 3 * average(p_43, time)
    P_load = 3 * average(p_load, time)

    P = {
        (1,2): P_12,
        (2,1): P_21,
        (2,3): P_23,
        (3,2): P_32,
        (2,5): P_25,
        (5,2): P_52,
        (3,5): P_35,
        (5,3): P_53,
        (3,4): P_34,
        (4,3): P_43,
        (5,6): P_load,
    }

    graph_labels = {
        (5,6): 'P_load'
    }
    image_file = os.path.join(images_folder, f'{experiment}_Ppower.png')
    fig = plot_5bus_network(V, P, graph_labels=graph_labels, type='Ppower')
    fig.savefig(image_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # calculate aparent power
    S_12 = 3 * v1_rms * i12_rms
    S_21 = 3 * v2_rms * i21_rms
    S_23 = 3 * v2_rms * i23_rms
    S_32 = 3 * v3_rms * i32_rms
    S_25 = 3 * v2_rms * i25_rms
    S_52 = 3 * v5_rms * i52_rms
    S_35 = 3 * v3_rms * i35_rms
    S_53 = 3 * v5_rms * i53_rms
    S_34 = 3 * v3_rms * i34_rms
    S_43 = 3 * v4_rms * i43_rms
    S_load = 3 * v5_rms * iload_rms

    S = {
        (1,2): S_12,
        (2,1): S_21,
        (2,3): S_23,
        (3,2): S_32,
        (2,5): S_25,
        (5,2): S_52,
        (3,5): S_35,
        (5,3): S_53,
        (3,4): S_34,
        (4,3): S_43,
        (5,6): S_load,
    }

    graph_labels = {
        (5,6): 'S_load'
    }
    image_file = os.path.join(images_folder, f'{experiment}_Spower.png')
    fig = plot_5bus_network(V, S, graph_labels=graph_labels, type='Spower')
    fig.savefig(image_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # calculate reactive power
    Q_12 = P_12 * np.tan(np.radians(phi_v1) - np.radians(phi_i12))
    Q_21 = P_21 * np.tan(np.radians(phi_v2) - np.radians(phi_i21))
    Q_23 = P_23 * np.tan(np.radians(phi_v2) - np.radians(phi_i23))
    Q_32 = P_32 * np.tan(np.radians(phi_v3) - np.radians(phi_i32))
    Q_25 = P_25 * np.tan(np.radians(phi_v2) - np.radians(phi_i25))
    Q_52 = P_52 * np.tan(np.radians(phi_v5) - np.radians(phi_i52))
    Q_35 = P_35 * np.tan(np.radians(phi_v3) - np.radians(phi_i35))
    Q_53 = P_53 * np.tan(np.radians(phi_v5) - np.radians(phi_i53))
    Q_34 = P_34 * np.tan(np.radians(phi_v3) - np.radians(phi_i34))
    Q_43 = P_43 * np.tan(np.radians(phi_v4) - np.radians(phi_i43))
    Q_load = P_load * np.tan(np.radians(phi_v5) - np.radians(phi_iload))

    Q = {
        (1,2): Q_12,
        (2,1): Q_21,
        (2,3): Q_23,
        (3,2): Q_32,
        (2,5): Q_25,
        (5,2): Q_52,
        (3,5): Q_35,
        (5,3): Q_53,
        (3,4): Q_34,
        (4,3): Q_43,
        (5,6): Q_load,
    }

    graph_labels = {
        (5,6): 'Q_load'
    }
    image_file = os.path.join(images_folder, f'{experiment}_Qpower.png')
    fig = plot_5bus_network(V, Q, graph_labels=graph_labels, type='Qpower')
    fig.savefig(image_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # plot P and Q together
    graph_labels = {
        (2,5): {'rotation': -60, 'offset': 0.175},
        (3,5): {'rotation': +60, 'offset': 0.175}
    }
    image_file = os.path.join(images_folder, f'{experiment}_PQpower.png')
    fig = plot_5bus_PQ(V, P, Q, graph_labels=graph_labels)
    fig.savefig(image_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # save summary CSV, format data to match word table
    v_base = 230/np.sqrt(3)  # Volts
    labels = ['1', '2', '3', '4', '5', '6', '7', '8']
    I_rms = [i12_rms, i23_rms, i25_rms, i32_rms, i35_rms, i52_rms, i53_rms, i43_rms]
    V_rms = [v1_rms, v2_rms, v2_rms, v3_rms, v3_rms, v5_rms, v5_rms, v4_rms]
    V_pu = [v/v_base for v in V_rms]
    Delta = [phi_v1, phi_v2, phi_v2, phi_v3, phi_v3, phi_v5, phi_v5, phi_v4]
    S = [S_12, S_23, S_25, S_32, S_35, S_52, S_53, S_43]
    P = [P_12, P_23, P_25, P_32, P_35, P_52, P_53, P_43]
    Q = [Q_12, Q_23, Q_25, Q_32, Q_35, Q_52, Q_53, Q_43]
    Phi = [phi_i12, phi_i23, phi_i25, phi_i32, phi_i35, phi_i52, phi_i53, phi_i43]

    data_summary = {
        'labels': labels,
        'I_rms': I_rms,
        'V_rms': V_rms,
        'V_pu': V_pu,
        'Delta': Delta,
        'S': S,
        'P': P,
        'Q': Q,
        'Phi': Phi,
        'filename': os.path.join(main_folder, 'summaries', f'{experiment}_summary.csv')
    }
    summary(**data_summary)


if __name__ == "__main__":

    for ex in ['ex1', 'ex2', 'ex3', 'ex4', 'ex5', 'ex6', 'ex7', 'ex8']:
        experiment = ex
        print(f"Running analysis for experiment: {experiment}")
        main(experiment=experiment, show=False)