import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv


def load_data(file_path):
    """Load data from a text file."""
    return np.loadtxt(file_path)


def compute_phase_angle(wave1, wave2):
    """
    Parameters:
        wave1 : array-like
        wave2 : array-like
    
    Returns:
        phi_deg : phase difference in degrees
        if wave1 is the reference, phi-deg returns the position of wave2 relative to wave1
        i.e. if phi1 = 0 and phi2 = -90, function returns -90 degrees
    """
    # FFT
    N1 = len(wave1)
    N2 = len(wave2)
    w1_fft = np.fft.fft(wave1)[:N1//2 + 1]
    w2_fft = np.fft.fft(wave2)[:N2//2 + 1]
    
    # Find fundamental as the bin with max magnitude
    W1_index = np.argmax(np.abs(w1_fft))
    W2_index = np.argmax(np.abs(w2_fft))
    
    # Phase of the fundamental
    W1_angle = np.angle(w1_fft[W1_index])
    W2_angle = np.angle(w2_fft[W2_index])
    
    # Phase difference (wave1 relative to wave2)
    phi_rad = W2_angle - W1_angle
    phi_deg = np.degrees(phi_rad)
    
    # if phi_deg > 0, wave2 leads wave1
    # if phi_deg < 0, wave2 lags wave1
    return phi_deg


def integrate(x, y, lower_limit, upper_limit):
    ''''
    compute integral by trapezoidal method
    '''

    integral = [0] # assume initial value is 0

    for i in range(1, len(y)):
        if x[i] < lower_limit or x[i] > upper_limit:
            continue
        dt = (x[i] - x[i-1])
        value = 0.5 * (i - (i-1)) * (y[i] + y[i-1]) * dt
        value += integral[-1]

        integral.append(value)

    return integral


def average(wave, time):
    '''
    calculate average value for complete time range of waveform
    '''
    return integrate(time, wave, time[0], time[-1])[-1]/time[-1]

def rms(wave, time):
    '''
    calculate rms value for complete time range of waveform
    '''
    return np.sqrt(integrate(time, wave**2, time[0], time[-1])[-1]/time[-1])

def summary(**kwargs):
    labels = kwargs.get('labels', None)
    if labels is None:
        raise ValueError("labels argument is required")

    filename = kwargs.get('filename', 'summary.csv')

    def get_arr(key):
        v = kwargs.get(key, None)
        if v is None:
            return []
        return list(v)

    I_rms = get_arr('I_rms')
    V_rms = get_arr('V_rms')
    V_pu = get_arr('V_pu')
    Delta = get_arr('Delta')
    S = get_arr('S')
    P = get_arr('P')
    Q = get_arr('Q')
    Phi = get_arr('Phi')

    def safe_get(arr, idx):
        try:
            return arr[idx]
        except Exception:
            return None

    def fmt(val):
        if val is None:
            return ''
        try:
            # handle numpy types
            return float(val)
        except Exception:
            return val

    fieldnames = ['label', 'I_rms', 'V_rms', 'V_pu', 'S', 'P', 'Delta', 'Phi', 'Q' ]
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, label in enumerate(labels):
            row = {
                'label': label,
                'I_rms': fmt(safe_get(I_rms, i)),
                'V_rms': fmt(safe_get(V_rms, i)),
                'V_pu': fmt(safe_get(V_pu, i)),
                'S': fmt(safe_get(S, i)),
                'P': fmt(safe_get(P, i)),
                'Delta': fmt(safe_get(Delta, i)),
                'Phi': fmt(safe_get(Phi, i)),
                'Q': fmt(safe_get(Q, i)),
            }
            writer.writerow(row)

    return filename


