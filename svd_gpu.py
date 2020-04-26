from datetime import datetime
from random import normalvariate
from tqdm import tqdm
import PySimpleGUI as sg
import numpy as np
import tensorflow as tf
from numpy.linalg import norm

TYPE = float

ERROR_VAL = 1e-10
DIMENSION = 10
DEVICE_NAME = "cpu"
SPACE_SIZE = 2


def tridiag(n):
    # generate triadiagonal matrix
    return np.diag(np.random.rand(n - 1), -1) \
           + np.diag(np.random.rand(n), 0) \
           + np.diag(np.random.rand(n - 1), 1)


def gen_unit_vector(n):
    un_norm = [normalvariate(0, 1) for _ in range(n)]
    # The singular values are sqrt of eigen values
    norm_val = np.sqrt(sum(x * x for x in un_norm))
    return [x / norm_val for x in un_norm]


def svd_1d(matrix, epsilon=ERROR_VAL):
    n, m = matrix.shape
    x = gen_unit_vector(min(n, m))
    curr_v = x

    if n > m:
        B = np.dot(matrix.T, matrix)
    else:
        B = np.dot(matrix, matrix.T)

    i = 0
    while True:
        # The eigenvalues of A.transpose * A  and since A.transponse * A is symmetric we know that the eigenvectors
        # will be orthogonal.
        i += 1
        last_v = curr_v
        curr_v = np.dot(B, last_v)
        curr_v = curr_v / norm(curr_v)

        # do normalization while converge error
        if abs(np.dot(curr_v, last_v)) > 1 - epsilon:
            return curr_v


def svd(matrix, k=None, epsilon=ERROR_VAL):
    matrix = np.array(matrix, dtype=TYPE)
    n, p = matrix.shape
    svd_iterator = []
    if k is None:
        k = min(n, p)

    # tqdm is progress bar
    for i in tqdm(range(k)):
        matrix_for1_d = matrix.copy()

        for singular_value, orthogonal_u, orthogonal_v in svd_iterator[:i]:
            matrix_for1_d -= singular_value * np.outer(orthogonal_u, orthogonal_v)

        if n > p:
            # Now we find the right
            # singular vectors(the columns of V ) by finding an orthonormal
            # set of eigen vectors of A.transpose * A.
            orthogonal_v = svd_1d(matrix_for1_d, epsilon=epsilon)  # next singular vector
            u_unnormalized = np.dot(matrix, orthogonal_v)
            sigma = norm(u_unnormalized)  # next singular value
            orthogonal_u = u_unnormalized / sigma
        else:
            # First we compute the singular values σi by finding the eigenvalues of A * A.transpose
            orthogonal_u = svd_1d(matrix_for1_d, epsilon=epsilon)  # next singular vector
            v_unnormalized = np.dot(matrix.T, orthogonal_u)
            sigma = norm(v_unnormalized)  # next singular value
            orthogonal_v = v_unnormalized / sigma

        svd_iterator.append((sigma, orthogonal_u, orthogonal_v))

    # compute final values
    diagonal_s, orthogonal_u, orthogonal_v = [np.array(x) for x in tqdm(zip(*svd_iterator))]
    return diagonal_s, orthogonal_u.T, orthogonal_v


def start(dimension=DIMENSION, device_name=DEVICE_NAME, space_size=SPACE_SIZE, error=ERROR_VAL, verbose=True):
    if device_name == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"

    a = tridiag(dimension)
    if verbose:
        print("\n" * space_size)
        print("start with matrix A:")
        print(a)
        print("\n" * space_size)

    start_time = datetime.now()
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)):
        with tf.device(device_name):
            s, u_t, v = svd(a, epsilon=error)
            if verbose:
                print("\n" * space_size)
                print("S n × n diagonal matrix:")
                print(s)
                print("\n" * space_size)
                print("U transposed p × n orthogonal matrix:")
                print(u_t)
                print("\n" * space_size)
                print("V n × p orthogonal matrix")
                print(v)

    print("\n" * space_size)
    print("Shape:", (dimension, dimension), "Device:", device_name)
    print("Time taken:", datetime.now() - start_time)
    print("\n" * space_size)


if __name__ == '__main__':
    sg.theme('DarkAmber')
    layout = [[sg.Image(r'fi-xnsuxl-tensorflow.png')],
              [sg.Text('Singular Value Decomposition')],
              [sg.Text('Please enter shape of the input matrix A')],
              [sg.InputText()],
              [sg.Text('Accuracy')],
              [sg.InputText()],
              [sg.Text('Space size')],
              [sg.InputText()],
              [sg.Frame(layout=[
                  [sg.Radio('GPU', "device", size=(10, 2), default=True),
                   sg.Radio('CPU', "device")],
              ], title='Choose CPU or GPU (Graphics card must be present and supported by Tensorflow)',
                  relief=sg.RELIEF_SUNKEN)],
              [sg.Checkbox('Verbose', size=(10, 1), default=True)],
              [sg.Button('Start'), sg.Button('Exit')]]
    window = sg.Window('SVD', layout)

    while True:
        event, values = window.read()
        if event in (None, 'Exit'):  # if user closes window or clicks exit
            break
        else:
            if values[4]:
                device = 'gpu'
            elif values[5]:
                device = 'cpu'
            else:
                device = 'unknown'
            start(dimension=int(values[1]), error=float(values[2]), device_name=device,
                  space_size=int(values[3]), verbose=values[6])

    window.close()
