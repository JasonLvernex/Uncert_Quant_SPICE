'''
Ancillary Functions for SPICE_Uncertainty 
'''

# import libs
import matplotlib.pyplot as plt
# from PhantomSimulation import PhantomSimulation,createSpatialCurve
# from SamplingKT import SamplingKT
import numpy as np
from fsl_mrs.utils import synthetic as syn
from fsl_mrs.core import basis
from fsl_mrs.utils.plotting import FID2Spec
from fsl_mrs.utils.misc import FIDToSpec
# from PlotFunctions import PlotFunctions
# from SPICEfuncs import SPICEFuncs
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from scipy.sparse import lil_matrix, coo_matrix
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import random
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import minimize
from functools import partial
import os
import json

N_SEQ_POINTS = 512#1024#2048 #512
N_BANDWIDTH = 2800#3000 #1000
N_VOXEL = 64
PEAK_0_ROUGH_IDX = 381#1560
PEAK_1_ROUGH_IDX = 336#1321

'''
Phantom simulation related
'''
def createSpatialCurve(K_POINTS):
        '''
        Describe 1D phantom amplitudes
        Plot and construct a 1D Phantom
        '''
        phantom_res = K_POINTS  # there are 32 points along the x-axis
        phantom_size = 1  # the range of the x-axis used to distribute the peaks, see follwoing line for further explainations
        x_step = phantom_size / phantom_res # our resolution/datapoints sampled are 32 point, so each "step" we move our detector will move along x_step distance along x-axis
        x_hr = np.arange(x_step, phantom_size+x_step/2, x_step) #Creates an array of spatial points from x_step to phantom_size with a step size of x_step
        x_hr_pos = x_hr - phantom_size/2 # move x_hr to actual positions in x-axis

        ################################# TESTING FOR BOUNDARY PRIOR #########################
        peak1_amp = 1-np.exp(-0.5/x_hr)  
        peak2_amp = np.flip(peak1_amp.copy())
        peak1_amp[(x_hr_pos<-0.4)|(x_hr_pos>0.4)] = 0
        peak2_amp[(x_hr_pos<-0.4)|(x_hr_pos>0.4)] = 0
        peak1_amp[(x_hr_pos<0.3)&(x_hr_pos>0.18)] = 0
        peak2_amp[(x_hr_pos<0.3)&(x_hr_pos>0.18)] = 0
        ################################# TESTING FOR BOUNDARY PRIOR #########################  

         # acquire boundary position information
        boundary_mask = (x_hr_pos < -0.4) | (x_hr_pos > 0.4) | ((x_hr_pos<0.3)&(x_hr_pos>0.18))
        print("boundary position (x_hr_pos):", x_hr_pos[boundary_mask])
        print("boundary position peak1_amp value:", peak1_amp[boundary_mask])
        boundary_indices = np.where(boundary_mask)[0]
        print("Boundary mask TRUE idx:", boundary_indices)
        lm_boundary_mask_idx = np.where(x_hr_pos < -0.4)[0]
        print(f'boundary left most next is peak 0: {peak1_amp[lm_boundary_mask_idx[-1]+1]}, peak 2: {peak2_amp[lm_boundary_mask_idx[-1]+1]}')


        # Combine all peaks
        combined_amp = peak1_amp + peak2_amp 



        plt.figure()  # Create a new figure
        plt.plot(x_hr_pos, peak1_amp, 's--', label='Peak 1')
        plt.plot(x_hr_pos, peak2_amp, 'o--', label='Peak 2')
        plt.plot(x_hr_pos, combined_amp, 'k-', label='Combined Amplitude')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.title('1D Phantom Amplitudes')
        plt.show()
        return phantom_res, phantom_size, x_hr, x_hr_pos, peak1_amp, peak2_amp, combined_amp

def createSpatialCurve_1D_Brain(K_POINTS:int,
                                Brain_img_smoothed:np.ndarray) -> tuple:
        '''
        Describe 1D phantom amplitudes
        Plot and construct a 1D Phantom
        '''
        phantom_res = K_POINTS  # there are 32 points along the x-axis
        phantom_size = 1  # the range of the x-axis used to distribute the peaks, see follwoing line for further explainations
        x_step = phantom_size / phantom_res # our resolution/datapoints sampled are 32 point, so each "step" we move our detector will move along x_step distance along x-axis
        x_hr = np.arange(x_step, phantom_size+x_step/2, x_step) #Creates an array of spatial points from x_step to phantom_size with a step size of x_step
        x_hr_pos = x_hr - phantom_size/2 # move x_hr to actual positions in x-axis

        ################################# TESTING FOR BOUNDARY PRIOR #########################
        ''' 
        Assumption:
        1. Brain_img_smoothed:
            1. value below or equal 1 should have no signal (either fat tissue/blood-brain barrier)
            2. value between 1 and 3 is grey matter
            3. value above 3 is white matter
        2. Chemical distribution:
            1. NAA is rich(value set to 1) in GM while Cho is rich in WM, when low, value set to 0.5
            2. When at CSF(middle of the brain slice and value is 2), both NAA and Cho are low set to 0.2
        '''
        peak1_amp = np.zeros_like(Brain_img_smoothed)
        peak2_amp = np.zeros_like(Brain_img_smoothed)

        for i, val in enumerate(Brain_img_smoothed):
            if val <= 1:
                peak1_amp[i] = 0  # No signal for fat tissue/blood-brain barrier
                peak2_amp[i] = 0
            elif val <= 3:
                peak1_amp[i] = 1.0  # NAA high in GM
                peak2_amp[i] = 0.5  # Cho low in GM
            elif val > 3:
                peak1_amp[i] = 0.5  # NAA low in WM
                peak2_amp[i] = 1.0  # Cho high in WM
            # Adjust CSF region (value ~2)
            if val == 2 and (i > (K_POINTS/2-8) and i < (K_POINTS/2+8)):
                peak1_amp[i] = 0.2  # NAA low in CSF
                peak2_amp[i] = 0.2  # Cho low in CSF
        ################################# TESTING FOR BOUNDARY PRIOR #########################  


        plt.figure()  # Create a new figure
           
        plt.plot(peak1_amp, '-', label='Peak 1')
        plt.plot(peak2_amp, '-', label='Peak 2')
        # plt.plot(x_hr_pos, combined_amp, 'k-', label='Combined Amplitude')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.title('1D Phantom Amplitudes')
        # Add the legend here
        plt.legend() 
        plt.show()
        return phantom_res, phantom_size, x_hr, x_hr_pos, peak1_amp, peak2_amp

def make_synthetic_fid(chemical_shift: float, line_broadening: float):
    """
    Create a synthetic, noiseless FID of a single peak.

    :param chemical_shift: Position of peak
    :type chemical_shift: float
    :param line_broadening: Amount of line broadening (Hz)
    :type line_broadening: float
    :return: FID and headers
    :rtype: tuple
    """
    fid, headers = syn.syntheticFID(
        noisecovariance=[[0]],
        points=N_SEQ_POINTS,#512
        bandwidth=N_BANDWIDTH,#1000
        chemicalshift=[chemical_shift],
        amplitude=[1],
        linewidth=[line_broadening],
    )
    headers['fwhm'] = line_broadening

    time_axis = headers['taxis']#peak1_hdr['taxis']

    ppm_axis = headers['ppmaxis']
    return fid[0], headers, time_axis, ppm_axis

def list_all_keys(d, prefix=''):
    if isinstance(d, dict):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            print(full_key)
            list_all_keys(v, full_key)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            full_key = f"{prefix}[{i}]"
            list_all_keys(item, full_key)

def read_basis_json(Basis_dir: str, filename: str) -> np.ndarray:
    """
    Reads a JSON file and converts the data to a NumPy array. Also prints all nested keys.

    Args:
        Basis_dir (str): Path to the folder containing the basis file
        filename (str): Name of the basis JSON file (without .json)

    Returns:
        np.ndarray: Top-level data as a NumPy array (if applicable)

    #Example usage:

    basis = read_basis_json('basis_data.json')

    # Access individual elements
    freqs = np.array(basis["frequencies"])
    linewidths = np.array(basis["linewidths"])
    amps = np.array(basis["amplitudes"])
    names = basis["metabolite_names"]

    """
    with open(Basis_dir + filename + '.json', 'r') as f:
        data = json.load(f)

    # print("Top-level keys:", data.keys())
    # print("\n--- All nested keys in the JSON ---")
    # list_all_keys(data)


    return data #array


def gen_voxel_signal(
          lb: list[float],
          concs: list[float],
          shifts: list[float],
          bm:list[np.ndarray],
          taxis:np.ndarray) -> np.ndarray:
    """Generate the FID signal for a voxel as a weighted sum of basis sets.

    :param lb: Line broadening in units of hertz
    :type lb: list[float]
    :param concs: Concentrations of each peak
    :type concs: list[float]
    :param shifts: Chemical shift of each peak
    :type shifts: list[float]
    :return: Summed FID of all peaks
    :rtype: np.ndarray
    """
    try:
        assert len({len(x) for x in (lb, concs, shifts)}) == 1
    except AssertionError as exc:
            print('The length of lb, concs, and shifts must match!')
            print(f'Currently they are {len(lb)}, {len(concs)}, {len(shifts)}')
            raise exc

    fids = []
    # taxis = taxis.reshape(512, 1)
    
    for shift, linews, concentr,basis_fid in zip(shifts, lb, concs, bm):
        # basis_fid = basis_fid.reshape(512, 1)
        broadening = np.exp(-linews * np.pi * taxis)

        fid = (basis_fid * broadening).T#make_synthetic_fid(shift, linews)
        fids.append(concentr * fid )
    return np.sum(fids, axis=0)


def gen_voxel_signal_discarded(
          lb: list[float],
          concs: list[float],
          shifts: list[float]) -> np.ndarray:
    """Generate the FID signal for a voxel as a weighted sum of basis sets.

    :param lb: Line broadening in units of hertz
    :type lb: list[float]
    :param concs: Concentrations of each peak
    :type concs: list[float]
    :param shifts: Chemical shift of each peak
    :type shifts: list[float]
    :return: Summed FID of all peaks
    :rtype: np.ndarray
    """
    try:
        assert len({len(x) for x in (lb, concs, shifts)}) == 1
    except AssertionError as exc:
            print('The length of lb, concs, and shifts must match!')
            print(f'Currently they are {len(lb)}, {len(concs)}, {len(shifts)}')
            raise exc

    fids = []
    for shift, linews, concentr in zip(shifts, lb, concs):
        fid, _, _, _ = make_synthetic_fid(shift, linews)
        fids.append(fid * concentr)
    return np.sum(fids, axis=0)

def make_1d_spectral_phantom(amp1:list, 
                             amp2:list,
                             peak_cs:list, 
                             lw:list,
                             bm:list,
                             taxis:np.ndarray) ->np.ndarray:
        """
        Construct the full 1D spatial phantom.

        :param linebroadening: Line broadening values
        :type linebroadening: list
        :return: Stack of FIDs
        :rtype: np.ndarray
        """
        fids = []
        for x1, x2 in zip(amp1, amp2):
            fids.append(gen_voxel_signal(lw, (x1, x2), peak_cs,bm,taxis))
        return np.stack(fids)


def gen_gt_ktspace(gt_ITspace:np.ndarray,
                   N_k: int,
                   Fmat:np.ndarray, 
                   time_axis: list,
                   plot: bool = False) -> tuple:
        """
        Generate the kt-space in silico phantom.

        :param gt_ITspace: ground truth image-time(FID)
        :type gt_ITspace: np.ndarray
        :param N_k: Number of k-space encodings in spatial dimension
        :type N_k: int
        :param plot: Flag to plot, defaults to False
        :type plot: bool, optional
        :return: kspace, the noisy kt-space data (N_k x #time)
        :rtype: np.ndarray
        :return: the noiseless ground truth kt-space data (K_x x #time)
        :rtype: np.ndarray
        """

        # Generate noiseless ground truth (space x time)
        gt = gt_ITspace

        k_x = np.arange(N_k)

        # Generate the kt-space representation using the direct multiplication with the encoding matrix
        kspace = Fmat @ gt[:, :]


        if plot:
            plt.pcolor(time_axis, k_x, np.abs(kspace[:, :]))
            plt.title('kt-space, truncated view in t')
            plt.xlabel('Time (#)')
            plt.xlim([0, time_axis[50]])
            plt.ylabel('$k_x$')
            plt.show()

        return kspace

def add_noise2kt(
    kspace: np.ndarray,
    rng: np.random.Generator,
    noise_SD: float) -> np.ndarray:
    """Adds complex gaussian noise to kt space model

    :param kspace: Noiseless kt-space
    :type kspace: np.ndarray
    :param rng: rng object
    :type rng: np.random.Generator
    :param noise_SD: Noise standard deviation
    :type noise_SD: float
    :return: Noisy kt-space data
    :rtype: np.ndarray
    """
    return kspace +(
            (noise_SD / np.sqrt(2)) * rng.standard_normal(size=kspace.shape)
            + (noise_SD / np.sqrt(2)) * 1j * rng.standard_normal(size=kspace.shape)
            )

def calc_F(N_k: int, plot_hdl: bool) -> np.ndarray:
    """Calculate the k-space encoding matrix F

    :param N_k: Number of k space points
    :type N_k: int
    :param plot_hdl: Plot F matrix for debugging
    :type plot_hdl: bool
    :return: F Matrix (N voxels x N k-space points)
    :rtype: np.ndarray
    """
    # Generate encoding matrix, F
    k_x = np.linspace(-0.5, 0.5, N_k)
    # Shift k_x to prevent phase roll
    k_x += (k_x[0] - k_x[1]) / 2

    N_x = N_k  # Match spatial resolution to k-space samples

    w = np.exp(-1j*2.*np.pi/N_x)

    # j, k = np.arange(N_x), np.arange(N_x)
    j, k = np.arange(-N_x/2,N_x/2), np.arange(-N_x/2,N_x/2)

    Fmat = np.power(np.power(w,j)[:,None],k)/np.sqrt(N_x)

    # Plot encoding matrix
    if plot_hdl:
            plt.imshow(np.angle(Fmat))
            plt.title('Encoding matrix F')
            plt.show()
    
    return Fmat

def Undersampe_zeroout(noisy_kt_space_US:np.ndarray, 
                       F_US:np.ndarray, 
                       UNDER_SAMPLE_NUM:int, 
                       K_POINTS: int,
                       time_axis:np.ndarray, 
                       plot: bool = False, 
                       handler: bool = False) -> tuple:
        if handler:
            skip_lns =  random.sample(range(0,K_POINTS-1), UNDER_SAMPLE_NUM) 
            
             # 🔑 Make safe copies to avoid modifying original inputs
            noisy_kt_space_US = noisy_kt_space_US.copy()
            F_US = F_US.copy()

            for _, i in enumerate(skip_lns):
                noisy_kt_space_US[i][:] = 0
                F_US[i][:] = 0
            # Plot encoding matrix
            # Generate encoding matrix, F
            k_x = np.linspace(-0.5, 0.5, K_POINTS)
            # This next line needed to stop a phase roll across space
            # A phase roll is an undesirable effect that can occur if the k-space sampling is not centered correctly. 
            # By shifting k_x slightly, the code ensures that the spatial encoding is aligned properly.
            # When the k-space sampling is not centered around zero, each spatial frequency component accumulates a phase offset. 
            # This offset manifests as a linear phase shift in the reconstructed spatial signal, which appears as a rolling or drifting effect.
            k_x += (k_x[0]-k_x[1])/2

            if plot:
                plt.imshow(np.angle(F_US))
                plt.title('Encoding matrix F Undersampled')
                plt.show()

                print(skip_lns) # plot which positions in k-space are selected
            if plot:
                plt.pcolor(time_axis, k_x, np.abs(noisy_kt_space_US[:, :]))
                plt.title('kt-space, truncated view in t Undersampled')
                plt.xlabel('Time (#)')
                plt.xlim([0, time_axis[50]])
                plt.ylabel('$k_x$')
                plt.show()
            return noisy_kt_space_US, F_US
        else:
            return noisy_kt_space_US, F_US


'''
Plotting and evaluation related
'''

def calc_rmse(rcon,ground_truth):
        """
        Calculate and print RMSE.

        :param rcon: Reconstructed data
        :type rcon: np.ndarray
        """
        rmse = np.linalg.norm(rcon - ground_truth) / np.linalg.norm(ground_truth)
        print(f'NRMSE: {rmse}')
        return rmse

def plot_spec_examples(ax, rcon,ground_truth, ppm_axis, limits=[-0.075, None], plot_gt=True):
        """
        Plot reconstructed spectra examples.

        :param ax: Matplotlib axis for plotting
        :type ax: matplotlib.axes.Axes
        :param rcon: Reconstructed spectra
        :type rcon: np.ndarray
        :param limits: Y-axis limits
        :type limits: list
        :param plot_gt: Whether to plot ground truth
        :type plot_gt: bool
        """
        # Define custom colors for the five curves
        colors = ['#00A5E3', '#8DD7BF', '#FF96C5', '#FF5768', '#FFBF65']
        
        for idx, voxel in enumerate([2, 8, 16, 24, 30]):
            # Use the corresponding color from the list
            ax.plot(ppm_axis, rcon[voxel, :].real + idx * 1.4, color=colors[idx])
            if plot_gt:
                # Plot ground truth with a consistent style (e.g., dotted black line)
                ax.plot(ppm_axis, FID2Spec(ground_truth[voxel, :]).real + idx * 1.4, 'k:')
        ax.set_ylim(limits)
        ax.set_xlim([ppm_axis[-1], ppm_axis[0]])

def plot_feq_bin( ax, recon,x_hr_pos,peak1_amp,peak2_amp, plot_gt=True):
        """
        Plot frequency bin estimates.

        :param ax: Matplotlib axis for plotting
        :type ax: matplotlib.axes.Axes
        :param recon: Reconstructed data
        :type recon: np.ndarray
        :param plot_gt: Whether to plot ground truth
        :type plot_gt: bool
        """
        peak2est = np.abs(recon)[:, PEAK_0_ROUGH_IDX]#64
        peak1est = np.abs(recon)[:, PEAK_1_ROUGH_IDX]#162
        peak2est /= 1.4  # Fudge factor
        peak1est /= 1.4  # Fudge factor
        xaxis = x_hr_pos
        ax.plot(xaxis, peak1est,color='blue',)
        ax.plot(xaxis, peak2est,color='orange',)
        combined_amp_est = peak1est + peak2est
        ax.plot(x_hr_pos, combined_amp_est, color='green', linestyle=':', label='Combined Amplitude')
        if plot_gt:
            ax.plot(x_hr_pos, peak1_amp, 'k:')
            ax.plot(x_hr_pos, peak2_amp, 'k:')
            combined_amp = peak1_amp + peak2_amp
            ax.plot(x_hr_pos, combined_amp, color='black', linestyle='-', label='Combined Amplitude')
        ax.set_ylim([-0.1, None]) #ax.set_ylim([-0.1, 2])

def plot_and_rmse(rcon,ground_truth, ppm_axis, x_hr_pos, peak1_amp,peak2_amp):
        """
        Summarize results of a particular reconstruction.

        :param rcon: Reconstructed data
        :type rcon: np.ndarray
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        calc_rmse(rcon,ground_truth)
        rconspec = FIDToSpec(rcon, axis=1)
        plot_spec_examples(ax1, rconspec, ground_truth, ppm_axis)
        plot_feq_bin(ax2, rconspec,x_hr_pos, peak1_amp,peak2_amp)
        plt.show()


'''
SPICE related
'''

def save_training_data_as_csv(training_data:np.ndarray, 
                              save_dir:str, 
                              filename:str, 
                              savecondition:bool):
    """
    Save training data with complex numbers to a CSV file.

    :param training_data: The training data array to save
    :type training_data: np.ndarray
    :param save_dir: save location
    :type save_dir: str
    :param filename: Name of the CSV file to save the data
    :type filename: str
    :param savecondition: Condition to decide whether to save the file
    :type savecondition: bool
    """
    if savecondition:
        # Save the real and imaginary parts separately
        real_part = training_data.real
        imag_part = training_data.imag

        # Combine real and imaginary parts into a 2D array
        combined_data = np.hstack((real_part, imag_part))

        # Save the combined data to a CSV
        filepath = os.path.join(save_dir, filename + '.csv')
        np.savetxt(filepath, combined_data, delimiter=',')
        print(f"Training data saved to {filepath}")

def read_training_data_from_csv(save_dir:str, 
                                filename:str)->np.ndarray:
        """
        Read training data with complex numbers from a CSV file.

        :param save_dir: save location
        :type save_dir: str
        :param filename: Name of the CSV file to save the data
        :type filename: str
        :return: The training data array
        :rtype: np.ndarray
        """
        filepath = os.path.join(save_dir, filename + '.csv')

        # Load the combined real and imaginary data
        combined_data = np.loadtxt(filepath, delimiter=',')

        # Split the real and imaginary parts
        num_columns = combined_data.shape[1]
        real_part = combined_data[:, :num_columns // 2]
        imag_part = combined_data[:, num_columns // 2:]

        # Reconstruct the complex array
        training_data = real_part + 1j * imag_part

        print(f"Training data loaded from {filepath}")
        return training_data

def CreateWaterImage(K_POINTS:int ,
                     plt_hander:bool = True)->np.ndarray:
    """Simulate Water constraint images

    Args:
        K_POINTS (int): number of K-space points
        plt_hander (bool, optional): plot handler. Defaults to True.

    Returns:
        np.ndarray: simulated high-SNR water proton image (or if we have more anatomical prior images)
    """
    # operations in image domain
    water_img = [2.0 for i in range(K_POINTS)] # i just set the everage value of 2 here, should adjust later
    # set the boundaries
    water_rou_1D = water_img

    # Convert to NumPy array
    water_rou_1D = np.array(water_img)

    # Define x-axis
    phantom_size = 1
    phantom_res = K_POINTS
    x_step = phantom_size / phantom_res
    W_e_xaxis = np.linspace(-0.5 + x_step, 0.5, K_POINTS)

   # Modify `water_rou_1D` for the specified ranges
    water_rou_1D[(W_e_xaxis >= -0.5) & (W_e_xaxis <= -0.4)] = 0
    water_rou_1D[(W_e_xaxis >= 0.4) & (W_e_xaxis <= 0.5)] = 0
    water_rou_1D[(W_e_xaxis >= 0.18) & (W_e_xaxis <= 0.3)] = 0

    curve_indices = np.where((W_e_xaxis >= -0.4) & (W_e_xaxis <= 0.12))  # Get indices of the curve range
    curve_x = W_e_xaxis[curve_indices]  # X-values for the curve
    curve_y = 10*(curve_x + 0.14)**2 + 1.0  # Define the quadratic curve

    # Ensure element-wise assignment of curve_y
    for idx, val in zip(curve_indices[0], curve_y):
        water_rou_1D[idx] = val



    # Plot the boundary map
    if plt_hander:
        plt.plot(W_e_xaxis, water_rou_1D,'*-')
        plt.title("Modified Water ROU 1D")
        plt.xlabel("X-axis")
        plt.ylabel("Water ROU")
        plt.grid()
        plt.show()
    return water_rou_1D

def confidence_variation_voxels(n1:float, n2:float, W_max:float, K:int, ak: list)->float:
        """Calculation of confidence between the two investigated voxels

        Args:
            n1 (float): Values of the referenced voxels
            n2 (float): Values of the referenced voxels
            W_max (float): max value for c_n1_n2 = 1/W_max confidence calculation
            K (int): Number of sources of prior information.
            ak (list): Weight for the k-th source of prior information (e.g., based on its relevance).

        Returns:
            float: confidence value between the two investigated voxels
        """

        c_n1_n2_sum = 0
        for  _ak in  ak:
            c_n1_n2_sum = _ak * (abs(n1-n2)**2)
        c_n1_n2 = c_n1_n2_sum/K
        if c_n1_n2 == 0:
            c_n1_n2 = 1/W_max
        return c_n1_n2

# def min_pool_adjacent_Voxel(Buffer,constraint_img,pool_size):
#          # Handle 1D input by converting it to 2D (row vector)
#         if Buffer.ndim == 1:
#             original_1d = True
#             matrix = matrix.reshape(1, -1)  # Convert to shape (1, n)
#         else:
#             original_1d = False

#         # Dimensions of the input matrix
#         m, n = constraint_img.shape

#         # Initialize the output matrix
#         pooled_matrix = np.zeros((m, n))  # Same size as the original matrix

#         # Perform min pooling with stride = 1
#         for i in range(m):
#             for j in range(n):
#                 # Define the current pooling window
#                 start_row = i
#                 start_col = j
#                 end_row = start_row + pool_size
#                 end_col = start_col + pool_size

#                 # Apply min pooling in the current window
#                 pooled_matrix[start_row:end_row, start_col:end_col] = np.min(Buffer[start_row:end_row, start_col:end_col,:])

#         # If the input was 1D, return a flattened result
#         if original_1d:
#             return pooled_matrix.flatten()

#         return pooled_matrix

def min_pool_adjacent_Voxel(Buffer,constraint_img,pool_size):
         # Handle 1D input by converting it to 2D (row vector)
        if Buffer.ndim == 1:
            original_1d = True
            matrix = matrix.reshape(1, -1)  # Convert to shape (1, n)
        else:
            original_1d = False

        # Dimensions of the input matrix
        m, n = constraint_img.shape

        # Initialize the output matrix
        pooled_matrix = np.zeros((m, n))  # Same size as the original matrix

        # Perform min pooling with stride = 1
        for i in range(m):
            for j in range(n):
                # Define the current pooling window
                start_row = i
                start_col = j
                end_row = start_row + pool_size
                end_col = start_col + pool_size

                # Apply min pooling in the current window
                pooled_matrix[start_row:end_row, start_col:end_col] = np.min(Buffer[start_row:end_row, start_col:end_col,:])

        # Perform  reversed min pooling with stride = 1
        for i in reversed(range(m)):
            for j in reversed(range(n)):
                start_row = i
                start_col = j
                end_row = min(start_row + pool_size, m)
                end_col = min(start_col + pool_size, n)

                pooled_matrix[start_row:end_row, start_col:end_col] = np.minimum(
                    pooled_matrix[start_row:end_row, start_col:end_col],
                    np.min(Buffer[start_row:end_row, start_col:end_col, :])
                )

        
        
        # If the input was 1D, return a flattened result
        if original_1d:
            return pooled_matrix.flatten()

        return pooled_matrix



def constraints_to_B(constraints_matrix, W_max:float, pool_size:int, K:int = 1, ak:list = [1], minpooling_Handler:bool = True):
    """
    Generate the finite-difference matrix (B) from a general-shaped physical boundary constraints matrix.
    Parameters:
        input_matrix (np.ndarray): A general-shaped matrix of any size (e.g., M x N).
        Each element corresponds to a voxel with a weight.
        
    Returns:
        scipy.sparse.lil_matrix: Finite-difference matrix B, where each row represents
        a pairwise neighbor relationship with computed weights.
    """
    # Deal with 1D input matrix
    if constraints_matrix.ndim == 1:
        constraints_matrix = constraints_matrix.reshape(1, -1) #  reshape it into a 2D matrix with a single row
        
    # Dimensions of the input matrix
    rows, cols = constraints_matrix.shape
    
    # Generate neighbor pairs (4-connectivity)
    neighbor_pairs = []
    
    # Plot out the Weight Matrix
    W_matrix = []
        
    for i in range(rows):
        for j in range(cols):
            if j + 1 < cols:  # Right neighbor
                neighbor_pairs.append(((i, j), (i, j + 1)))
            if i + 1 < rows:  # Bottom neighbor
                neighbor_pairs.append(((i, j), (i + 1, j)))
                
    # Number of neighbor pairs
    P = len(neighbor_pairs)
    
    # Total number of voxels (N)
    N = rows * cols
    
    # Create an empty sparse matrix for B
    B = lil_matrix((P, N))
    
    if not minpooling_Handler or pool_size ==1:
        # Populate B matrix
        for p, ((i1, j1), (i2, j2)) in enumerate(neighbor_pairs):
            # Values of the neighboring voxels
            voxel_value_1 = constraints_matrix[i1, j1]
            voxel_value_2 = constraints_matrix[i2, j2]
            
            # Compute confidence variation
            confidence = confidence_variation_voxels(voxel_value_1, voxel_value_2, W_max, K, ak)
            
            # Compute weight based on confidence
            weight = np.sqrt(np.minimum(W_max, 1 / confidence))
            
            # Map 2D coordinates to 1D indices for B matrix
            n1 = i1 * cols + j1
            n2 = i2 * cols + j2
            
            # Add weights to the finite-difference matrix
            B[p, n1] = weight
            B[p, n2] = -weight
            W_matrix.append(weight)
    
    if minpooling_Handler and pool_size !=1:
        # Create a large matrix with voxels(value replaced with W_max) and its connections
        # B_large = np.full((rows * 2 - 1, cols * 2 - 1), self.W_max)
        # Populate B matrix
        weight_buffer = [[[] for _ in range(cols)] for _ in range(rows)]
        for p, ((i1, j1), (i2, j2)) in enumerate(neighbor_pairs):
            # Values of the neighboring voxels
            voxel_value_1 = constraints_matrix[i1, j1]
            voxel_value_2 = constraints_matrix[i2, j2]

            # Compute confidence variation
            confidence = confidence_variation_voxels(voxel_value_1, voxel_value_2, W_max, K, ak)
            
            # Compute weight based on confidence
            weight = np.sqrt(np.minimum(W_max, 1 / confidence))

            #save weights for further min-pooling
            weight_buffer[i1][j1].append(weight)

        # Pad weight_buffer to make it rectangular
        max_length = max(len(cell) for row in weight_buffer for cell in row)  # Find the longest sub-list
        for i in range(rows):
            for j in range(cols):
                # Pad each sub-list with self.W_max
                weight_buffer[i][j] = weight_buffer[i][j] + [W_max] * (max_length - len(weight_buffer[i][j]))

        
        weight_buffer = np.array(weight_buffer)
        weight_buffer = min_pool_adjacent_Voxel(weight_buffer, constraints_matrix, pool_size)
        
        for p, ((i1, j1), (i2, j2)) in enumerate(neighbor_pairs):
            # Map 2D coordinates to 1D indices for B matrix
            n1 = i1 * cols + j1
            n2 = i2 * cols + j2
            
            # Add weights to the finite-difference matrix
            B[p, n1] = weight_buffer[i1][j1]
            B[p, n2] = -weight_buffer[i1][j1]
            W_matrix.append(weight_buffer[i1][j1])

        
    return B, W_matrix, P

def Plot_W_WE(W_matrix:list, P:int):
        """Plot out the Weight Mtrix

        Args:
            W_matrix (list): edge preserving pair values
            P (int): number of pairs
        """
        # # Define x-axis
        # W_matrix_xaxis = np.linspace(-0.5, 0.5, P)
        # Plot the boundary map
        plt.plot(W_matrix,marker = '*')
        plt.title("Weight Matrix of Edge Preserving")
        plt.xlabel("X-axis")
        plt.ylabel("Weight")
        plt.grid()
        plt.show()

def SPICEWithSpatialConstrain(noisy_kt_spaces:np.ndarray, 
                              F:np.ndarray, 
                              V:np.ndarray,
                              K_POINTS:int, 
                              NUM_SPICE_RANK:int,
                              W_edge:np.ndarray,
                              Solver:str = "cg",
                              lamda_1:float = 15) -> tuple:
        lamda_1 = lamda_1 # edge preserving weighting
        # Matrix-Vector Multiplication Function
        
        #Conjugate Grandient Solver
        if Solver == "cg" or Solver =="Conjugate Grandient":
            def mv(x):
                return ((F.conj().T @ F @ (x.reshape(K_POINTS, NUM_SPICE_RANK) @ V.conj().T)) @ V).ravel() + lamda_1 * (W_edge.conj().T @ W_edge @ x.reshape(K_POINTS, NUM_SPICE_RANK)).ravel() #+ ll * X
            
            
            # A is a LinearOperator that behaves like a large matrix but uses mv(x) for matrix-vector multiplication, avoiding the need to store a potentially huge matrix in memory.
            A = LinearOperator(shape=(K_POINTS*NUM_SPICE_RANK, K_POINTS*NUM_SPICE_RANK), matvec=mv) 
            # D is the Fourier-encoded version of the noisy data (noisy_kt_space) after applying the conjugate transpose of F.
            D = F.conj().T @ noisy_kt_spaces  #matrix FH(Hermitian F, F.conj.T)  is used to "undo" the encoding effect introduced by F
                                            # FH x F × x=FH × b
            # b is the right-hand side of the equation to be solved, created by projecting D onto the spectral basis V
            b = D @ V
            b.shape #(32, 10)
            
            # Conjugate Gradient Solver (cg)
            #cg is a conjugate gradient solver that iteratively solves the linear system A⋅X=b.
            X = cg(A, b.ravel())[0]
        
        #Analytical Solution
        if Solver == "analytical" or Solver =="Analytical":
            X = np.linalg.inv(np.eye(K_POINTS)+lamda_1 * W_edge.conj().T @ W_edge)@F.conj().T@noisy_kt_spaces@V ####timed 0.05 for trying####
            #X = inv(A)*b

        spice_est = X.reshape(K_POINTS,NUM_SPICE_RANK) @ V.conj().T
        U = X.reshape(K_POINTS,NUM_SPICE_RANK)
        return  spice_est, U


'''
Uncert analysis related
'''
def plot_spec_ana(ax, rcon, ppm_axis, limits = [0,0.05]):
    """
    Plot uncertainty in the spectral domain for specified voxels.

    :param rcon: Spectral data (e.g., standard deviation of reconstructions)
    :param ppm_axis: Array of ppm values
    :param limits: Tuple specifying limits for the z-axis
    """

    # Plot the curves for the specified voxels
    for voxel in [2, 8, 16, 24, 30]:  # Indices of specified voxels
        ax.plot(ppm_axis, [voxel] * len(ppm_axis), rcon[:, voxel].real)

    # Set axis labels and limits
    ax.set_zlim(limits)
    ax.set_xlim([ppm_axis[-1], ppm_axis[0]])
    ax.set_xlabel('$\\delta$ / ppm')
    ax.set_ylabel('Voxel #')
    ax.set_zlabel('Uncertainty')

    # Adjust the viewing angle
    ax.view_init(elev=15, azim=-80)

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()


def plot_mc_compare_spec(res_array1: np.ndarray,res_array2: np.ndarray,res_array3: np.ndarray,ppm_axis,plot_spec_mc,plot_spec_ana):
        """
        Summarize and plot Monte Carlo results.

        :param res_array: Array of Monte Carlo reconstructions
        :param plot_and_rmse: Function to plot and calculate RMSE
        """
        # plot_and_rmse(res_array.mean(axis=0))

        res_array_spec1 = FIDToSpec(res_array1, axis=-1)
        res_array_spec2 = FIDToSpec(res_array2, axis=-1)
        res_array_spec3 =res_array3#FIDToSpec(res_array3, axis=-1)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.set_title("Spectral Uncert for cg")
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.set_title("Spectral Uncert for analy")
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.set_title("Spectral Uncert for analy Laplace")
        ax1.autoscale()
        ax2.autoscale()
        ax3.autoscale()

        # print((res_array_spec.std(axis=0)).shape)

        plot_spec_mc(ax1, res_array_spec1.std(axis=0),ppm_axis)
        plot_spec_mc(ax2, res_array_spec2.std(axis=0),ppm_axis)
        plot_spec_ana(ax3, res_array_spec3,ppm_axis)
        # plot_spatial_mc(ax2, res_array_spec.std(axis=0), water_rou_1D = water_rou_1D)
        plt.show()

def plot_mc_compare_spat(res_array1: np.ndarray,res_array2: np.ndarray,plot_spatial_mc, water_rou_1D):
        """
        Summarize and plot Monte Carlo results.

        :param res_array: Array of Monte Carlo reconstructions
        :param plot_and_rmse: Function to plot and calculate RMSE
        """
        # plot_and_rmse(res_array.mean(axis=0))

        res_array_spec1 = FIDToSpec(res_array1, axis=-1)
        res_array_spec2 = FIDToSpec(res_array2, axis=-1)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("Spatial Uncert for cg")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("Spatial Uncert for analy")
        # ax1.autoscale()
        # ax2.autoscale()

        # print((res_array_spec.std(axis=0)).shape)
        plot_spatial_mc(ax1, res_array_spec1.std(axis=0), water_rou_1D = water_rou_1D)
        plot_spatial_mc(ax2, res_array_spec2.std(axis=0), water_rou_1D = water_rou_1D)
        plt.show()

def runmc(recon_func:callable, 
          add_noise:callable,
          gen_undersample:callable,
          UNDER_SAMPLE_NUM: int,  
          seed_mc:int,
          W_edge:np.ndarray,
          lamda_1:float,
          Solver:str,
          kt_space_gt:np.ndarray,
          F:np.ndarray,
          V:np.ndarray,
          K_POINTS:int, 
          NUM_SPICE_RANK: int,
          time_axis:np.ndarray, 
          noise_SD:float,
          iterations:int=500,
          handler:bool = False,) -> tuple:
        """
        Run Monte Carlo simulations for SPICE reconstruction.

        :param recon_func: Reconstruction function
        :param gen_ktspace: Function to generate k-space data
        :param Peak_lws_gt_mc: Ground truth linewidths for Monte Carlo
        :param K_POINTS: Number of k-space points
        :param iterations: Number of Monte Carlo iterations
        :return: Array of reconstructions
        """
        output = []
        output_U = []
        rng = np.random.default_rng(seed=seed_mc)  # create rng
        # kt_groundtruth = np.copy(self.kt_space_gt)
        # print("I've reloaded")
        for _ in range(iterations):
            # kt_groundtruth = self.kt_space_gt
            gen_ktspace = add_noise(kspace=kt_space_gt, rng=rng, noise_SD = noise_SD)
            # assert np.allclose(self.kt_space_gt, kt_groundtruth)
            gen_undersample_result, new_F = gen_undersample(noisy_kt_space_US = gen_ktspace,F_US = F,  
                                                     UNDER_SAMPLE_NUM = UNDER_SAMPLE_NUM, K_POINTS= K_POINTS, time_axis = time_axis, handler = handler)
            result_spec_est,result_U = recon_func(noisy_kt_spaces = gen_undersample_result,F = new_F,V = V, W_edge = W_edge,
                                                  lamda_1 = lamda_1,Solver=Solver, K_POINTS = K_POINTS, NUM_SPICE_RANK = NUM_SPICE_RANK)
            output.append(result_spec_est)
            output_U.append(result_U)
            # Print current iteration number every 50 steps
            if _ % 50 == 0:
                print(f"Iteration {_} completed.")
        return np.asarray(output), np.asarray(output_U)


'''
    Below are functions for Analytical Method
'''
def calc_Covariance_old(NOISE_SD,lamda_1,W_edge,K_POINTS):
        Cov = NOISE_SD**2 * np.linalg.inv(np.eye(K_POINTS) + lamda_1 * W_edge.conj().T @ W_edge)
        print('shape of Cov:', Cov.shape)
        return Cov


def Calc_Uncert_Amat(NOISE_SD:float,
                    lamda_1:float,
                    W_edge:np.ndarray,
                    K_POINTS:int) -> np.ndarray:
    A = (1/(NOISE_SD**2)) * (np.eye(K_POINTS) + lamda_1 * W_edge.T @ W_edge)
    # print('shape of A:', A.shape)
    return A

def Select_freq_w(freq,V):
    Vh = V.conj().T
    Vh_Spec = []
    for i in range(Vh.shape[0]):  # Iterate over rows of Vh
        Vh_Spec.append(np.array(FID2Spec(Vh[i, :])))  # Apply FID2Spec and store result
    # print('shape of V:', Vh.shape)
    # Convert Vh_Spec to a NumPy array (stacking along axis 0)
    Vh_Spec = np.vstack(Vh_Spec)  # Ensures proper 2D array format

    # Compute sum of squared magnitudes at frequency index `freq`
    square_sum = np.sum(Vh_Spec[:, freq] * np.conjugate(Vh_Spec[:, freq]))
    return square_sum

def calc_Covariance_spat(NOISE_SD:float,
                         lamda_1:float,
                         W_edge:np.ndarray,
                         K_POINTS:int,
                         V:np.ndarray,
                         freq:int):
    A = Calc_Uncert_Amat(NOISE_SD,lamda_1,W_edge,K_POINTS)
    w = Select_freq_w(freq,V)
    Cov = w*np.linalg.inv(A)
    # print('cov value:', Cov)
    # print('shape of cov:', Cov.shape)
    return Cov

def calc_Covariance_spat_overall(
                         NOISE_SD:float,
                         lamda_1:float,
                         W_edge:np.ndarray,
                         K_POINTS:int,
                         V:np.ndarray
                         ):
    A = Calc_Uncert_Amat(NOISE_SD,lamda_1,W_edge,K_POINTS)
    def unmasked_w(V):
        Vh = V.conj().T
        Vh_Spec = []
        for i in range(Vh.shape[0]):  # Iterate over rows of Vh
            Vh_Spec.append(np.array(FID2Spec(Vh[i, :])))  # Apply FID2Spec and store result
        # print('shape of V:', Vh.shape)
        # Convert Vh_Spec to a NumPy array (stacking along axis 0)
        Vh_Spec = np.vstack(Vh_Spec)  # Ensures proper 2D array format

        # Compute sum of squared magnitudes at frequency index `freq`
        square_sum = np.sum(Vh_Spec[:] * np.conjugate(Vh_Spec[:]))
        return square_sum
    w = unmasked_w(V)
    # Cov = w*np.linalg.inv(A)
    Cov = np.linalg.inv(A).astype(np.complex128)
    # print('cov value:', Cov)
    # print('shape of cov:', Cov.shape)
    return Cov

def complex_multivariate_normal(mean, cov, n_samples):
    """
    Sample from a complex multivariate normal distribution.
    mean: (n,) complex vector
    cov: (n, n) complex Hermitian covariance matrix
    n_samples: int
    Returns: (n_samples, n) complex samples
    """
    # n = mean.shape[0]

    # # Cholesky decomposition
    # L = np.linalg.cholesky(cov)

    # # Standard complex normal: real and imaginary parts N(0, 0.5)
    # w_real = np.random.normal(0, 1 / np.sqrt(2), size=(n_samples, n))
    # w_imag = np.random.normal(0, 1 / np.sqrt(2), size=(n_samples, n))
    # w = w_real + 1j * w_imag

    # # Generate complex samples
    # samples = w @ L.T + mean

    n = mean.shape[0]
    std_dev = np.sqrt(np.abs(np.diag(cov)))  # 标准差，确保正实数

    samples = np.zeros((n_samples, n), dtype=complex)
    for i in range(n):
        real_part = np.random.normal(loc=np.real(mean[i]), scale=std_dev[i], size=n_samples)
        imag_part = np.random.normal(loc=np.imag(mean[i]), scale=std_dev[i], size=n_samples)
        samples[:, i] = real_part + 1j * imag_part

    return samples



def calc_std_uncert( Cov):
    std_uncert = np.sqrt(np.abs(np.diag(Cov)))
    # std_uncert = np.sqrt(np.diag(Cov))
    # print('std_uncert value:', std_uncert)
    return std_uncert

def calc_uncert_array(NOISE_SD,lamda_1,W_edge,K_POINTS, V):
    uncert_array = []

    # Iterate over freq from 0 to 511.
    for freq in np.arange(N_SEQ_POINTS):
        cov_1 = calc_Covariance_spat(NOISE_SD, lamda_1, W_edge, K_POINTS, V, freq)
        uncert_1 = calc_std_uncert(cov_1)
        uncert_array.append(np.array(uncert_1))


    uncert_array = np.vstack(uncert_array) # turn into 2D numpy array
    print('shape of uncert_array:', uncert_array.shape)
    plt.matshow(abs(uncert_array))

    # Calculate the variance of each row
    variances = np.var(uncert_array, axis=1)

    # # Find the row number with the highest variance
    # max_variance_index = np.argmax(variances)
    max_variance_index = np.argpartition(variances, -5)[-5:]
    max_variance_index = max_variance_index[np.argsort(variances[max_variance_index])[::-1]]
    print(f"max_variance_index: {max_variance_index}")
    return uncert_array

def plot_spatial_ana(std_uncert, water_rou_1D=None, limits: list = [0,2.5], scale_factor=1, x_hr_pos =[]):
    """
    Plot the analytical uncertainty results along with optional additional data.

    :param std_uncert: Analytical uncertainty (array or matrix with uncertainty values).
    :param x_hr_pos: High-resolution x-axis positions (spatial positions for plotting).
    :param limits: Y-axis limits for the plot.
    :param W_matrix: Optional weight matrix to overlay.
    :param P: Optional additional matrix for further use (not used in this plot).
    :param W_e_xaxis: Optional additional x-axis for weights (not used in this plot).
    :param water_rou_1D: Optional water ROU data to overlay.
    :param scale_factor: Scaling factor for plotting water ROU data.
    """

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot analytical uncertainty
    if std_uncert is not None:
        ax.plot(x_hr_pos, std_uncert, label='Analytical Uncertainty', linestyle='--', linewidth=2)

    # Plot the Water ROU if provided
    if water_rou_1D is not None:
        ax.plot(x_hr_pos, water_rou_1D * scale_factor, label='Water ROU 1D', linestyle=':', linewidth=1.5)

    # Set plot labels and limits
    ax.set_ylim(limits)
    ax.set_xlabel('x (Spatial Position)', fontsize=12)
    ax.set_ylabel('Value / Uncertainty', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Display the plot
    plt.title('Spatial Analytical Uncertainty Results', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_spatial_mc_ana_combined(mc_cg,mc_analyt, std_uncert1_analyt,std_uncert2_analyt,x_hr_pos, water_rou_1D=None):
    res_array_spec1 = FIDToSpec(mc_cg, axis=-1)
    res_array_spec2 = FIDToSpec(mc_analyt, axis=-1)
    
    # Compute recon data
    peak2est_cg = np.abs(res_array_spec1.std(axis=0))[:, PEAK_0_ROUGH_IDX]
    peak1est_cg = np.abs(res_array_spec1.std(axis=0))[:, PEAK_1_ROUGH_IDX]
    peak2est_analyt = np.abs(res_array_spec2.std(axis=0))[:, PEAK_0_ROUGH_IDX]
    peak1est_analyt = np.abs(res_array_spec2.std(axis=0))[:, PEAK_1_ROUGH_IDX]
    # peak2est = (np.abs(recon)).std(axis=0)[:, 64]
    # peak1est = (np.abs(recon)).std(axis=0)[:, 162]
    # peak2est_cg /= 1.4  # Fudge factor
    # peak1est_cg /= 1.4  # Fudge factor
    # peak2est_analyt /= 1.4  # Fudge factor
    # peak1est_analyt /= 1.4  # Fudge factor
    xaxis = x_hr_pos

    # Plot the data
    print("Starting plot function...")
    fig, ax = plt.subplots()
    print("Figure and Axes created.")

    ax.plot(xaxis, peak1est_cg , label='Peak 1 cg MC Uncertainty')
    ax.plot(xaxis, peak2est_cg , label='Peak 2 cg MC Uncertainty')
    ax.plot(xaxis, peak1est_analyt, label='Peak 1 analyt MC Uncertainty', linestyle='dashdot', linewidth=2)
    ax.plot(xaxis, peak2est_analyt, label='Peak 2 analyt MC Uncertainty', linestyle='dashdot', linewidth=2)
    ax.plot(xaxis, std_uncert1_analyt, label='Analytical Uncertainty Peak1', linestyle='--', linewidth=2)
    ax.plot(xaxis, std_uncert2_analyt, label='Analytical Uncertainty Peak2', linestyle='--', linewidth=2)

    # Add Water ROU plot if data is provided
    if water_rou_1D is not None:
        if len(water_rou_1D) == len(xaxis):  # Ensure shape compatibility
            ax.plot(xaxis, water_rou_1D*0.02, label='Water ROU 1D', linestyle=':')
        else:
            print("Shape mismatch: Water ROU 1D and x-axis lengths differ. Skipping plot.")

    # Enhance the plot
    ax.set_ylim([0,1e-2])
    ax.set_xlabel('x')
    ax.set_ylabel('Uncertainty')
    ax.set_title('Spatial Uncertainty Results', fontsize=14)
    ax.legend()
    # self.ax.set_ylim(limits)
    # self.ax.legend(title=f"mc2ana Scale Factor: {mc2ana_Scale:.2f},\n Enlarge Scale Factor: {enlarge_scale:.2f}")
    # plt.tight_layout()


    plt.show()
    return peak1est_cg,std_uncert1_analyt


def plot_spec_analyt( ax, rcon,ppm_axis,limits):
    """
    Plot uncertainty in the spectral domain for specified voxels.

    :param rcon: Spectral data (e.g., standard deviation of reconstructions)
    :param ppm_axis: Array of ppm values
    :param limits: Tuple specifying limits for the z-axis
    """

    # Plot the curves for the specified voxels
    for voxel in [2, 8, 16, 24, 30]:  # Indices of specified voxels
        ax.plot(ppm_axis, [voxel] * len(ppm_axis), rcon[:, voxel].real)

    # Set axis labels and limits
    ax.set_zlim(limits)
    ax.set_xlim([ppm_axis[-1], ppm_axis[0]])
    ax.set_xlabel('$\\delta$ / ppm')
    ax.set_ylabel('Voxel #')
    ax.set_zlabel('Uncertainty')

    # Adjust the viewing angle
    ax.view_init(elev=15, azim=-80)

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()


def plot_mc_compare_spec(res_array1: np.ndarray,res_array2: np.ndarray,res_array3: np.ndarray,plot_spec_mc,plot_spec_analyt,ppm_axis,limits):
        """
        Summarize and plot Monte Carlo results.

        :param res_array: Array of Monte Carlo reconstructions
        :param plot_and_rmse: Function to plot and calculate RMSE
        """
        # plot_and_rmse(res_array.mean(axis=0))

        res_array_spec1 = FIDToSpec(res_array1, axis=-1)
        res_array_spec2 = FIDToSpec(res_array2, axis=-1)
        res_array_spec3 =res_array3#FIDToSpec(res_array3, axis=-1)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.set_title("Spectral Uncert for cg")
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.set_title("Spectral Uncert for analy")
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.set_title("Spectral Uncert for analy Laplace")
        ax1.autoscale()
        ax2.autoscale()
        ax3.autoscale()

        # print((res_array_spec.std(axis=0)).shape)

        plot_spec_mc(ax1, res_array_spec1.std(axis=0),ppm_axis,limits)
        plot_spec_mc(ax2, res_array_spec2.std(axis=0),ppm_axis,limits)
        plot_spec_analyt(ax3, res_array_spec3,ppm_axis,limits)
        # plot_spatial_mc(ax2, res_array_spec.std(axis=0), water_rou_1D = water_rou_1D)
        plt.show()

def plot_mc_compare_spat(res_array1: np.ndarray,res_array2: np.ndarray,plot_spatial_mc, water_rou_1D,x_hr_pos,limits):
        """
        Summarize and plot Monte Carlo results.

        :param res_array: Array of Monte Carlo reconstructions
        :param plot_and_rmse: Function to plot and calculate RMSE
        """
        # plot_and_rmse(res_array.mean(axis=0))

        res_array_spec1 = FIDToSpec(res_array1, axis=-1)
        res_array_spec2 = FIDToSpec(res_array2, axis=-1)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("Spatial Uncert for cg")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("Spatial Uncert for analy")
        # ax1.autoscale()
        # ax2.autoscale()

        # print((res_array_spec.std(axis=0)).shape)
        plot_spatial_mc(ax1, res_array_spec1.std(axis=0), water_rou_1D = water_rou_1D,x_hr_pos = x_hr_pos,limits =limits)
        plot_spatial_mc(ax2, res_array_spec2.std(axis=0), water_rou_1D = water_rou_1D,x_hr_pos = x_hr_pos,limits =limits)
        plt.show()


def plot_spec_mc( ax, rcon,ppm_axis,limits):
    """
    Plot uncertainty in the spectral domain for specified voxels.

    :param ax: Matplotlib 3D axis for plotting
    :param rcon: Spectral data (e.g., standard deviation of reconstructions)
    :param limits: Limits for the z-axis
    """
    for voxel in [2, 8, 16, 24, 30]:  # Specify voxel indices
        ax.plot(ppm_axis, rcon[voxel, :].real, zs=voxel, zdir='y')  # Use `voxel` as the z-coordinate
    ax.set_zlim(limits)
    ax.set_xlim([ppm_axis[-1], ppm_axis[0]])
    ax.set_xlabel('$\\delta$ / ppm')
    ax.set_ylabel('Voxel #')
    ax.set_zlabel('Uncertainty')
    ax.view_init(elev=15, azim=-80, roll=0)


def plot_spatial_mc( ax, recon, water_rou_1D=None,x_hr_pos=[],limits =[0,1e-2],scale_factor=1):
    """
    Plot spatial uncertainty with additional data overlays.

    :param ax: Matplotlib axis for plotting
    :param recon: Reconstructed data
    :param limits: Limits for the y-axis
    :param W_matrix: Weight matrix (optional)
    :param P: Additional parameter (optional)
    :param W_e_xaxis: x-axis data for weight matrix (optional)
    :param water_rou_1D: Water density data (optional)
    """
    peak2est = np.abs(recon)[:, PEAK_0_ROUGH_IDX]
    peak1est = np.abs(recon)[:, PEAK_1_ROUGH_IDX]
    # peak2est /= 1.4  # Fudge factor
    # peak1est /= 1.4  # Fudge factor
    xaxis = x_hr_pos
    ax.plot(xaxis, peak1est, label='Peak 1 Estimation')
    ax.plot(xaxis, peak2est, label='Peak 2 Estimation')
    ax.set_ylim(limits)
    ax.set_xlabel('x')
    ax.set_ylabel('Uncertainty')

    if water_rou_1D is not None:
        ax.plot(xaxis, water_rou_1D * scale_factor, label='Water ROU 1D', linestyle=':')

    ax.legend()

def Create_laplacian_samples(spice_est:np.ndarray,
                             Vh:np.ndarray,
                             cov_overall:np.ndarray,
                             n_samples: int ) -> np.ndarray:
    """using the laplacian approximation calculated covarience 
    to generate a span of data for MC calculation of fitting uncertainty

    Args:
        spice_est_cg (np.ndarray): estimated KI-space from a signle run of SPICE
        cov_overall (np.ndarray): laplacian method calculated covarience
        n_samples (int): number of sample for data points

    Returns:
        np.ndarray: generated data for MC uncertainty with laplacian method
    """
    n_dim, n_channels = spice_est.shape

    # # Result: a 3D array of shape (1000, 32, 512)
    # samples = np.zeros((n_samples, n_dim, n_channels))

    # for i in range(n_channels):
    #     mean_vec = spice_est_cg[:, i]
    #     samples[:, :, i] = np.random.multivariate_normal(mean=mean_vec, cov=cov_overall, size=n_samples)

    samples = np.zeros((n_samples, n_dim, n_channels), dtype=np.complex128)
    std_dev = np.sqrt(np.abs(np.diag(cov_overall))) 

    # for i in range(n_channels):
    #     mean_vec = spice_est[:, i]
    #     real_part = np.random.normal(loc=np.real(mean_vec)[None, :], scale=std_dev[None, :], size=(n_samples, n_dim))
    #     imag_part = np.random.normal(loc=np.imag(mean_vec)[None, :], scale=std_dev[None, :], size=(n_samples, n_dim))
    #     samples[:, :, i] = real_part + 1j * imag_part

    # Expand std_dev to shape (n_dim, n_channels)
    std_dev_matrix = np.repeat(std_dev[:, None], n_channels, axis=1)  # shape (n_dim, n_channels)

    # Sample real and imaginary parts
    real_samples = np.random.normal(loc=np.real(spice_est)[None, :, :],
                                    scale=std_dev_matrix[None, :, :],
                                    size=(n_samples, n_dim, n_channels))
    
    imag_samples = np.random.normal(loc=np.imag(spice_est)[None, :, :],
                                    scale=std_dev_matrix[None, :, :],
                                    size=(n_samples, n_dim, n_channels))

    samples = real_samples + 1j * imag_samples  # (n_samples, n_dim, n_channels)

    return samples@Vh


''' 
MRS fit
'''
# def Sig_func_Multi_Peak(bm_list, lw_list, Cm, time_axis,n_voxels):
#     taxis = np.array(time_axis).reshape(512, 1)
#     min_exp = -50
#     max_exp = 50

#     # Check that lw_list length matches bm_list length
#     assert len(lw_list) == len(bm_list), f"lw_list length {len(lw_list)} doesn't match bm_list length {len(bm_list)}."

#     fids = []
#     for i in range(len(bm_list)):  # For each peak
#         this_lw = lw_list[i]


#         # If it's a scalar,  -> extends to (32,)
#         if np.isscalar(this_lw):
#             lw_array = np.ones((1, n_voxels)) * this_lw
#         else:
#             # If it's an array, make sure the shape is (32,)
#             this_lw = np.asarray(this_lw)
#             assert this_lw.shape == (n_voxels,), f"Expected shape (32,), but got {this_lw.shape}"
#             lw_array = this_lw[np.newaxis, :]


#         if any([not (min_exp <= ll <= max_exp) for ll in lw_array[0]]):
#             raise ValueError(f'lw must be in the range {min_exp} to {max_exp}, it is {this_lw}.')

#         broadening = np.exp(-lw_array * np.pi * taxis)  # (512,32)

#         fid = (np.real(bm_list[i]).reshape(512,1)) * broadening  # (512,32)
#         fid = fid * Cm[i]  # multiplied by the concentration (scalar)
#         fids.append(fid)

#     total = np.sum(fids, axis=0).T  # (32,512)
#     return total

def Sig_func_Multi_Peak(bm_list, lw_list, Cm, time_axis, n_voxels):
    taxis = np.array(time_axis).reshape(N_SEQ_POINTS, 1)
    min_exp = -50
    max_exp = 50

    n_peaks = len(bm_list)
    fids = []

    # 自动转换成列表形式
    if isinstance(lw_list, np.ndarray):
        lw_list = [lw_list[i, :] if lw_list.ndim == 2 else lw_list[i] for i in range(n_peaks)]
    if isinstance(Cm, np.ndarray):
        Cm = [Cm[i, :] if Cm.ndim == 2 else Cm[i] for i in range(n_peaks)]

    for i in range(n_peaks):
        this_lw = lw_list[i]
        this_cm = Cm[i]

        # 如果是标量，扩展为 (n_voxels,)
        if np.isscalar(this_lw):
            this_lw = np.ones(n_voxels) * this_lw
        else:
            this_lw = np.asarray(this_lw)
            assert this_lw.shape == (n_voxels,), f"lw[{i}] shape {this_lw.shape} != ({n_voxels},)"

        if np.isscalar(this_cm):
            this_cm = np.ones(n_voxels) * this_cm
        else:
            this_cm = np.asarray(this_cm)
            assert this_cm.shape == (n_voxels,), f"Cm[{i}] shape {this_cm.shape} != ({n_voxels},)"

        if not np.all((min_exp <= this_lw) & (this_lw <= max_exp)):
            raise ValueError(f"lw must be in range [{min_exp}, {max_exp}], got {this_lw}.")

        lw_array = this_lw[np.newaxis, :]
        broadening = np.exp(-lw_array * np.pi * taxis)  # (512, n_voxels)

        fid = bm_list[i].reshape(N_SEQ_POINTS, 1) * broadening
        fid = fid * this_cm
        fids.append(fid)

    total = np.sum(fids, axis=0).T  # -> (n_voxels, 512)
    return total






def fit_mrs_spectrum_nonlinear(bm_FIDs, time_axis, spectrum, initial_guesses):
    """Fits an MRS spectrum using multiple Lorentzian peaks, with added input checks."""

    spectrum = np.real(spectrum.flatten())

    # === 安全检查：确认所有输入数值合法 ===
    if not (np.all(np.isfinite(spectrum)) and 
            np.all(np.isfinite(time_axis)) and 
            np.all(np.isfinite(bm_FIDs)) and 
            np.all(np.isfinite(initial_guesses))):
        raise ValueError("Input data contains NaN or Inf values.")

    n_peaks = len(bm_FIDs)
    n_voxels = N_VOXEL

    def model_func(x_dummy, *params):
        lw_list = []
        Cm_list = []

        idx = 0
        for _ in range(n_peaks):
            lw_list.append(params[idx:idx+n_voxels])
            idx += n_voxels
            Cm_list.append(params[idx:idx+n_voxels])
            idx += n_voxels

        result = Sig_func_Multi_Peak(bm_FIDs, lw_list, Cm_list, time_axis, n_voxels)
        return np.real(result.flatten())

    # Define bounds
    bounds_C_m = [(1e-5, 2)] * N_VOXEL
    bounds_lw = [(4.5, 13)] * N_VOXEL
    full_bounds = bounds_lw + bounds_C_m + bounds_lw + bounds_C_m

    lower_bounds = [b[0] for b in full_bounds]
    upper_bounds = [b[1] for b in full_bounds]

    # === 拟合 + 错误捕获 ===
    try:
        popt, pcov = curve_fit(model_func, time_axis, spectrum,
                               p0=initial_guesses, bounds=(lower_bounds, upper_bounds),
                               method='trf')  # 'trf' is more stable than default
    except Exception as e:
        print("Curve fitting failed:", e)
        return None, None

    return popt, pcov



def fit_mrs_spectrum_nonlinear_ls(bm_FIDs, time_axis, spectrum, initial_guesses):
    """Fits an MRS spectrum using least squares optimization."""
    
    # Ensure spectrum is real (take real part) if it contains complex numbers
    spectrum = np.real(spectrum)
    n_voxels =N_VOXEL
    
    def residuals(params):
        lw1 = params[0:N_VOXEL]
        C1 = params[N_VOXEL:2*N_VOXEL]
        lw2 = params[2*N_VOXEL:3*N_VOXEL]
        C2 = params[3*N_VOXEL:4*N_VOXEL]

        lw_list = [lw1, lw2]
        C_list = [C1, C2]

        pred = Sig_func_Multi_Peak(bm_FIDs, lw_list, C_list, time_axis,n_voxels)
        
        pred = np.real(pred)
        return pred.flatten() - spectrum.flatten()
    
    # Define bounds
    bounds_C_m = [(1e-5, 2)] * N_VOXEL
    bounds_lw = [(4.5, 13)] * N_VOXEL
    full_bounds = bounds_lw + bounds_C_m + bounds_lw + bounds_C_m  # total: 128 bounds

    lower_bounds = [b[0] for b in full_bounds]
    upper_bounds = [b[1] for b in full_bounds]

    result = least_squares(residuals, initial_guesses, bounds=(lower_bounds, upper_bounds), method='trf', max_nfev=5000)
    
    popt = result.x
    pcov = None

    return popt, pcov



def plot_popt(popt):
    """Plots optimized parameters."""
    lw1 = popt[0:N_VOXEL]
    fit1 = popt[N_VOXEL:2*N_VOXEL]
    lw2 = popt[2*N_VOXEL:3*N_VOXEL]
    fit2 = popt[3*N_VOXEL:4*N_VOXEL]
    
    # print(f"Optimized linewidths: lw1 = {lw1}, lw2 = {lw2}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(fit1, marker='o', linestyle='-', label='Fit 1')
    plt.plot(fit2, marker='s', linestyle='-', label='Fit 2')
    plt.xlabel('Voxel Index')
    plt.ylabel('Value')
    plt.title('Optimized Fit Parameters from MRS Fitting')
    plt.legend()
    plt.show()

def plot_bm(bm_FIDs:list):
    """Plots the spectral basis bm for indices 0 and 1."""
    bm_0 = FID2Spec(bm_FIDs[0])
    bm_1 = FID2Spec(bm_FIDs[1])

    plt.figure(figsize=(8, 5))
    plt.plot(np.real(bm_0), label="Real Part (bm 0)", linestyle="-")
    plt.plot(np.imag(bm_0), label="Imaginary Part (bm 0)", linestyle="dashed")
    plt.plot(np.real(bm_1), label="Real Part (bm 1)", linestyle="-.")
    plt.plot(np.imag(bm_1), label="Imaginary Part (bm 1)", linestyle="dotted")

    plt.xlabel("Frequency Index")
    plt.ylabel("Intensity")
    plt.title("Spectral Basis (bm_0 and bm_1)")
    plt.legend()
    plt.show()

def plot_bm_and_bmFID(bm_FIDs:list, ppm_axis):
    """Plots the spectral basis bm and its corresponding FID for indices 0 and 1 separately."""
    bm_FID_0 = bm_FIDs[0]
    bm_FID_1 = bm_FIDs[1]

    # bm_0 = np.fft.fftshift(np.fft.fft(bm_FID_0))
    bm_0 = FID2Spec(bm_FID_0)
    bm_1 = FID2Spec(bm_FID_1)

    # Plot bm_0 and bm_1
    plt.figure(figsize=(8, 5))
    plt.plot(ppm_axis, np.real(bm_0), label="Real Part (bm 0)", linestyle="-")
    plt.plot(ppm_axis, np.imag(bm_0), label="Imaginary Part (bm 0)", linestyle="dashed")
    plt.plot(ppm_axis, np.real(bm_1), label="Real Part (bm 1)", linestyle="-")
    plt.plot(ppm_axis, np.imag(bm_1), label="Imaginary Part (bm 1)", linestyle="dashed")
    plt.xlabel("Frequency Index")
    plt.ylabel("Intensity")
    plt.title("Spectral Basis (bm_0 and bm_1)")
    plt.legend()
    plt.show()

    # Plot bm_FID_0 and bm_FID_1
    plt.figure(figsize=(8, 5))
    plt.plot(np.real(bm_FID_0), label="Real Part (bm_FID 0)", linestyle="-")
    # plt.plot(np.imag(bm_FID_0), label="Imaginary Part (bm_FID 0)", linestyle="dashed")
    plt.plot(np.real(bm_FID_1), label="Real Part (bm_FID 1)", linestyle="-.")
    # plt.plot(np.imag(bm_FID_1), label="Imaginary Part (bm_FID 1)", linestyle="dotted")
    plt.xlabel("Time Index")
    plt.ylabel("Intensity")
    plt.title("FID Signal (bm_FID_0 and bm_FID_1)")
    plt.legend()
    plt.show()

def mc_basis(spice_mc_U, bm_FIDs, taxis):
    iterations = len(spice_mc_U) if len(spice_mc_U) <= 5000 else 5000
    output_cm1 = []
    output_cm2 = []
    output_LW1 = []
    output_LW2 = []
    n_skipped = 0  # 计数拟合失败的 spectrum 数量

    for i in range(iterations):
        est_U = spice_mc_U[i,:,:]
        spectrum = est_U
        # spectrum = spectrum.flatten()  # Flatten to 1D array (16384,)

        # Flatten initial guesses into a single 1D array
        initial_guesses = [8,2,8,2]
        initial_guesses = np.array(initial_guesses, dtype=np.float64)

        popt, pcov = fit_mrs_spectrum_lstsq_batch_vbv(bm_FIDs,taxis,spectrum,initial_guesses)
         # 如果拟合失败，跳过当前 spectrum
        if popt is None:
            print("⚠️ 拟合失败，当前 spectrum 被跳过")
            n_skipped += 1
            continue
        lw1 = popt[0:N_VOXEL]
        fit1 = popt[N_VOXEL:2*N_VOXEL]
        lw2 = popt[2*N_VOXEL:3*N_VOXEL]
        fit2 = popt[3*N_VOXEL:4*N_VOXEL]
        output_cm1.append(fit1)
        output_cm2.append(fit2)
        output_LW1.append(lw1)
        output_LW2.append(lw2)

        # Print current iteration number every 50 steps
        if i % 50 == 0:
            print(f"Iteration {i} completed.")

        # plot_popt(popt)
    print(f"✅ 共跳过 {n_skipped} 个 spectrum")
    return np.asarray(output_cm1), np.asarray(output_cm2),np.asarray(output_LW1),np.asarray(output_LW2)


def mc_basis_vbv(spice_mc_U, bm_FIDs, taxis):
    iterations = min(len(spice_mc_U), 100)
    output_cm1 = []
    output_cm2 = []
    output_LW1 = []
    output_LW2 = []
    n_skipped = 0

    for i in range(iterations):
        est_U = spice_mc_U[i]  # shape: (N_VOXEL, N_TIME)
        cm1_iter = []
        cm2_iter = []
        lw1_iter = []
        lw2_iter = []

        all_success = True  # track if all voxels were successful

        for v in range(N_VOXEL):
            spectrum = est_U[v, :].flatten()

            initial_guesses = list([10.0]) + [5] + [10.0] + [5]  # 1 peak × 4 params
            initial_guesses = np.array(initial_guesses, dtype=np.float64)

            popt, _ = fit_mrs_spectrum_lstsq_batch_vbv(bm_FIDs, taxis, spectrum, initial_guesses)#fit_mrs_spectrum_nonlinear_vbv(bm_FIDs, taxis, spectrum, initial_guesses) # commented sue to using the new lstsq method

            if popt is None:
                all_success = False
                break  # 如果有一个 voxel 拟合失败，就跳过整次 i

            lw1_iter.append(popt[0])
            cm1_iter.append(popt[1])
            lw2_iter.append(popt[2])
            cm2_iter.append(popt[3])

        if not all_success:
            print(f"⚠️ 第 {i} 个 MC sample 中至少一个 voxel 拟合失败，跳过整个 sample")
            n_skipped += 1
            continue

        output_LW1.append(lw1_iter)
        output_LW2.append(lw2_iter)
        output_cm1.append(cm1_iter)
        output_cm2.append(cm2_iter)

        if i % 50 == 0:
            print(f"Iteration {i} completed.")

    print(f"✅ 共跳过 {n_skipped} 个 MC spectrum 样本")
    return np.array(output_cm1), np.array(output_cm2), np.array(output_LW1), np.array(output_LW2)

def fit_mrs_spectrum_nonlinear_vbv(bm_FIDs, time_axis, spectrum, initial_guesses):
    """
    Fits a single-voxel MRS spectrum using multiple Lorentzian peaks.
    bm_FIDs: list of basis signals (1D arrays of length T)
    spectrum: 1D array of length T
    initial_guesses: list or array, length = 4 × n_peaks
                     [lw1, C1, lw2, C2, ...]
    """
    spectrum = np.real(spectrum.flatten())
    n_peaks = len(bm_FIDs)

    if not (np.all(np.isfinite(spectrum)) and 
            np.all(np.isfinite(time_axis)) and 
            np.all(np.isfinite(initial_guesses)) and 
            all(np.all(np.isfinite(fid)) for fid in bm_FIDs)):
        raise ValueError("Input contains NaN or Inf.")

    def model_func(x_dummy, *params):
        lw_list = []
        Cm_list = []
        for i in range(n_peaks):
            lw_list.append(params[2*i])
            Cm_list.append(params[2*i + 1])

        # 这里传入 1 voxel 的 linewidth 和 Cm，需要包一层 []
        result = Sig_func_Multi_Peak(bm_FIDs, [np.array([lw]) for lw in lw_list],
                                                [np.array([cm]) for cm in Cm_list],
                                                time_axis, 1)
        return np.real(result.flatten())

    # Define bounds
    bounds_C_m = [(1e-5, 5)] 
    bounds_lw = [(4.5, 20)] 
    full_bounds = bounds_lw + bounds_C_m + bounds_lw + bounds_C_m

    lower_bounds = [b[0] for b in full_bounds]
    upper_bounds = [b[1] for b in full_bounds]

    try:
        popt, pcov = curve_fit(model_func, time_axis, spectrum,
                               p0=initial_guesses,
                               bounds=(lower_bounds, upper_bounds),
                               method='trf')
    except Exception as e:
        print("Curve fitting failed:", e)
        return None, None

    return popt, pcov



def fit_mrs_spectrum_lstsq_batch_vbv(bm_FIDs, time_axis, spectra, initial_guesses):
    """
    Batched fitting for multiple voxels using fit_single_voxel().
    Inputs:
        bm_FIDs: list of basis FIDs
        time_axis: 1D array of length T
        spectra: 2D array (N_voxel, T)
        initial_guesses: ignored (fit_single_voxel uses internal x0)
    Returns:
        popt_all: (N_voxel, 2 * n_peaks)
        pcov: None
    """
    # spectra = np.real(spectra)
    N_voxel = spectra.shape[0]
    n_params = 2 * len(bm_FIDs)

    lw1_list = []
    cm1_list = []
    lw2_list = []
    cm2_list = []
    failed_voxels = 0

    def fit_single_voxel(data, basis_fid, timeaxis, initial_guesses):

        def fwd_model(x, bases):
            fid = np.zeros((N_SEQ_POINTS), dtype=complex)
            concs = x[1::2]
            linewidths = x[0::2]
            for cm, lw, basis in zip(concs, linewidths, bases):
                fid += cm * basis * np.exp(-lw * np.pi * timeaxis)

            return fid
        
        def jac(x, bases):

            grad = []
            concs = x[1::2]
            linewidths = x[0::2]
            for cm, lw, basis in zip(concs, linewidths, bases):
                grad.append(- cm * np.pi * timeaxis * basis * np.exp(-lw * np.pi * timeaxis))  # dSdlw
                grad.append(basis * np.exp(-lw * np.pi * timeaxis))  # dSdc

            return np.asarray(grad)
        
        def loss(x):
            return np.linalg.norm(data - fwd_model(x, basis_fid))
        
        def grad(x):
            S = fwd_model(x, basis_fid)
            dS = jac(x, basis_fid)
            out = np.real(
                np.sum(
                    S * np.conj(dS) + np.conj(S) * dS - np.conj(data) * dS - data * np.conj(dS),
                    axis=1))

            return out
        
        x0 = initial_guesses#[8, 0.2, 8, 0.4]
        bounds = ((0, None), (0, None), (0, None), (0, None))
        xout = minimize(loss, x0, bounds=bounds, jac=grad)
        print(xout)
        return xout.x

    for v in range(N_voxel):
        try:
            spectrum = spectra[v, :]
            popt = fit_single_voxel(spectrum, bm_FIDs, time_axis, initial_guesses)
            lw1_list.append(popt[0])
            cm1_list.append(popt[1])
            lw2_list.append(popt[2])
            cm2_list.append(popt[3])
        except Exception as e:
            print(f"⚠️ Voxel {v} fitting failed:", e)
            failed_voxels += 1

    if failed_voxels > 0:
        print(f"⚠️ 共 {failed_voxels} 个 voxel 拟合失败")
    
    popt = np.concatenate([lw1_list, cm1_list, lw2_list, cm2_list])
    pcov = None

    return popt, pcov



