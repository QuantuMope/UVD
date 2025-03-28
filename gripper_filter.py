import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Sample data - replace with your actual data
# This is just for demonstration
time = np.linspace(0, 200, 201)
xyz_dist = np.sin(time / 30) * 0.5 + 0.5  # Create a sample xyz_dist curve
gripper_action = np.zeros_like(time)
gripper_action[60:150] = 1.0  # Create a sample gripper action


# Create a function to detect when gripper position finishes changing
def detect_gripper_transitions(gripper_signal, window_size=10, threshold=1e-2):
    """
    Detects when the gripper position stabilizes after changing,
    ensuring a valid opening transition must occur before a valid closing transition

    Parameters:
    -----------
    gripper_signal : array-like
        The gripper position over time
    window_size : int
        Number of consecutive samples to check for stability
    threshold : float
        Maximum allowed variation to consider the signal stable

    Returns:
    --------
    transition_points : array-like
        Indices where transitions have completed
    transition_signal : array-like
        Signal that spikes at transition completion points
    """
    # Calculate the derivative of the signal
    diff = np.abs(np.diff(gripper_signal, prepend=gripper_signal[0]))

    # Initialize the output signal
    transition_signal = np.zeros_like(gripper_signal)
    transition_points = []

    # Track the gripper state to ensure proper open->close sequence
    # -1 = initial (no transition yet detected)
    #  0 = closed
    #  1 = open
    gripper_state = -1

    # Identify transition completion points (where derivative becomes near-zero after being non-zero)
    for i in range(window_size, len(diff)):
        # Check if current window is stable (all differences below threshold)
        window_stable = all(d < threshold for d in diff[i - window_size + 1:i + 1])

        # Check if previous point had significant change
        previous_changing = diff[i - window_size] >= threshold

        # Only process if we've just stabilized after a change
        if window_stable and previous_changing:
            transition_index = i - window_size + 1  # This is the start of the stability window

            # Determine if this is an opening or closing transition
            # Compare the average signal value before and after the transition
            before_value = np.mean(gripper_signal[max(0, transition_index - window_size):transition_index])
            after_value = np.mean(
                gripper_signal[transition_index:min(len(gripper_signal), transition_index + window_size)])

            # If signal increased, it's an opening transition
            if after_value > before_value:
                is_opening = True
            else:
                is_opening = False

            # Apply the state transition logic
            if is_opening:
                # Opening transitions are valid if we're in initial state or closed state
                if gripper_state in [-1, 0]:
                    gripper_state = 1  # Mark as open
                    transition_points.append(transition_index)
                    transition_signal[transition_index] = 1.0
            else:
                # Closing transitions are only valid if we're in open state
                if gripper_state == 1:
                    gripper_state = 0  # Mark as closed
                    transition_points.append(transition_index)
                    transition_signal[transition_index] = 1.0

    return np.array(transition_points), transition_signal


# # Apply the detection function
# transition_points, transition_signal = detect_gripper_transitions(gripper_action)
#
# # Plot the results
# plt.figure(figsize=(12, 6))
# plt.plot(time, xyz_dist, label='xyz_dist', linewidth=2)
# plt.plot(time, gripper_action, label='gripper action', linewidth=2)
# plt.plot(time, transition_signal, 'r--', label='gripper transition', linewidth=2)
#
# # Add vertical lines at transition points
# for tp in transition_points:
#     plt.axvline(x=time[tp], color='g', linestyle='--', alpha=0.7)
#
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.title('Gripper Transition Detection')
# plt.tight_layout()
# plt.show()


# Alternative approach: Using the derivative directly
def create_transition_spikes(gripper_signal, smoothing=3):
    """
    Creates spikes at the points where the gripper has completed a transition

    Parameters:
    -----------
    gripper_signal : array-like
        The gripper position over time
    smoothing : int
        Window size for smoothing the derivative

    Returns:
    --------
    spike_signal : array-like
        Signal with spikes at transition completion points
    """
    # Calculate derivative
    deriv = np.abs(np.diff(gripper_signal, prepend=gripper_signal[0]))

    # Smooth the derivative
    smoothed_deriv = np.convolve(deriv, np.ones(smoothing) / smoothing, mode='same')

    # Find peaks in the derivative - these are where changes are happening fastest
    peaks, _ = find_peaks(smoothed_deriv)

    # Create a spike signal
    spike_signal = np.zeros_like(gripper_signal)
    spike_signal[peaks] = 1.0

    return spike_signal


# # Apply the alternative approach
# spike_signal = create_transition_spikes(gripper_action)
#
# # Plot the results with the alternative approach
# plt.figure(figsize=(12, 6))
# plt.plot(time, xyz_dist, label='xyz_dist', linewidth=2)
# plt.plot(time, gripper_action, label='gripper action', linewidth=2)
# plt.plot(time, spike_signal, 'r--', label='transition spikes', linewidth=2)
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.title('Gripper Transition Spikes')
# plt.tight_layout()
# plt.show()