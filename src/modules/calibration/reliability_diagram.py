from .calibration_curves import plot_calibration_curve


def generate_reliability_diagram(*args, **kwargs):
    # Wrapper for terminology
    plot_calibration_curve(*args, **kwargs)
