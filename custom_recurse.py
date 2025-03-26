from scipy.signal import argrelextrema
import uvd
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import click
from uvd.decomp.kernel_reg import KernelRegression


@click.command()
@click.option("--data_path", type=str, required=True)
@click.option("--subgoal_target", type=int, default=None)
def main(data_path, subgoal_target):
    pass


if __name__ == "__main__":
    main()
