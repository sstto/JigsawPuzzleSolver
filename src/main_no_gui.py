import sys
from Puzzle.Puzzle import Puzzle
import matplotlib.pyplot as plt
import argparse


fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = fig_size[0] * 2
plt.rcParams["figure.figsize"] = fig_size

Puzzle('../input/animal72.png')