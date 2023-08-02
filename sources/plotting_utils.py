import os
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sources.LevelSetFunction import LevelSetFunction


class Config(Enum):
    IMAGES_PATH = 'images'


class PlottingUtils:

    def __init__(self, mesh):
        self.mesh = mesh

    def make_plots(
            self,
            displacement: np.ndarray,
            density: np.ndarray,
            velocity: np.ndarray,
            iteration: int
    ):
        self.draw(
            density,
            f'density/density{iteration}',
            ratio=1 / 3,
            norm=colors.Normalize(vmin=0, vmax=1),
            cmap='gray_r'
        )
        self.draw(
            velocity,
            f'velocity/velocity{iteration}',
            ratio=1 / 3,
            norm=colors.Normalize(vmin=-1, vmax=4),
            cmap='Blues_r'
        )
        self.plot_displ(
            displacement,
            density,
            scale_factor=1,
            file_name=f'displacement/displacement{iteration}',
            ratio = 1 / 3
        )

    def initial_domain_plot(self, density: np.ndarray, phi: np.ndarray):
        self.draw(
            density,
            'initial_domain',
            ratio=1 / 3,
            norm=colors.Normalize(vmin=0, vmax=1),
            cmap='gray_r'
        )
        self.draw(
            phi,
            'initial_phi',
            ratio=1 / 3
        )


    def plot_displ(self, displ, density, scale_factor: float, file_name: str, ratio: float = None):
        half = len(displ) // 2
        displacements = scale_factor * np.vstack((displ[:half], displ[half:])).T

        before = tri.Triangulation(
            x=self.mesh.coordinates2D[:, 0],
            y=self.mesh.coordinates2D[:, 1],
            triangles=self.mesh.nodes_of_elem
        )
        before.set_mask(density < 0.5)
        plt.triplot(before, color='#1f77b4')
    
        after = tri.Triangulation(
            x=self.mesh.coordinates2D[:, 0] + displacements[:, 0],
            y=self.mesh.coordinates2D[:, 1] + displacements[:, 1],
            triangles=self.mesh.nodes_of_elem
        )
        after.set_mask(density < 0.5)
        plt.triplot(after, color='#ff7f0e')

        ax = plt.gca()
        if ratio is not None:
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

        plt.grid()
        plt.savefig(os.path.join(Config.IMAGES_PATH.value, file_name), bbox_inches='tight')
        plt.close()

    def draw(self, elem_values: np.ndarray, file_name: str, ratio: float = None, norm=None, cmap='gray', colorbar_ticks=None):
        triangulation = tri.Triangulation(
            x=self.mesh.coordinates2D[:, 0],
            y=self.mesh.coordinates2D[:, 1],
            triangles=self.mesh.nodes_of_elem
        )

        fig, ax = plt.subplots()
        img = ax.tripcolor(triangulation, elem_values, cmap=cmap, norm=norm)

        cbar_ax = ax
        if ratio is not None:
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

            ax_div = make_axes_locatable(ax)
            cbar_ax = ax_div.append_axes('right', size='3%', pad='1%')

        fig.colorbar(img, cax=cbar_ax, ticks=colorbar_ticks)

        ax.set_xlabel('x')
        ax.set_ylabel('y', rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(os.path.join(Config.IMAGES_PATH.value, file_name), bbox_inches='tight')
        plt.close(fig)

    def plot_implicit_function(self, rbf: LevelSetFunction, file_name: str):
        dom_x = np.linspace(0, 180, 100)
        dom_y = np.linspace(0, 60, 100)
        X, Y = np.meshgrid(dom_x, dom_y)
        Z = np.array([
            rbf(v) for v in zip(X.flatten(), Y.flatten())
        ]).reshape(X.shape)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_box_aspect((np.ptp(X), np.ptp(Y), 20))
        ax.plot_surface([0, 180, 180, 0], [0, 0, 60, 60], np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), color='tab:blue')
        ax.plot_surface(X, Y, Z, cmap='viridis')

        ax.view_init(120, 30)

        plt.savefig(os.path.join(Config.IMAGES_PATH.value, file_name), bbox_inches='tight')
        plt.close(fig)
