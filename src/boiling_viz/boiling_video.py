import imageio
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, ImageMagickWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from typing import List, Union

from boiling_viz.fields import FieldBase, array_to_fields

def fig_to_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

FieldType = Union[
    str,
    np.ndarray,
    FieldBase, 
    List[FieldBase]
]

class BoilingVideoBuilder:
    def __init__(self, fields: FieldType):
        
        # NOTE: these if statements are intended to fall through:
        #   - hdf5 -> numpy -> [FieldBase]
        if isinstance(fields, str):
            assert fields.endswith(".hdf5"), "fields path must be hdf5 file"
            with h5py.File(fields, "r") as handle:
                sdf = handle["dfun"][:]
                temp = handle["temperature"][:]
                velx = handle["velx"][:]
                vely = handle["vely"][:]
            fields = np.stack((sdf, temp, velx, vely), axis=-1)
            
        if isinstance(fields, np.ndarray):
            fields = array_to_fields(fields)
        if isinstance(fields, FieldBase):
            fields = [fields]

        self.fields: List[FieldBase] = fields

    def make_video(
        self, 
        path: str,
        duration: int,
        colorbars: bool,
        step_counter: bool,
        field_titles: bool,
        transparent_nan: bool
    ):
        num_axes = len(self.fields)
        
        height, width = self.fields[0].field[0].shape
        aspect = width / height
        fig_height = 4  # base height in inches
        fig, axes = plt.subplots(1, num_axes, figsize=(fig_height * aspect * num_axes, fig_height), layout="constrained")   
        fig.patch.set_alpha(0)

        if isinstance(axes, plt.Axes):
            axes_iterable = [axes]
        else:
            axes_iterable = axes.ravel()
        
        for ax in axes_iterable:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        def animate(timestep: int):
            ims = []
            for ax, field in zip(axes_iterable, self.fields):
                ax.clear()
                im = field.plot(ax, timestep)
                ims.append(im)
                if colorbars:
                    field.colorbar(ax, im)
                if field_titles:
                    ax.set_title(field.name)
            if step_counter:
                axes_iterable[0].set_ylabel(f"Step {timestep + 1}") 
            return ims               
            
        timesteps = min(f.timesteps() for f in self.fields)
        anim = FuncAnimation(fig, animate, timesteps)

        fps = timesteps / (duration / 1000)
        savefig_kwargs={"transparent": True, "facecolor": "none"}
        
        if transparent_nan:
            # imagemagick is not pip installable, so only use it as writer
            # when needed for handling transparent pixels
            writer = ImageMagickWriter(fps=fps)
            writer.bin_path = lambda: 'magick'
        else:
            writer = PillowWriter(fps=fps)

        anim.save(path, writer=writer, savefig_kwargs=savefig_kwargs)
        
        plt.close()