import imageio
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Union

from boiling_viz.fields import FieldBase, array_to_fields

def fig_to_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

class BoilingVideoBuilder:
    def __init__(self, fields: Union[str, np.array, FieldBase, List[FieldBase]]):
        
        # NOTE: these if statements can all fall through:
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
        self.fields = fields

    def make_video(
        self, 
        path: str,
        duration: int,
        colorbars: bool,
        step_counter: bool,
        field_titles: bool
    ):
        num_axes = len(self.fields)
        
        height, width = self.fields[0].field[0].shape
        aspect = width / height
        fig_height = 4  # base height in inches
        fig, axes = plt.subplots(1, num_axes, figsize=(fig_height * aspect * num_axes, fig_height), layout="constrained")   
        
        if isinstance(axes, plt.Axes):
            axes_iterable = [axes]
        else:
            axes_iterable = axes.ravel()
        
        for ax in axes_iterable:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        # Plot the first time for each field on their axis, 
        # get a list of im to update for later frames.
        ims = [
            field.plot(ax, timestep=0) 
            for ax, field in zip(axes_iterable, self.fields)
        ]

        if step_counter:
            axes_iterable[0].set_ylabel("Step 1")
            
        if field_titles:
            for ax, field in zip(axes_iterable, self.fields):
                ax.set_title(field.name)
        
        if colorbars:
            caxes = []
            for im, ax, field in zip(ims, axes_iterable, self.fields):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cb = field.colorbar(cax, im)
                caxes.append(cax)
        
        fig.canvas.draw()
        fig.set_layout_engine(None)
        
        # Iterate through remaining timesteps and just update axis with timestep's data
        with imageio.get_writer(path, duration=duration) as writer:
            timesteps = min(f.timesteps() for f in self.fields)
            for timestep in range(1, timesteps):    
                if step_counter:
                    axes_iterable[0].set_ylabel(f"Step {timestep + 1}")
                for ax, im, field in zip(axes_iterable, ims, self.fields):
                    im.set_data(field.field[timestep].data)
                  
                if colorbars:
                    for im, cax, field in zip(ims, caxes, self.fields):
                        cax.cla()
                        field.colorbar(cax, im)
                
                fig.canvas.draw()                               
                writer.append_data(fig_to_array(fig))

        plt.close()