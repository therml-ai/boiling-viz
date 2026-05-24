import h5py
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, ImageMagickWriter
import matplotlib.patheffects as pe
from PIL import Image
from typing import List, Union

from boiling_viz.fields import FieldBase, array_to_fields


def fig_to_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]


def apply_text_visibility(ax):
    stroke = [pe.withStroke(linewidth=2, foreground="white")]
    for text in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        text.set_color("black")
        text.set_path_effects(stroke)


class TransparentPillowWriter(PillowWriter):
    r"""
    This is basically a hack to get transparent background working.
    It is required that .save use disposal=2.
    """

    def grab_frame(self, **savefig_kwargs):
        # _validate_grabframe_kwargs(savefig_kwargs)
        self.fig.canvas.draw()
        buf = BytesIO()
        self.fig.savefig(buf, **{**savefig_kwargs, "format": "rgba", "dpi": self.dpi})
        im = Image.frombuffer(
            "RGBA", self.frame_size, buf.getbuffer(), "raw", "RGBA", 0, 1
        )
        if im.getextrema()[3][0] < 255:
            # This frame has transparency, so we'll just add it as is.
            self._frames.append(im)
        else:
            # Without transparency, we switch to RGB mode, which converts to P mode a
            # little better if needed (specifically, this helps with GIF output.)
            self._frames.append(im.convert("RGB"))

    def finish(self):
        self._frames[0].save(
            self.outfile,
            save_all=True,
            append_images=self._frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
            disposal=2,
        )


FieldType = Union[str, np.ndarray, FieldBase, List[FieldBase]]


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
            assert fields.shape[-1] == 4, "last axis needs sdf, temp, velx, vely"
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
        transparent_nan: bool,
    ):
        num_axes = len(self.fields)

        height, width = self.fields[0].field[0].shape
        aspect = width / height
        fig_height = 4  # base height in inches
        fig, axes = plt.subplots(
            1,
            num_axes,
            figsize=(fig_height * aspect * num_axes, fig_height),
            layout="constrained",
        )
        fig.patch.set_alpha(0)

        if isinstance(axes, plt.Axes):
            axes_iterable = [axes]
        else:
            axes_iterable = axes.ravel()

        def animate(timestep: int):
            ims = []
            for ax, field in zip(axes_iterable, self.fields):
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                im = field.plot(ax, timestep)
                ims.append(im)
                if colorbars:
                    field.colorbar(ax, im)
                if field_titles:
                    ax.set_title(field.name)
                apply_text_visibility(ax)
            if step_counter:
                axes_iterable[0].set_ylabel(f"Step {timestep + 1}")
            fig.canvas.draw()
            return ims

        timesteps = min(f.timesteps() for f in self.fields)
        anim = FuncAnimation(fig, animate, timesteps)

        fps = timesteps / (duration / 1000)
        savefig_kwargs = {"transparent": True, "facecolor": "none"}

        if transparent_nan:
            # imagemagick is not pip installable, so only use it as writer
            # when needed for handling transparent pixels
            writer = ImageMagickWriter(fps=fps)
            writer.bin_path = lambda: "magick"
        else:
            writer = TransparentPillowWriter(fps=fps)

        anim.save(path, writer=writer, savefig_kwargs=savefig_kwargs)

        plt.close()
