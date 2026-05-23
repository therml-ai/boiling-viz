from boiling_viz import boiling_video
import numpy as np

from boiling_viz.fields import array_to_fields, Phase, TemperatureTransparent
from boiling_viz.boiling_video import BoilingVideoBuilder
from boiling_viz.cmap import temp_gray_black_cmap

# Make a video with sdf, temp, and velocity magnitude
data = np.load("example/slice.npy")
fields = array_to_fields(data)
builder = BoilingVideoBuilder(fields)
builder.make_video(
    path="example/sample.gif", 
    duration=1000 * (1 / 10), 
    colorbars=False, 
    step_counter=True, 
    field_titles=True,
    transparent_nan=False
)

# Make a video of the phase field
data = np.load("example/slice.npy")
phase_field = Phase(np.flip((data[..., 0] > 0).astype(float), 1))
builder = BoilingVideoBuilder([phase_field])
builder.make_video(
    path="example/phase.gif", 
    duration=1000 * (1 / 10), 
    colorbars=False, 
    step_counter=True, 
    field_titles=True,
    transparent_nan=False
)

# Make a video of temp with a different cmap
data = np.load("example/slice.npy")
temp_field = TemperatureTransparent(np.flip(data[..., 1], 1), cmap=temp_gray_black_cmap())
builder = BoilingVideoBuilder([temp_field])
builder.make_video(
    path="example/black_temp.gif", 
    duration=1000 * (1 / 10), 
    colorbars=False,
    step_counter=False,
    field_titles=False,
    transparent_nan=True
)