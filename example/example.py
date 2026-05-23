import numpy as np

from boiling_viz.fields import array_to_fields, Phase
from boiling_viz.boiling_video import BoilingVideoBuilder

# Make a video with sdf, temp, and velocity magnitude
data = np.load("example/slice.npy")
fields = array_to_fields(data)
builder = BoilingVideoBuilder(fields)
builder.make_video(
    path="example/sample.gif", 
    duration=1000 * (1 / 10), 
    colorbars=False, 
    step_counter=True, 
    field_titles=True
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
    field_titles=True
)