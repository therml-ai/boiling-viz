import numpy as np
import os
import tempfile

from boiling_viz.cmap import phase_binary_cmap
from boiling_viz.fields import array_to_fields, FieldBase, SDF, Phase
from boiling_viz.boiling_video import BoilingVideoBuilder

def test_boiling_video():
    #data = np.random.randn(10, 64, 64, 4)
    data = np.load("example/slice.npy")
    fields = array_to_fields(data)
    builder = BoilingVideoBuilder(fields)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "rollout.gif")
        builder.make_video(path, 1000 * (1 / 10), False, True)
    
    assert all(isinstance(f, FieldBase) for f in builder.fields)
    assert isinstance(builder.fields[0], SDF)
    assert len(builder.fields) == 3
    
def test_boiling_phase_video():
    data = np.random.randn(10, 64, 64)
    phase_field = Phase(data, phase_binary_cmap())
    builder = BoilingVideoBuilder(phase_field)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "phase.gif")
        builder.make_video(path, 1000 * (1 / 10), False, True)
        
    assert isinstance(builder.fields[0], Phase)