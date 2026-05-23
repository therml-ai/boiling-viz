import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable

from boiling_viz.cmap import phase_binary_cmap, sdf_cmap, temp_green_cmap, vel_mag_cmap


class FixedTwoSlopeNorm(TwoSlopeNorm):
    def autoscale(self, A):
        pass

    def autoscale_None(self, A):
        pass

    def scaled(self):
        return True


class FieldBase:
    def __init__(self, name, field, cmap):
        self.name = name
        self.field = field
        self.cmap = cmap

    def min(self):
        return self.field.min()

    def max(self):
        return self.field.max()

    def timesteps(self):
        return self.field.shape[0]

    def plot(self, ax, timestep: int):
        raise NotImplementedError

    def colorbar(self, ax, im):
        raise NotImplementedError


class SDF(FieldBase):
    def __init__(self, field: np.ndarray, cmap=sdf_cmap()):
        super().__init__("SDF", field, cmap)
        self.norm = FixedTwoSlopeNorm(vcenter=0, vmin=-6, vmax=0.00001)

    def plot(self, ax, timestep: int):
        sdf = self.field[timestep]
        assert sdf.ndim == 2, "SDF must be a 2D tensor (H, W)"
        im = ax.imshow(sdf, cmap=self.cmap, norm=self.norm)
        return im

    def colorbar(self, cax, im):
        sm = ScalarMappable(norm=self.norm, cmap=self.cmap)
        im.set_clim(vmin=self.norm.vmin, vmax=self.norm.vmax)
        return plt.colorbar(sm, cax=cax, fraction=0.05, pad=0.05)


class Phase(FieldBase):
    def __init__(self, field: np.ndarray, cmap=phase_binary_cmap()):
        super().__init__("Phase", field, cmap)

    def plot(self, ax, timestep):
        phase = self.field[timestep]
        assert phase.ndim == 2, "Phase must be a 2D tensor (H, W)"
        im = ax.imshow(
            (phase > 0).astype(float),
            vmin=0,
            vmax=1,
            cmap=self.cmap,
            interpolation="nearest",
        )
        return im

    def colorbar(self, ax):
        pass
        # plt.colorbar(im, ax, fraction=0.05, pad=0.05, vmin=0, vmax=1)


class Temperature(FieldBase):
    def __init__(
        self,
        field: np.ndarray,
        cmap=temp_green_cmap(),
        min_temp: float = None,
        max_temp: float = None,
    ):
        super().__init__("Temperature", field, cmap)
        self.min_temp = min_temp if min_temp else self.min()
        self.max_temp = max_temp if max_temp else self.max()
        self.norm = Normalize(vmin=self.min_temp, vmax=self.max_temp, clip=True)

    def plot(self, ax, timestep: int):
        temp = self.field[timestep]
        assert temp.ndim == 2
        im = ax.imshow(temp, cmap=self.cmap, norm=self.norm, interpolation="nearest")
        return im

    def colorbar(self, cax, im):
        sm = ScalarMappable(norm=self.norm, cmap=self.cmap)
        im.set_clim(vmin=self.norm.vmin, vmax=self.norm.vmax)
        return plt.colorbar(sm, cax=cax, fraction=0.05, pad=0.05)


class TemperatureTransparent(Temperature):
    def __init__(
        self,
        field: np.ndarray,
        cmap=temp_green_cmap(),
        min_temp: float = None,
        max_temp: float = None,
    ):
        cmap.set_bad(alpha=0.0)
        super().__init__(field, cmap, min_temp, max_temp)

    def plot(self, ax, timestep: int):
        temp = self.field[timestep]
        assert temp.ndim == 2
        temp_masked = np.where(temp <= 52, np.nan, temp)
        im = ax.imshow(
            temp_masked, cmap=self.cmap, norm=self.norm, interpolation="nearest"
        )
        return im

    def colorbar(self, cax, im):
        sm = ScalarMappable(norm=self.norm, cmap=self.cmap)
        im.set_clim(vmin=self.norm.vmin, vmax=self.norm.vmax)
        return plt.colorbar(sm, cax=cax, fraction=0.05, pad=0.05)


class VelMag(FieldBase):
    def __init__(self, field: np.ndarray, cmap=vel_mag_cmap()):
        super().__init__("Vel. Mag.", field, cmap)
        index = int(0.99 * self.field.size)
        self.vmax = np.sort(self.field.flatten())[index]
        self.norm = Normalize(vmin=0, vmax=self.vmax)

    def plot(self, ax, timestep: int):
        vel_mag = self.field[timestep]
        assert vel_mag.ndim == 2, "Vel mag must be a 2D tensor (H, W)"
        im = ax.imshow(vel_mag, cmap=self.cmap, norm=self.norm, interpolation="nearest")
        return im

    def colorbar(self, cax, im):
        sm = ScalarMappable(norm=self.norm, cmap=self.cmap)
        im.set_clim(vmin=self.norm.vmin, vmax=self.norm.vmax)
        return plt.colorbar(sm, cax=cax, fraction=0.05, pad=0.05)


def array_to_fields(arr: np.ndarray):
    assert arr.ndim == 4
    sdf = np.flip(arr[..., 0], 1)
    temp = np.flip(arr[..., 1], 1)
    xvel = np.flip(arr[..., 2], 1)
    yvel = np.flip(arr[..., 3], 1)
    return [
        SDF(sdf, cmap=sdf_cmap()),
        Temperature(temp, cmap=temp_green_cmap()),
        VelMag(np.sqrt(xvel**2 + yvel**2), cmap=vel_mag_cmap()),
    ]
