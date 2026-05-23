from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn # noqa: F401

def sdf_cmap():
    return "RdYlBu"

def phase_binary_cmap():
    return ListedColormap(["white", "black"])

def temp_green_cmap():
    temp_ranges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.134, 0.167,
                   0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    color_codes = ['#0000FF', '#0443FF', '#0E7AFF', '#16B4FF', '#1FF1FF', '#21FFD3',
                   '#22FF9B', '#22FF67', '#22FF15', '#29FF06', '#45FF07', '#6DFF08',
                   '#9EFF09', '#D4FF0A', '#FEF30A', '#FEB709', '#FD7D08', '#FC4908',
                   '#FC1407', '#FB0007']
    colors = list(zip(temp_ranges, color_codes))
    return LinearSegmentedColormap.from_list('temp_green', colors)

def temp_gray_black_cmap():
    bulk_color = "#D3D1C7"
    sat_color = "#888780"
    hot1_color = "#000000"
    hot2_color = "#000000"
    temp_ranges = [0.0, 0.02, 0.3, 1.0]
    colors = list(zip(temp_ranges, [bulk_color, sat_color, hot1_color, hot2_color]))
    return LinearSegmentedColormap.from_list("temp_gray_black", colors)

def vel_mag_cmap():
    return "rocket"