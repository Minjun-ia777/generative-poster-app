# -*- coding: utf-8 -*-
"""
Final Generative Poster Creator (ipywidgets version)

This is an "all-in-one" application that combines the techniques
from Weeks 2-5 into a single interactive project.

This version is designed for Google Colab / Jupyter notebooks and
uses `ipywidgets.interact` for sliders and controls.

Features:
- Framework: `ipywidgets` (Week 5)
- Shapes: Includes 'Blob' (Week 2), 'Heart' (New), and 'Star' (New).
- Palettes: Loads palettes from 'palette.csv' (Week 5) in addition to defaults.
- 3D Effects: Shadows, highlights, and sliders for 'Perspective' and 'Shadow'.

To run, you need:
pip install matplotlib numpy scipy pandas ipywidgets
"""

import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import splprep, splev
# Import ipywidgets
from ipywidgets import interact, widgets

# --- 1. Palette and Data Setup (Week 5) ---
PALETTE_FILE = "palette.csv"
WIDTH, HEIGHT = 1200, 1600 # Define global constants for poster size

# Default palettes
COLOR_PALETTES_DICT = {
    "Pastel": {"palette": ["#FFADAD", "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF", "#A0C4FF", "#BDB2FF"], "bg": "#F0EAD6"},
    "Vivid": {"palette": ["#FF595E", "#FFCA3A", "#8AC926", "#1982C4", "#6A4C93", "#F58553"], "bg": "#FFFFFF"},
    "Monochrome": {"palette": ["#222222", "#555555", "#888888", "#BBBBBB", "#EEEEEE"], "bg": "#D1D1D1"}
}

def load_csv_palette():
    """Loads palettes from the CSV file (Week 5)."""
    if not os.path.exists(PALETTE_FILE):
        print(f"Warning: '{PALETTE_FILE}' not found. Creating an example file.")
        df_init = pd.DataFrame([
            {"name": "csv_sky", "r": 0.4, "g": 0.7, "b": 1.0},
            {"name": "csv_sun", "r": 1.0, "g": 0.8, "b": 0.2},
            {"name": "csv_forest", "r": 0.2, "g": 0.6, "b": 0.3}
        ])
        df_init.to_csv(PALETTE_FILE, index=False)

    try:
        df = pd.read_csv(PALETTE_FILE)
        # Convert r, g, b columns to hex colors
        colors = []
        for row in df.itertuples():
            # Ensure values are in the 0-255 range
            r, g, b = int(row.r * 255), int(row.g * 255), int(row.b * 255)
            r, g, b = max(0, min(r, 255)), max(0, min(g, 255)), max(0, min(b, 255))
            colors.append(f'#{r:02x}{g:02x}{b:02x}')

        if colors:
            print(f"'CSV_Palette' loaded with {len(colors)} colors.")
            # Add to the main dictionary
            COLOR_PALETTES_DICT["CSV_Palette"] = {"palette": colors, "bg": "#FAFAFA"}
        else:
            print("CSV palette was empty.")
    except Exception as e:
        print(f"Error loading '{PALETTE_FILE}': {e}")

# Load the CSV palette *before* creating the widget options
load_csv_palette()
PALETTE_NAMES = list(COLOR_PALETTES_DICT.keys())


# --- 2. Shape Functions (Weeks 2, 4, New) ---

def create_smooth_blob(center_x, center_y, min_radius, max_radius, num_points):
    """Creates points for a smooth blob (Week 2)."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = np.random.uniform(min_radius, max_radius, num_points)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    tck, u = splprep([x, y], s=0, per=True)
    unew = np.linspace(u.min(), u.max(), 1000)
    x_smooth, y_smooth = splev(unew, tck, der=0)
    # Return points relative to the center
    return np.array([x_smooth + center_x, y_smooth + center_y]).T

def create_heart_points(center_x, center_y, scale):
    """New! Creates points for a heart shape."""
    t = np.linspace(0, 2 * np.pi, 100)
    x = scale * 16 * np.sin(t)**3
    y = scale * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    # Center and scale
    return np.array([x + center_x, y + center_y]).T

def create_star_points(center_x, center_y, scale, num_points=5):
    """New! Creates points for a star shape."""
    angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
    outer_radius = scale * 10
    inner_radius = scale * 4
    radii = np.array([outer_radius if i % 2 == 0 else inner_radius for i in range(num_points * 2)])
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    # Center and scale
    return np.array([x + center_x, y + center_y]).T


# --- 3. Main Drawing Logic (Combined function for ipywidgets) ---

def draw_3d_shape(ax, points, color, alpha, shadow_offset):
    """
    New! Draws any selected shape with shadow and highlight.
    """

    # 2. Draw the Shadow (Week 4)
    shadow_points = points.copy()
    shadow_points[:, 0] += shadow_offset
    shadow_points[:, 1] -= shadow_offset
    shadow = patches.Polygon(shadow_points, facecolor='black', alpha=0.2, edgecolor='none')
    ax.add_patch(shadow)

    # 3. Draw the Main Shape
    polygon = patches.Polygon(points, facecolor=color, alpha=alpha, edgecolor='none')
    ax.add_patch(polygon)

    # 4. Draw the Highlight (New!)
    highlight_points = points.copy()
    # Move it slightly up and to the left
    highlight_points[:, 0] -= shadow_offset * 0.2
    highlight_points[:, 1] += shadow_offset * 0.2
    # Make it 90% of the size
    center = np.mean(points, axis=0)
    highlight_points = (highlight_points - center) * 0.9 + center

    highlight = patches.Polygon(highlight_points, facecolor='white', alpha=0.15, edgecolor='none')
    ax.add_patch(highlight)


def generate_interactive_poster(
    layers,
    palette_name,
    shape_type,
    min_radius,
    max_radius,
    wobble,
    alpha,
    shadow_offset,
    perspective,
    seed
    ):
    """
    This is the main function called by `interact`.
    It generates one poster based on the widget values.
    """

    # Set seed (Week 2)
    np.random.seed(seed)
    random.seed(seed)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 13))

    # Get palette and background (Weeks 2, 5)
    palette_info = COLOR_PALETTES_DICT[palette_name]
    palette = palette_info['palette']
    fig.patch.set_facecolor(palette_info['bg'])

    # Configure axes
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis('off')
    ax.set_aspect('equal') # Ensure shapes aren't stretched

    print("\n--- Redrawing Canvas ---")
    print(f"Seed: {seed}, Shape: {shape_type}, Palette: {palette_name}")

    for i in range(layers):
        # Position (Week 2)
        center_x = int(random.gauss(WIDTH / 2, WIDTH / 3.5))
        center_y = int(random.gauss(HEIGHT / 2, HEIGHT / 3.5))

        # Perspective Sizing (Week 4)
        scale_factor = max(0.1, 1.0 - (center_y / HEIGHT) * perspective)

        color = random.choice(palette)

        # 1. Get points for the selected shape
        if shape_type == 'Blob':
            min_rad = min_radius * scale_factor
            max_rad = max_radius * scale_factor
            points = create_smooth_blob(center_x, center_y, min_rad, max_rad, wobble)
        elif shape_type == 'Heart':
            scale = (min_radius / 70) * scale_factor * 0.7
            points = create_heart_points(center_x, center_y, scale)
        elif shape_type == 'Star':
            scale = (min_radius / 70) * scale_factor * 2.5
            points = create_star_points(center_x, center_y, scale)

        # Draw the 3D shape
        draw_3d_shape(ax, points, color, alpha, shadow_offset)

    # NOTE: To save, right-click the generated image in your notebook.
    plt.show()


# --- 4. UI Setup and Execution (ipywidgets) ---

# This is the main execution call.
# It links the function above to all the widgets.
interact(
    generate_interactive_poster,

    # --- Define all the widgets ---
    layers=widgets.IntSlider(min=1, max=50, step=1, value=15, description='Layers:'),
    palette_name=widgets.Dropdown(options=PALETTE_NAMES, value=PALETTE_NAMES[0], description='Palette:'),
    shape_type=widgets.RadioButtons(options=['Blob', 'Heart', 'Star'], value='Blob', description='Shape:'),
    min_radius=widgets.IntSlider(min=10, max=200, step=5, value=70, description='Min Radius:'),
    max_radius=widgets.IntSlider(min=50, max=500, step=5, value=250, description='Max Radius:'),
    wobble=widgets.IntSlider(min=4, max=20, step=1, value=6, description='Wobble (Blob):'),
    alpha=widgets.FloatSlider(min=0.1, max=1.0, step=0.05, value=0.7, description='Alpha:'),
    shadow_offset=widgets.IntSlider(min=0, max=50, step=1, value=10, description='Shadow:'),
    perspective=widgets.FloatSlider(min=0.0, max=1.0, step=0.05, value=0.6, description='Perspective:'),
    seed=widgets.IntSlider(min=0, max=10000, step=1, value=random.randint(0, 10000), description='Seed:')
);

