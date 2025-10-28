# -*- coding: utf-8 -*-
"""
Streamlit Generative Poster Creator

Final version with restored seed range and expanded shape list.
"""

import streamlit as st
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import splprep, splev
from io import BytesIO 

# Set wide mode for a better layout
st.set_page_config(layout="wide", page_title="Generative Poster Creator")


# --- 1. Palette and Data Setup ---
PALETTE_FILE = "palette.csv"
WIDTH, HEIGHT = 1200, 1600 # Define global constants for poster size

# Default palettes (feel free to add more here!)
COLOR_PALETTES_DICT = {
    "Pastel": {"palette": ["#FFADAD", "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF", "#A0C4FF", "#BDB2FF"], "bg": "#F0EAD6"},
    "Vivid": {"palette": ["#FF595E", "#FFCA3A", "#8AC926", "#1982C4", "#6A4C93", "#F58553"], "bg": "#FFFFFF"},
    "Monochrome": {"palette": ["#222222", "#555555", "#888888", "#BBBBBB", "#EEEEEE"], "bg": "#D1D1D1"},
    "Ocean": {"palette": ["#0077B6", "#0096C7", "#00B4D8", "#48CAE4", "#90E0EF"], "bg": "#F0F8FF"}, # New Example
    "Sunset": {"palette": ["#E76F51", "#F4A261", "#E9C46A", "#2A9D8F", "#264653"], "bg": "#FFFBEF"} # New Example
}

# Use Streamlit's cache to only run this once
@st.cache_data
def load_csv_palette():
    """Loads palettes from the CSV file and adds to the dictionary."""
    if not os.path.exists(PALETTE_FILE):
        df_init = pd.DataFrame([
            {"name": "csv_sky", "r": 0.4, "g": 0.7, "b": 1.0},
            {"name": "csv_sun", "r": 1.0, "g": 0.8, "b": 0.2},
            {"name": "csv_forest", "r": 0.2, "g": 0.6, "b": 0.3}
        ])
        df_init.to_csv(PALETTE_FILE, index=False)
        st.sidebar.warning(f"'{PALETTE_FILE}' not found. An example was created (commit to repo for cloud use).")

    try:
        df = pd.read_csv(PALETTE_FILE)
        colors = []
        for row in df.itertuples():
            r, g, b = int(row.r * 255), int(row.g * 255), int(row.b * 255)
            r, g, b = max(0, min(r, 255)), max(0, min(g, 255)), max(0, min(b, 255))
            colors.append(f'#{r:02x}{g:02x}{b:02x}')

        if colors:
            COLOR_PALETTES_DICT["CSV_Palette"] = {"palette": colors, "bg": "#FAFAFA"}
            st.sidebar.success(f"'CSV_Palette' loaded with {len(colors)} colors.")
        else:
            st.sidebar.info("CSV palette was empty.")
    except Exception as e:
        st.sidebar.error(f"Error loading '{PALETTE_FILE}': {e}")

# Load the CSV palette and get the names
load_csv_palette()
PALETTE_NAMES = list(COLOR_PALETTES_DICT.keys())


# --- 2. Shape Functions ---

def create_smooth_blob(center_x, center_y, min_radius, max_radius, num_points):
    """Creates points for a smooth blob."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = np.random.uniform(min_radius, max_radius, num_points)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    tck, u = splprep([x, y], s=0, per=True)
    unew = np.linspace(u.min(), u.max(), 1000)
    x_smooth, y_smooth = splev(unew, tck, der=0)
    return np.array([x_smooth + center_x, y_smooth + center_y]).T

def create_heart_points(center_x, center_y, scale):
    """Creates points for a heart shape."""
    t = np.linspace(0, 2 * np.pi, 100)
    x = scale * 16 * np.sin(t)**3
    y = scale * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    return np.array([x + center_x, y + center_y]).T

def create_star_points(center_x, center_y, scale, num_points=5):
    """Creates points for a star shape."""
    angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
    outer_radius = scale * 10
    inner_radius = scale * 4
    radii = np.array([outer_radius if i % 2 == 0 else inner_radius for i in range(num_points * 2)])
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.array([x + center_x, y + center_y]).T

def create_triangle_points(center_x, center_y, scale):
    """Creates points for a simple triangle shape."""
    height = scale * 15
    side = height / (np.sqrt(3) / 2)
    x = [0, side / 2, -side / 2, 0]
    y = [height / 2, -height / 2, -height / 2, height / 2]
    return np.array([np.array(x) + center_x, np.array(y) + center_y]).T

def create_circle_points(center_x, center_y, scale, num_points=100):
    """Creates points for a smooth circle shape."""
    radius = scale * 10
    t = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return np.array([x + center_x, y + center_y]).T

def create_rectangle_points(center_x, center_y, scale):
    """Creates points for a rectangle shape (aspect ratio 3:4 for poster look)."""
    width = scale * 20
    height = scale * 26.6
    x = [-width/2, width/2, width/2, -width/2, -width/2]
    y = [height/2, height/2, -height/2, -height/2, height/2]
    return np.array([np.array(x) + center_x, np.array(y) + center_y]).T


# --- 3. Drawing Logic ---

def draw_3d_shape(ax, points, color, alpha, shadow_offset):
    """Draws any selected shape with shadow and highlight."""
    # Draw the Shadow
    shadow_points = points.copy()
    shadow_points[:, 0] += shadow_offset
    shadow_points[:, 1] -= shadow_offset
    shadow = patches.Polygon(shadow_points, facecolor='black', alpha=0.2, edgecolor='none')
    ax.add_patch(shadow)

    # Draw the Main Shape
    polygon = patches.Polygon(points, facecolor=color, alpha=alpha, edgecolor='none')
    ax.add_patch(polygon)

    # Draw the Highlight
    highlight_points = points.copy()
    highlight_points[:, 0] -= shadow_offset * 0.2
    highlight_points[:, 1] += shadow_offset * 0.2
    center = np.mean(points, axis=0)
    highlight_points = (highlight_points - center) * 0.9 + center

    highlight = patches.Polygon(highlight_points, facecolor='white', alpha=0.15, edgecolor='none')
    ax.add_patch(highlight)


def generate_poster(
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
    Generates one poster based on the widget values.
    Returns the matplotlib figure.
    """
    # Set seed
    np.random.seed(seed)
    random.seed(seed)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 13))

    # Get palette and background
    palette_info = COLOR_PALETTES_DICT[palette_name]
    palette = palette_info['palette']
    fig.patch.set_facecolor(palette_info['bg'])

    # Configure axes
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis('off')
    ax.set_aspect('equal')

    for i in range(layers):
        # Position
        center_x = int(random.gauss(WIDTH / 2, WIDTH / 3.5))
        center_y = int(random.gauss(HEIGHT / 2, HEIGHT / 3.5))

        # Perspective Sizing
        scale_factor = max(0.1, 1.0 - (center_y / HEIGHT) * perspective)

        color = random.choice(palette)
        
        # Scaling factor based on min_radius for all non-blob shapes
        base_scale = (min_radius / 70) * scale_factor

        # 1. Get points for the selected shape
        if shape_type == 'Blob':
            min_rad = min_radius * scale_factor
            max_rad = max_radius * scale_factor
            points = create_smooth_blob(center_x, center_y, min_rad, max_rad, wobble)
        elif shape_type == 'Heart':
            scale = base_scale * 0.7
            points = create_heart_points(center_x, center_y, scale)
        elif shape_type == 'Star':
            scale = base_scale * 2.5
            points = create_star_points(center_x, center_y, scale)
        elif shape_type == 'Triangle':
            scale = base_scale * 2.5
            points = create_triangle_points(center_x, center_y, scale)
        elif shape_type == 'Circle':
            scale = base_scale * 2.5
            points = create_circle_points(center_x, center_y, scale)
        elif shape_type == 'Rectangle':
            scale = base_scale * 2.0
            points = create_rectangle_points(center_x, center_y, scale)

        # Draw the 3D shape
        draw_3d_shape(ax, points, color, alpha, shadow_offset)

    return fig


# --- Helper Function for Download ---
def fig_to_bytes(fig):
    """Converts Matplotlib figure to PNG bytes for download and closes the figure."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig) 
    return buf.getvalue()


# --- 4. Streamlit UI Setup and Execution ---

st.title("Generative Poster Creator 🎨")
st.markdown("Use the controls in the sidebar to create a unique poster!")

# --- Widget Definitions in the Sidebar ---
with st.sidebar:
    st.header("Poster Controls")

    # Shape and Radius
    shape_type = st.radio(
        "Shape:", 
        ['Blob', 'Heart', 'Star', 'Rectangle', 'Triangle', 'Circle'],
        index=0
    )
    min_radius = st.slider("Min Radius/Base Scale:", 10, 200, 70, 5)

    # Blob-specific control visibility
    if shape_type == 'Blob':
        max_radius = st.slider("Max Radius (Blob):", 50, 1000, 250, 5)
        wobble = st.slider("Wobble Points (Blob):", 4, 20, 6, 1)
        st.info("Min Radius and Max Radius control the range of sizes.")
    else:
        # Default/ignored values for other shapes
        max_radius = 500
        wobble = 6
        st.info("For other shapes, 'Min Radius/Base Scale' controls the size.")

    # Appearance
    st.subheader("Appearance & Effects")
    layers = st.slider("Layers (Shape Count):", 1, 50, 15, 1)
    palette_name = st.selectbox("Palette:", PALETTE_NAMES, index=0)
    alpha = st.slider("Alpha (Transparency):", 0.1, 1.0, 0.7, 0.05)
    shadow_offset = st.slider("Shadow Offset:", 0, 50, 10, 1)
    perspective = st.slider("Perspective (Vertical Scale):", 0.0, 1.0, 0.6, 0.05)

    # Seed (Restored to original wider range)
    st.subheader("Randomness")
    default_seed = random.randint(0, 50000)
    if 'seed' not in st.session_state:
        st.session_state.seed = default_seed
    
    # MAXIMUM SEED VALUE RESTORED TO 10000
    seed = st.number_input("Seed:", 0, 50000, st.session_state.seed, 1, key='seed_input')
    st.session_state.seed = seed

# Generate the poster
poster_fig = generate_poster(
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
)

# st.pyplot displays the matplotlib figure
st.pyplot(poster_fig)

# Add download button
st.download_button(
    label="Download Poster (PNG)",
    data=fig_to_bytes(poster_fig), 
    file_name="generative_poster.png",
    mime="image/png"
)
