from random import randint

import bpy
import pandas as pd

df = pd.read_csv(
    r"/home/rmr62/Projects/LlmEmbeddingXrVizualization/data/reduced_embeddings_umap.csv"
)


# Clear existing mesh objects in the scene
bpy.ops.object.select_all(action="DESELECT")
bpy.ops.object.select_by_type(type="MESH")
bpy.ops.object.delete()


# Generate unique colors for each label
unique_sectors = df["sector"].unique()
sector_colors = {
    label: (randint(0, 255) / 255, randint(0, 255) / 255, randint(0, 255) / 255, 1)
    for label in unique_sectors
}


# Function to create a sphere at a given location
def create_sphere(location, color, radius=0.1):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    obj = bpy.context.object
    mat = bpy.data.materials.new(name="Material")
    mat.diffuse_color = color
    obj.data.materials.append(mat)
    return obj


# Plotting the data
for index, row in df.iterrows():
    point_location = (row[0], row[1], row[2])

    sector = row["sector"]
    label = row["Label"]
    color = sector_colors.get(
        sector, (1, 1, 1, 1)
    )  # Default to white if sector not found

    sphere = create_sphere(point_location, color)

    location = point_location
    text_obj = bpy.ops.object.text_add(
        location=(location[0] + 0.1, location[1], location[2])
    )
    text_obj = bpy.context.object
    text_obj.data.body = label
    text_obj.scale = (0.2, 0.2, 0.2)  # Adjust the scale of the text if necessary
    text_obj.rotation_euler = (
        1.5708,
        0,
        0,
    )  # Rotate the text 90 degrees around the X-axis
    bpy.ops.object.convert(target="MESH")

# Create a legend
legend_x = max(df["0"]) + 2  # Position legend to the right of the chart
legend_y = max(df["1"])
legend_z = 0

# Define materials dictionary
materials = {sector: bpy.data.materials.new(name=sector) for sector in unique_sectors}
for sector, color in sector_colors.items():
    materials[sector].diffuse_color = color

for i, (category, material) in enumerate(materials.items()):
    # Create a small sphere for the legend
    legend_sphere_size = 0.2
    legend_sphere_location = (legend_x, legend_y - i * 1, legend_z)
    # create_sphere(legend_sphere_location, material.diffuse_color, legend_sphere_size)

    # Create a text object for the legend
    text_obj = bpy.ops.object.text_add(
        location=(legend_x + 0.5, legend_y - i * 1, legend_z)
    )
    text_obj = bpy.context.object
    text_obj.data.body = category
    # Set the text color to match the sector color
    text_material = bpy.data.materials.new(name=f"{category}_Text")
    text_material.diffuse_color = sector_colors[category]
    if text_obj.data.materials:
        text_obj.data.materials[0] = text_material
    else:
        text_obj.data.materials.append(text_material)
    bpy.ops.object.convert(target="MESH")

# Export to DAE
bpy.ops.wm.collada_export(filepath="output.dae")
