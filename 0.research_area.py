import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import transbigdata as tbd
from collections import Counter
import numpy as np
import matplotlib.ticker as ticker
from shapely import wkt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches

# Set Chinese font to SimSun
plt.rcParams['font.family'] = 'SimSun'
# Add Times New Roman font for English and numbers
fm.fontManager.addfont(fm.findfont('Times New Roman'))
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
# Solve the problem of negative sign display
plt.rcParams['axes.unicode_minus'] = False
# Globally set font size to 11
plt.rcParams.update({'font.size': 11})


def correct_linestring_format(linestring_str):
    start_index = linestring_str.find('(') + 1
    end_index = linestring_str.rfind(')')
    coordinates = linestring_str[start_index:end_index]
    corrected_coordinates = coordinates.replace(';', ', ')
    return f'LINESTRING({corrected_coordinates})'


# Read road network file
df = pd.read_csv('./data/road_network_segment_level.csv')
# Apply function to geom column
df['geom'] = df['geom'].apply(correct_linestring_format)
# Split cid column based on '_' to create Up_Node and Down_Node columns
df[['Up_Node', 'Down_Node']] = df['cid'].str.split('_', expand=True)
# Convert geom column to GeoSeries
df['geom'] = gpd.GeoSeries.from_wkt(df['geom'])
# Create GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geom')

# Read original traffic file
data = pd.read_csv(r"./data/loop.csv")
data = data[['DET_ID', ' ROAD_ID', 'FTIME', 'TTIME', 'COUNT', 'REG_COUNT', 'LAR_COUNT', 'TURN']]
data = data.drop_duplicates()
print(data.columns)

# Convert Min and Max columns to datetime type
data['Min'] = pd.to_datetime(data['FTIME'])
data['Max'] = pd.to_datetime(data['TTIME'])
data = data[~((data['Min'].dt.hour > 20) | (data['Min'].dt.hour < 9))]
print(data.columns)

# Traffic volume # Split  ROAD_ID column based on '_' to create Up_Node and Down_Node columns
data[['Up_Node', 'Down_Node']] = data[' ROAD_ID'].str.split('_', expand=True)
data["COUNT"] = 1
data['Volume'] = data.groupby(['Down_Node', 'FTIME', "TTIME"])['COUNT'].transform('sum')
data = data.drop_duplicates(subset=["Down_Node", 'FTIME', "TTIME"]).reset_index(drop=True)

data['Down_Node'] = data['Down_Node'].astype(str)
DownNodeList = list(set(list(data["Down_Node"])))
# print(DownNodeList)

# Check if there is data
F = dict(Counter(data["Down_Node"]))
F["5005"] = 4200
F["5017"] = 4200

gdf['is_in'] = (gdf['Down_Node'].isin(DownNodeList) & gdf['Up_Node'].isin(DownNodeList)).astype(int)
# gdf = gdf[(gdf["is_in"] == 1)].reset_index(drop=True)
gdf['Count'] = gdf['Down_Node'].map(F)  # Data volume  # Only filtered for DownNode
gdf = gdf[(gdf["Count"] > 4000)].reset_index(drop=True)
gdf['Count1'] = gdf['Up_Node'].map(F)  # Data volume  # Only filtered for DownNode
gdf = gdf[(gdf["Count1"] > 4000)].reset_index(drop=True)

Unused = [9114, 9115, 9062, 9063, 9102, 9101, 9113, 9122, 9111, 9110, 9100, 9098,
          9099, 9002, 9112, 9029, 9030, 9021, 9022, 9068, 9031, 9034, 9032, 9035,
          5429, 9036, 9072, 8368, 4929, 4947, 4986, 9042, 9043, 9040, 9041, 8461,
          9079, 9078, 9051, 9052, 4393, 8347, 4718, 9006, 9007, 9009, 4609, 4530,
          4520, 4502, 4534, 4582, 4474, 4448, 4423, 5217, 5247, 9054, 4415, 9014,
          5052, 8356, 8357, 9165, 8371, 9149, 9011, 4943]
gdf = gdf[~(gdf['Down_Node'].isin([str(Node) for Node in Unused]) | gdf['Up_Node'].isin([str(Node) for Node in Unused]))].reset_index(drop=True)

gdf = gdf[~(gdf['Down_Node'].isin(['5026']) & gdf['Up_Node'].isin(['8359']))].reset_index(drop=True)
gdf = gdf[~(gdf['Down_Node'].isin(['8359']) & gdf['Up_Node'].isin(['5026']))].reset_index(drop=True)

# 4. Define bidirectional connection data from node 5005 to 4973
new_data = [
    {
        'cid': '5005_4973',
        'nlane': 0,
        'turn': 'B',
        'dnroad': '4973_5005',
        'geom': wkt.loads('LINESTRING(118.772823689 30.953919491, 118.763825315 30.952785773)'),
        'len': 10497.121,
        'Up_Node': '5005',
        'Down_Node': '4973'
    },
    {
        'cid': '4973_5005',
        'nlane': 0,
        'turn': 'B',
        'dnroad': '5005_4973',
        'geom': wkt.loads('LINESTRING(118.763825315 30.952785773, 118.772823689 30.953919491)'),
        'len': 10497.121,
        'Up_Node': '4973',
        'Down_Node': '5005'
    }
]
new_df = gpd.GeoDataFrame(new_data, geometry='geom', crs='EPSG:4326')
print(new_df)

gdf = pd.concat([gdf, new_df])
print(gdf.columns)
# gdf = gdf.drop(columns='geometry')
gdf.to_csv(r"./data/gdf.csv", index=False)
# print(gdf)

# Draw the figure
fig, ax = plt.subplots(figsize=(6.3, 4))

# Plot road network
gdf.plot(ax=ax,
         linewidth=0.8,  # Set line width to 2
         color='black'   # Set line color to red
         )

# Calculate map boundaries and center point
bounds = gdf.total_bounds  # Get the bounds of the GeoDataFrame
min_lon, min_lat, max_lon, max_lat = bounds
# Define display range
bounds = [min_lon, min_lat, max_lon, max_lat+0.001]
# Add map base
tbd.plot_map(plt, bounds, zoom=16, style=1)

gdf = gdf.drop_duplicates(subset=["Down_Node"]).reset_index(drop=True)
print(len(gdf))

index = 0
# Mark and label with red dots at points corresponding to Down_Node
for idx, row in gdf.iterrows():
    if row['is_in'] == 1 and int(row['Down_Node']) not in Unused and int(row['Up_Node']) not in Unused:
        if int(row['Down_Node']) == 4724:
            point = row['geom'].coords[-1]  # Take the last point of the line as the intersection
            ax.plot(point[0], point[1], marker='o', color='blue', markersize=2)  # Draw red dot
            index += 1
        else:
            point = row['geom'].coords[-1]  # Take the last point of the line as the intersection
            ax.plot(point[0], point[1], marker='o', color='blue', markersize=2)  # Draw red dot
            index += 1
    else:
        pass
print(index)

plt.xlabel('Longitude', fontweight='bold')
plt.ylabel('Latitude', fontweight='bold')

ax.spines['left'].set_linewidth(1)  # Example thickness 1, adjustable
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)

ax.tick_params(width=1, length=1)  # Tick line width

# Bold tick values
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight('bold')

# Set x-axis tick positions and labels
x_ticks = np.arange(118.72, 118.78, 0.02)  # From 0 to 10, interval 2
plt.xticks(x_ticks)

# Set x-axis formatter
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))  # Disable scientific notation

# Add direction indicator
arrow = mpatches.FancyArrowPatch(
    (0.1, 0.8),
    (0.1, 0.9),
    arrowstyle='fancy',
    mutation_scale=20,
    facecolor='white',
    edgecolor='black',
    linewidth=1.5,
    transform=ax.transAxes
)
ax.add_patch(arrow)
ax.text(0.1, 0.93, 'N', ha='center', va='center', fontweight='bold', color='black', transform=ax.transAxes)
"""
# Assuming the latitude of the current map, calculate the meters per degree of latitude and longitude (example, need to adjust according to reality)
lat = 31  # Assumed latitude
lon_per_degree = 111319 * np.cos(np.radians(lat))  # Meters per degree of longitude
lat_per_degree = 110574  # Meters per degree of latitude

# Map range
min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
# print(min_lon, min_lat, max_lon, max_lat)

# Scale setting
scale_total_m = 2 * 1000  # 3 kilometers, converted to meters
scale_step_m = 1 * 1000  # 1 kilometer per segment
scale_lon = scale_total_m / lon_per_degree  # Convert to longitude span

# Draw scale position (example in lower left corner)
x_start = min_lon + 0.05 * (max_lon - min_lon)
y_start = min_lat + 0.05 * (max_lat - min_lat)

# Draw main line segment
ax.plot([x_start, x_start + scale_lon], [y_start, y_start], color='black', linewidth=1.5)

# Draw ticks
for i in range(0, int(scale_total_m / scale_step_m) + 1):
    x_tick = x_start + i * (scale_lon / (scale_total_m / scale_step_m))
    ax.vlines(x_tick, y_start - 0.0004, y_start + 0.0004, color='black', linewidth=1.5)

# Add labels
labels = [0, 1, 2]
for i, label in enumerate(labels):
    x_label = x_start + i * (scale_lon / (scale_total_m / scale_step_m))
    ax.text(x_label, y_start - 0.0006, f'{label}km', ha='center', va='top', fontsize=10)
"""
# Display the figure
plt.savefig(r"research_area.jpg", dpi=1000)