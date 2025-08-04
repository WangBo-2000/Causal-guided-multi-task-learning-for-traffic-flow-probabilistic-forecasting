import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString, Point
from collections import Counter
import numpy as np

gdf = pd.read_csv(r"./data/gdf.csv")
gdf["Down_Node"] = gdf["Down_Node"].astype(str)

"""
data = pd.read_csv(r"loop.csv")
data = data[['DET_ID', ' ROAD_ID', 'FTIME', 'TTIME', 'COUNT', 'REG_COUNT', 'LAR_COUNT', 'TURN']]
data = data.drop_duplicates()
print(data.columns)

# Split the ROAD_ID column based on '_' to create Up_Node and Down_Node columns
data[['Up_Node', 'Down_Node']] = data[' ROAD_ID'].str.split('_', expand=True)

# Convert Min and Max columns to datetime type
data['Min'] = pd.to_datetime(data['FTIME'])
data['Max'] = pd.to_datetime(data['TTIME'])
data = data[~((data['Min'].dt.hour > 20) | (data['Min'].dt.hour < 9))]

# Remove data based on intersections
Downs = list(set(list(gdf["Down_Node"])))
print(len(Downs))
data = data[data["Down_Node"].isin(Downs)].reset_index(drop=True)

# Sum the traffic volume for each road segment
data['Volume'] = data.groupby(['Down_Node', 'FTIME', "TTIME"])['COUNT'].transform('sum')
data = data.drop_duplicates(subset=["Down_Node", 'FTIME', "TTIME"]).reset_index(drop=True)
print(len(data))

data = data[['Down_Node', 'FTIME', 'TTIME', 'Volume']]
data = data.sort_values(by=["Down_Node", "FTIME"]).reset_index(drop=True)
data.to_csv(r"flow.csv", index=False)
"""

data = pd.read_csv(r"flow.csv")
print(len(data))

# Convert the FTIME column to datetime type
data['FTIME'] = pd.to_datetime(data['FTIME'])

# Group by Down_Node and date
grouped = data.groupby(['Down_Node', data['FTIME'].dt.date])

result = []
for (down_node, date), group in grouped:
    # Create a complete time index with 5-minute intervals from 9:00 AM to 8:55 PM
    full_time_index = pd.date_range(start=f'{date} 09:00:00', end=f'{date} 20:55:00', freq='5T')
    # Set FTIME as the index
    group = group.set_index('FTIME')

    # print(group)
    if len(group) == 144:
        result.append(group)
    else:
        # print(group)
        # print(len(group))
        # Reindex, missing values will be automatically filled with NaN
        group = group.reindex(full_time_index)

        # Resample with a 5-minute frequency
        resampled = group['Volume'].resample('5T')

        # Perform linear interpolation
        interpolated = resampled.interpolate(method='linear')

        # Round up to the nearest integer
        interpolated = np.ceil(interpolated)

        # Reset index and add Down_Node and date columns
        interpolated_df = interpolated.reset_index().rename(columns={'index': 'FTIME'})
        interpolated_df['Down_Node'] = down_node
        interpolated_df['Date'] = date

        # interpolated_df['TTIME'] = interpolated_df['FTIME'] + pd.Timedelta(minutes=5)

        interpolated_df["Volume"] = interpolated_df["Volume"].fillna(0)

        print(interpolated_df)
        print("-------------------------")
        result.append(interpolated_df)

# Combine all results
final_result = pd.concat(result, ignore_index=True)

# Save the result as a new CSV file
new_file_path = 'flow_interpolated.csv'
final_result.to_csv(new_file_path)


# Convert FTIME and TTIME columns to datetime type
# final_result['FTIME'] = pd.to_datetime(final_result['FTIME'])
final_result['TTIME'] = pd.to_datetime(final_result['TTIME'])

# Group by the Down_Node column and calculate the minimum start time and maximum end time for each node
result = final_result.groupby('Down_Node').agg(
    最小记录时间=('TTIME', 'min'),
    最大记录时间=('TTIME', 'max')
).reset_index()

# Save the result as a CSV file
csv_path = './data/flow_min_max_time.csv'
result.to_csv(csv_path, index=False, encoding="gbk")