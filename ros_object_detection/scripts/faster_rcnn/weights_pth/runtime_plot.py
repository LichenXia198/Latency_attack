import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# # Step 1: Read the CSV file
# file_path = 'inference_log_post.csv'  # Replace with your file path
# data = pd.read_csv(file_path)

# # Step 2: Extract the columns
# box_num = data['Number of Proposals']
# run_time = data['Post-Processing Time (seconds)']

# # Step 3: Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(box_num, run_time, marker='o', linestyle='-', color='b')
# plt.xlabel('Box Number')
# plt.ylabel('Run Time')
# plt.title('Run Time vs Box Number')
# plt.grid(True)

# # Step 4: Save the plot as an image
# plt.savefig('run_time_vs_box_num.png', dpi=300)  # Saves the plot as a PNG file with 300 DPI

# # Optional: If you want to close the plot to avoid displaying it
# plt.close()
# List of file paths
# file_paths = [
#     'experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_second_1.csv',
#     'experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_second_2.csv',
#     'experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_second_3.csv',
#     'experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_second_4.csv',
#     'experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_second_5.csv',
#     'experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_second_6.csv',
#     'experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_second_7.csv',
#     'experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_second_8.csv',
#     'experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_second_9.csv'
# ]
file_paths = [
    'inference_log_post_score=0.5_draw_1.csv',
    'inference_log_post_score=0.5_draw_2.csv',
    'inference_log_post_score=0.5_draw_3.csv',
    'inference_log_post_score=0.5_draw_4.csv',
    'inference_log_post_score=0.5_draw_5.csv',
    'inference_log_post_score=0.5_draw_6.csv',
    'inference_log_post_score=0.5_draw_7.csv',
    'inference_log_post_score=0.5_draw_8.csv',
    'inference_log_post_score=0.5_draw_9.csv'
]

# Step 1: Read all CSV files into a list of DataFrames
dfs = [pd.read_csv(file_path) for file_path in file_paths]

# Step 2: Combine all DataFrames by taking the mean across them
combined_df = pd.concat(dfs).groupby(level=0).mean().iloc[:, :]


# Step 3: Normalize the columns
normalized_df = (combined_df - combined_df.min()) / (combined_df.max() - combined_df.min())
# normalized_df = combined_df

# Step 4: Calculate the correlation coefficients
correlation_coefficient = normalized_df.iloc[:, 0].corr(normalized_df.iloc[:, 1])
correlation_coefficient1 = normalized_df.iloc[:, 0].corr(normalized_df.iloc[:, 2])
correlation_coefficient2 = normalized_df.iloc[:, 0].corr(normalized_df.iloc[:, 3])

# Display correlation values
correlation_values = [correlation_coefficient, correlation_coefficient1, correlation_coefficient2]
labels = ['Proposals vs Post-Processing + Draw Time', 'Proposals vs Post-Processing Time', 'Proposals vs Draw Time']

# Step 4: Plot the normalized columns as line charts in the same figure
plt.figure(figsize=(10, 6))
for i, (label, value) in enumerate(zip(labels, correlation_values)):
    plt.text(0.95, 0.15 - i * 0.05, f"{label}: {value:.4f}", fontsize=10, ha='right', va='center', transform=plt.gca().transAxes)

# Plot each column
plt.plot(normalized_df.iloc[:, 0], label='Number of Proposals', color='blue')
# plt.plot(normalized_df.iloc[:, 1], label='Post-Processing + Draw Time', color='orange')
# plt.plot(normalized_df.iloc[:, 2], label='Post-Processing Time', color='red')
plt.plot(normalized_df.iloc[:, 3], label='Draw Time', color='green')
# Load the CSV file into a DataFrame
# df = pd.read_csv('inference_log_post_score=0.5_maskrcnn.csv')
# df = pd.read_csv('experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_1.csv')
# df = pd.read_csv('experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_2.csv')
# df = pd.read_csv('experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_3.csv')
# df = pd.read_csv('experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_4.csv')

# # Define a function to remove outliers based on IQR
# def remove_outliers(series):
#     Q1 = series.quantile(0.25)
#     Q3 = series.quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     return series[(series >= lower_bound) & (series <= upper_bound)]

# # Apply average filter (moving average) with a window size of 3 to both columns
# # window_size = 3
# # smoothed_col1 = df.iloc[:, 0].rolling(window=window_size, center=True).mean()
# # smoothed_col2 = df.iloc[:, 1].rolling(window=window_size, center=True).mean()

# # # Remove outliers from the second column
# # col2_without_outliers = remove_outliers(smoothed_col2)

# # # Normalize the first column after smoothing
# # normalized_col1 = (smoothed_col1 - smoothed_col1.min()) / (smoothed_col1.max() - smoothed_col1.min())

# # # Normalize the second column after smoothing and removing outliers
# # normalized_col2 = (smoothed_col2 - smoothed_col2.min()) / (smoothed_col2.max() - smoothed_col2.min())

# # Normalize the first two columns
# normalized_col1 = (df.iloc[:, 0] - df.iloc[:, 0].min()) / (df.iloc[:, 0].max() - df.iloc[:, 0].min())

# normalized_col2 = (df.iloc[:, 1] - df.iloc[:, 1].min()) / (df.iloc[:, 1].max() - df.iloc[:, 1].min())

# normalized_col3 = (df.iloc[:, 2] - df.iloc[:, 2].min()) / (df.iloc[:, 2].max() - df.iloc[:, 2].min())

# normalized_col4 = (df.iloc[:, 3] - df.iloc[:, 3].min()) / (df.iloc[:, 3].max() - df.iloc[:, 3].min())

# # # Remove outliers from the second column
# # col2_without_outliers = remove_outliers(df.iloc[:, 1])

# # # Normalize the second column after removing outliers
# # normalized_col2 = (col2_without_outliers - col2_without_outliers.min()) / (col2_without_outliers.max() - col2_without_outliers.min())

# # Plot the normalized columns as line charts in the same figure
# plt.figure(figsize=(10, 6))

# # Plot the first normalized column
# plt.plot(normalized_col1, label='num of proposals', color='blue')

# # Plot the second normalized column
# plt.plot(normalized_col2, label='post-processing time', color='orange')

# # Plot the first normalized column
# plt.plot(normalized_col3, label='first stage time', color='red')

# # Plot the second normalized column
# plt.plot(normalized_col4, label='full time', color='green')

# Add titles and labels
plt.title('Normalized Line Charts for First Two Columns')
plt.xlabel('Sample Index')
plt.ylabel('Normalized Value')
plt.legend()

# plt.savefig('run_time_vs_box_num_score=0.5_outlier_removed_maskrcnn.png', dpi=300)
# plt.savefig('run_time_vs_box_num_score=0.5_average_start_from_10.png', dpi=300)
plt.savefig('run_time_vs_box_num_score=0.5_average_start_draw_single.png', dpi=300)
# plt.savefig('experiment_fitness=proposals_score=0.5_limited=100000_outlier_removed.png', dpi=300)
# plt.savefig('experiment_fitness=proposals_score=0.5_limited=100000_full_metrics_average_start_from_10_second.png', dpi=300)
# plt.savefig('experiment_fitness=proposals_score=0.5_limited=100000_inference_times_unnormalized_average_start_from_10.png', dpi=300)
plt.close()


# # Calculate the correlation coefficient
# correlation_coefficient = normalized_df.iloc[:, 0].corr(normalized_df.iloc[:, 1])
# correlation_coefficient1 = normalized_df.iloc[:, 0].corr(normalized_df.iloc[:, 2])
# correlation_coefficient2 = normalized_df.iloc[:, 0].corr(normalized_df.iloc[:, 3])
# # Print the correlation coefficient
# print(f"Correlation Coefficient between the two columns: {correlation_coefficient:.4f}")
# print(f"Correlation Coefficient between the two columns: {correlation_coefficient1:.4f}")
# print(f"Correlation Coefficient between the two columns: {correlation_coefficient2:.4f}")