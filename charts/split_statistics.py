import pandas as pd

def statistics(file_path):
    df = pd.read_csv(file_path) 
    df = df.rename(columns={'difference': 'duration'})

    # count the number of instances in the split column
    count_split = df['split'].value_counts().reset_index()
    print("\nDatapoints in each split:")
    print(count_split)
    
    # Calculate the sum of 'duration' based on 'split'
    sum_duration_split = df.groupby('split')['duration'].sum().reset_index()
    # change duration to hours
    sum_duration_split['duration'] = sum_duration_split['duration'].apply(lambda x: round(x / 3600000, 2))
    total_duration = sum_duration_split['duration'].sum()

    # calculate the sum of 'duration' based on WAB_AQ_category
    sum_duration_by_category = df.groupby(['WAB_AQ_category'])['duration'].sum().reset_index()
    # change duration to hours
    sum_duration_by_category['duration'] = sum_duration_by_category['duration'].apply(lambda x: round(x / 3600000, 2))
    total_duration_category = sum_duration_by_category['duration'].sum()

    # calculate duration percentages for each category
    sum_duration_by_category['percentage'] = (sum_duration_by_category['duration'] / total_duration_category).apply(lambda x: round(x * 100, 2))
    print("\nDuration of audios (hour) based on 'WAB_AQ_category':")
    print(sum_duration_by_category)
    
    # Calculate duration percentages for each split
    sum_duration_split['percentage'] = (sum_duration_split['duration'] / total_duration).apply(lambda x: round(x * 100, 2))
    print("\nDuration of audios (hour) in each split:")
    print(sum_duration_split)

    # Calculate the sum of 'duration' based on 'WAB_AQ_category' for each split
    sum_duration_by_category = df.groupby(['split', 'WAB_AQ_category'])['duration'].sum().reset_index()
    
    # Calculate percentages for each category within each split
    sum_duration_by_category = sum_duration_by_category.merge(
        sum_duration_split[['split', 'duration']], 
        on='split', 
        suffixes=('', '_total')
    )
    sum_duration_by_category['duration']=sum_duration_by_category['duration'].apply(lambda x: round(x / 3600000, 2))
    sum_duration_by_category['percentage'] = (sum_duration_by_category['duration'] / sum_duration_by_category['duration_total']).apply(lambda x: round(x * 100, 2))

    print("\nDuration of audios (hour) based on 'WAB_AQ_category' in each split:")
    print(sum_duration_by_category[['split', 'WAB_AQ_category', 'duration', 'percentage']])
    print("\n")

def main():
    csv_file = "../../data_processed/dataset_splitted.csv"
    statistics(csv_file)

if __name__ == "__main__":
    main()