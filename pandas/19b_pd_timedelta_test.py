import pandas as pd

        culture_time_delta_ufeme = culture + '_TIME_DELTA_UFEME'
        df_feature_merged[culture_time_delta_ufeme] = (df_feature_merged[culture_specimen_datetime_column] - df_feature_merged[specimen_collection_datetime_column]).dt.total_seconds()
        df_feature_merged[culture_time_delta_ufeme] = df_feature_merged[df_feature_merged[culture_time_delta_ufeme].abs() <= 86400]
