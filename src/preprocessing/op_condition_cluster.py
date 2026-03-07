from sklearn.cluster import KMeans

def cluster_operating_conditions(df, n_clusters=1):
    kmeans = KMeans(n_clusters=n_clusters, random_state = 42)

    df['op_cluster'] = kmeans.fit_predict(
            df[['op1','op2','op3']]
        )
    return df
