import featuretools as ft

def merge_featuretools(df_parent, df_related, parent_column, related_column, date_column):
    """Automated feature engineering

    More info:

    https://www.featuretools.com
    https://github.com/featuretools/featuretools
    https://docs.featuretools.com
    http://www.jmaxkanter.com/static/papers/DSAA_DSM_2015.pdf
    """

    # Create the entityset
    es = ft.EntitySet('parent')

    # Add the entities to the entityset
    es = es.entity_from_dataframe('parent', df_parent, index=parent_column)
    es = es.entity_from_dataframe('relate', df_related, make_index=True,
                                  time_index=date_column,
                                  index='related_id')

    # Define the relationships
    relationship = ft.Relationship(es['parent'][parent_column], es['relate'][related_column])

    # Add the relationships
    es = es.add_relationships([relationship])

    # Deep feature synthesis
    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity='parent')

    return feature_matrix.reset_index()
