from ast import literal_eval
import pandas as pd
from shapely.geometry import shape, Point
from shapely.ops import transform
from rtree import index
import pyproj


def create_index(poly_gdf):
    idx = index.Index()
    for poly in poly_gdf.itertuples():
        idx.insert(poly.Index, shape(poly.geometry).bounds)
    return idx


def get_similarity(poi_name, poly_name, metric):
    pass
#     s1, s2 = helpers_lgm_interlinking.transform(
#         poi_name, poly_name, sorting=True, canonical=True)
#     if metric == 'lgm_sim_dl':
#         return lgm_sim(s1, s2)
#     elif metric == 'lgm_sim_jw':
#         return lgm_sim(s1, s2, metric='jaro_winkler')
#     elif metric == 'lgm_sim_me':
#         return lgm_sim(s1, s2, metric='monge_elkan')
#     elif metric == 'lgm_sim_jw_r':
#         return lgm_sim(s1[::-1], s2[::-1], metric='jaro_winkler')
#     else:
#         return avg_lgm_sim(s1, s2, metric='damerau_levenshtein')


def get_within_matches(poi_gdf, poly_gdf, step_poly_ids, idx):
    df = poi_gdf.copy()
    within = []
    for poi in df.itertuples():
        matched = None
        hits = idx.intersection(poi.geometry.coords[0])
        for i in hits:
            if poly_gdf.iloc[i]['id'] not in step_poly_ids:
                continue
            if poi.geometry.within(shape(poly_gdf.iloc[i]['geometry'])):
                matched = poly_gdf.iloc[i]['id']
                break
        within.append(matched)
    df['within'] = within
    df = pd.merge(df, poly_gdf, left_on=['within'], right_on=['id'])
    return df


def get_nearby_matches(poi_gdf, poly_gdf, step_poly_ids, idx):
    df = poi_gdf.copy()
    nearby = []
    for poi in df.itertuples():
        nearest = None
        hits = idx.nearest(poi.geometry.coords[0], 1)
        for i in hits:
            if poly_gdf.iloc[i]['id'] not in step_poly_ids:
                continue
            nearest = poly_gdf.iloc[i]['id']
            break
        nearby.append(nearest)
    df['nearby'] = nearby
    df = pd.merge(df, poly_gdf, left_on=['nearby'], right_on=['id'])
    return df


def apply_pre_matching_filters(poly_gdf, filters):
    df = poly_gdf.copy()
    if filters[0] == 'named':
        df = df[df['name'] != '[]']
    if filters[1] is not None:
        df = df[df['area'] < filters[1]]
    return df


def apply_post_matching_filters(matches_df, filters):
    df = matches_df.copy()
    if filters[0] is not None:
        df[filters[0]] = df.apply(lambda x: max([
            get_similarity(x['name_x'], poly_name, filters[0]) for poly_name in literal_eval(x['name_y'])
        ]), axis=1)
    if filters[1] is not None:
        df = df[df[filters[0]] > filters[1]]
    if filters[2] is not None:
        df = df[df['distance'] < filters[2]]
    return df


def get_poi_poly_matches(poi_gdf, poly_gdf, idx, strategy):
    matches = []
    unmatched_poi_df = poi_gdf.copy()
    for step, settings in strategy.items():
        mode, pre_filters, post_filters = settings
        # Apply pre-matching filters to polys
        step_poly_df = apply_pre_matching_filters(poly_gdf, pre_filters)
        # Get matches
        step_poly_ids = list(step_poly_df['id'])
        if mode == 'within':
            step_matches_df = get_within_matches(unmatched_poi_df, poly_gdf, step_poly_ids, idx)
        else:
            step_matches_df = get_nearby_matches(unmatched_poi_df, poly_gdf, step_poly_ids, idx)
        step_matches_df['distance'] = step_matches_df.apply(
            lambda x: shape(x['geometry_y']).distance(x['geometry_x']), axis=1
        )
        # Apply post-matching filters to current matches
        step_matches_df = apply_post_matching_filters(step_matches_df, post_filters)
        matches.append(step_matches_df)
        unmatched_poi_df = unmatched_poi_df[~unmatched_poi_df['poi_id'].isin(list(step_matches_df['poi_id']))]
    matches_df = pd.concat(matches, ignore_index=True, sort=False)
    print('Total matches:', len(matches_df))

    # cols_to_keep = ['poi_id', 'name_x', 'theme', 'class_name', 'subclass_n', 'geometry_x', 'id', 'name_y', 'tags',
    # 'geometry_y', 'area', 'lgm_sim_damerau_levenshtein', 'lgm_sim_jaro_winkler', 'lgm_sim_monge_elkan',
    # 'lgm_sim_jaro_winkler_r', 'avg_lgm_sim_damerau_levenshtein']
    # nearby = nearby[cols_to_keep]
    # nearby.columns = ['poi_id', 'poi_name', 'poi_theme', 'poi_class_name', 'poi_subclass_n', 'poi_geometry',
    # 'poly_id', 'poly_names', 'poly_tags', 'poly_geometry', 'poly_area', 'lgm_sim_damerau_levenshtein',
    # 'lgm_sim_jaro_winkler', 'lgm_sim_monge_elkan', 'lgm_sim_jaro_winkler_r', 'avg_lgm_sim_damerau_levenshtein']

    return matches_df


#   Example of strategy arg:

    # matching_strategy = [
    #     {1: ['within', ['named', 20000], ['avg_lgm_sim_dl', 0.5, None]],
    #      2: ['within', ['unnamed', 10000], [None, None, None]],
    #      3: ['nearby', ['named', 20000], ['lgm_sim_jw', 0.5, 50]],
    #      4: ['nearby', ['unnamed', 10000], [None, None, 50]]}
    # ]


def get_distance(p1, p2):
    """It finds the minimum distance between two Points

    Parameters
    ----------
    p1 : shapely geometric object
        The first point
    p2 : shapely geometric object
        The second point

    Returns
    -------
    list
        Returns the minimum distance. The value follows the geometric object projection.

    """
    dist = 5000
    try:
        dist = min(dist, p1.distance(p2))
    except TypeError as err:
        print(f'{err}')

    return [dist]


class Projection:
    """Transform coordinates of a geometric object among specified projections"""
    def __init__(self, src='4326', dest='3857'):
        """

        Parameters
        ----------
        src : str
            The current EPSG crs code.
        dest : str
            The target EPSG crs code.
        """
        self.project = pyproj.Transformer.from_proj(
            pyproj.Proj(f'epsg:{src}'),  # source coordinate system
            pyproj.Proj(f'epsg:{dest}'))  # destination coordinate system

    def change_projection(self, lon, lat):
        """Transforms the coordinates of a geometric object to the new projection.

        Parameters
        ----------
        lon : float
            The longitude of the geometric Point.
        lat : float
            The latitude of the geometric Point.

        Returns
        -------
            A shapely Point on the new projection.
        """
        p = Point(0, 0)
        try:
            p = transform(self.project.transform, Point(lon, lat))
        except TypeError as err:
            print(f'{err} for ({lon}, {lat})')

        return p
