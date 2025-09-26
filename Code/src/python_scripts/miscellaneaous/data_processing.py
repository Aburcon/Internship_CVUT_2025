#=======================#
# Libraries importation #
#=======================#

from pandas import DataFrame

#==================================#
# Extracting data from a dataframe #
#==================================#

def extract_data(df: DataFrame, provider: str = None, techno: str = None, department: str = None, region: str = None, min_info: bool = True) -> DataFrame:
    """ Extracts data from a pandas DataFrame.
        
        Parameters
        ----------
        df : DataFrame
            The pandas DataFrame of your data.
        provider, techno, department, region : str
            The fields you want to select according to df.
        min_info : bool (default=True)
            True if you want to keep only the essential.
    """
    df.dropna(subset='id_station_anfr', inplace=True)
    df.set_index('id_station_anfr', inplace=True)

    if(min_info):
        df = df[['nom_op', 'x', 'y', 'latitude', 'longitude', 'nom_reg', 'nom_dep', 'nom_com', 'site_2g', 'site_3g', 'site_4g', 'site_5g']]
    if(provider):
        df = df.loc[df['nom_op'] == provider]
        df = df.drop(columns=['nom_op'])
    if(techno in ['2g', '3g', '4g', '5g']):
        df = df.loc[df[f"site_{techno}"] == 1]
        # df = df.drop(columns=[f"site_{techno}"])
    if(department):
        df = df.loc[df['nom_dep'] == department]
        df = df.drop(columns=['nom_dep', 'nom_reg'])
    if(region!=None):
        df = df.loc[df['nom_reg'] == region]
        df = df.drop(columns=['nom_reg'])

    return df
    