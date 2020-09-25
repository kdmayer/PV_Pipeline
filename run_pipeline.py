from __future__ import unicode_literals
import yaml
from pathlib import Path
import os
from prompt_toolkit import prompt
import pandas as pd

from src.pipeline_components.tile_creator import TileCreator
from src.pipeline_components.tile_downloader import TileDownloader
from src.pipeline_components.tile_processor import TileProcessor
from src.pipeline_components.tile_updater import TileCoordsUpdater
from src.utils.geojson_handler_utils import GeoJsonHandler

def main():

    # ------- Read configuration -------

    config_file = 'config.yml'

    with open(config_file, 'rb') as f:

        conf = yaml.load(f, Loader=yaml.FullLoader)

    run_tile_creator = conf.get('run_tile_creator', 0)
    run_tile_downloader = conf.get('run_tile_downloader', 0)
    run_tile_processor = conf.get('run_tile_processor', 0)
    run_tile_updater = conf.get('run_tile_coords_updater', 0)
    tile_coords_path = conf.get('tile_coords_path', 'data/coords/TileCoords.pickle')
    geojson_path = conf.get('geojson_path', 'utils/deutschlandGeoJSON/2_bundeslaender/1_sehr_hoch.geo.json')
    downloaded_path = conf.get('downloaded_path', 'logs/processing/DownloadedTiles.csv')
    processed_path = conf.get('processed_path','logs/processing/Processed.csv')


    # ------- GeoJsonHandler provides utility functions -------

    nrw_handler = GeoJsonHandler(geojson_path)

    # ------- TileCreator creates pickle file with all tiles in NRW and their respective minx, miny, maxx, maxy coordinates -------

    if run_tile_creator:

        print("Starting to create a pickle file with all bounding box coordinates for tiles within NRW ... This will take a while")

        tileCreator = TileCreator(configuration=conf, polygon=nrw_handler.polygon)

        tileCreator.defineTileCoords()

        print('Pickle file has been sucessfully created')

    # Tile_coords is a list of tuples. Each tuple specifies its respective tile by minx, miny, maxx, maxy.
    tile_coords = nrw_handler.returnTileCoords(path_to_pickle=Path(tile_coords_path))

    print(f'{len(tile_coords)} tiles have been identified.')

    # ------- TileDownloader downloads tiles from openNRW in a multi-threaded fashion -------

    if run_tile_downloader:

        print('Starting to download ' + str(len(tile_coords)) + '. This will take a while.')

        downloader = TileDownloader(configuration=conf, polygon=nrw_handler.polygon, tile_coords=tile_coords)

    if os.path.exists(Path(downloaded_path)):

        # Load DownloadedTiles.csv file
        downloadedTiles_df = pd.read_table(downloaded_path, header=None)

        print(f"{downloadedTiles_df[0].nunique()} unique tiles have been successfully downloaded.")

    if run_tile_processor:

        tileProcessor = TileProcessor(configuration=conf, polygon=nrw_handler.polygon)

        tileProcessor.run()

    if os.path.exists(processed_path):

        # Load DownloadedTiles.csv file
        processedTiles_df = pd.read_table(processed_path, header=None)

        print(f"{processedTiles_df[0].nunique()} unique tiles have been successfully processed.")

    if run_tile_updater:

        updater = TileCoordsUpdater(configuration=conf, tile_coords=tile_coords)

        updater.update()


if __name__ == '__main__':

    main()