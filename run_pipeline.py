'''
Workflow:
1. If TileCoords.pickle does not yet exist, create it with tile_creator.py to have a list of coordinates for all 596,722 tiles covering NRW.
2. Download tiles from TileCoords.pickle in a multi-threaded fashion with tile_downloader.py
3. Process all completely downloaded files to identify and locate existing PV panels with tile_processor.py
4. Run "tile_updater.py" to update TileCoords.pickle by removing all completely downloaded tiles and go back to step 2.

'''
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

    tile_dir = conf.get('tile_dir', 'data/tiles')
    os.environ['tile_dir'] = tile_dir

    tile_coords_path = conf.get('tile_coords_dir', 'data/coords/TileCoords.pickle')
    os.environ['tile_coords_path'] = tile_coords_path

    geojson_path = conf.get('geojson_path', 'utils/deutschlandGeoJSON/2_bundeslaender/1_sehr_hoch.geo.json')
    os.environ['geojson_path'] = geojson_path

    pv_db_path = conf.get('pv_db_path', 'data/pv_database/PVs_NRW.csv')
    os.environ['pv_db_path'] = pv_db_path

    processed_path = conf.get('processed_path', 'logs/processing/Processed.csv')
    os.environ['processed_path'] = processed_path

    not_processed_path = conf.get('not_processed_path', 'logs/processing/notProcessed.csv')
    os.environ['not_processed_path'] = not_processed_path

    downloaded_path = conf.get('downloaded_path', 'logs/processing/DownloadedTiles.csv')
    os.environ['downloaded_path'] = downloaded_path

    not_downloaded_path = conf.get('not_downloaded_path', 'logs/processing/notDownloadedTiles.csv')
    os.environ['not_downloaded_path'] = not_downloaded_path

    checkpoint_path = conf.get('checkpoint_path', 'models/DeepSolar_openNRW_classification.tar')
    os.environ['checkpoint_path'] = checkpoint_path

    batch_size = conf.get('batch_size', 10)
    os.environ['batch_size'] = f"{batch_size}"

    input_size = conf.get('input_size', 299)
    os.environ['input_size'] = f"{input_size}"

    threshold = conf.get('threshold', 0.5)
    os.environ['threshold'] = f"{threshold}"

    # ------- GeoJsonHandler provides utility functions -------

    nrw_handler = GeoJsonHandler(geojson_path)

    # ------- TileCreator creates pickle file with all tiles in NRW and their respective minx, miny, maxx, maxy coordinates -------

    # If pickle file does not exist, run tile_creator.py, else continue

    if Path(os.environ['tile_coords_path']).exists():

        print("Pickle file containing tile coordinates does exist. Proceed ...")

    else:

        print("Pickle file containing tile coordinates does not yet exist. Wait a couple of minutes ...")

        tileCreator = TileCreator(polygon=nrw_handler.polygon)

        tileCreator.defineTileCoords()

        print('Pickle file has been sucessfully created')

    # Tile_coords is a list of tuples. Each tuple specifies its respective tile by minx, miny, maxx, maxy.
    tile_coords = nrw_handler.returnTileCoords(path_to_pickle=Path(tile_coords_path))

    print(f'{len(tile_coords)} tiles have been identified within NRW')

    # ------- TileDownloader downloads tiles from openNRW in a multi-threaded fashion -------

    # We set this to <= 1 because Mac OS appears to store a hidden file ".DS_Store" in the directory
    if len(os.listdir(tile_dir)) <= 1:

        print("No tiles have been downloaded so far.")

        text = prompt('Do you want to start the downloading process? (y/n)')

        if text == 'y' or text == 'yes':

           print('Start downloading ... This will take a while')

           downloader = TileDownloader(polygon=nrw_handler.polygon, tile_coords=tile_coords)

        elif text == 'n' or text == 'no':

            print('Downloading step is skipped')

        else:

            print('Invalid input. Please restart script and enter "y", "yes", "n", "no", when encountering prompt.')

    if os.path.exists(Path(downloaded_path)):

        # Load DownloadedTiles.csv file
        downloadedTiles_df = pd.read_csv(Path("./logs/downloading/DownloadedTiles.csv"), header=None)

        print(f"Out of 596,722 tiles within NRW, {len(downloadedTiles_df.index)} tiles have been successfully downloaded.")

    text = prompt('Do you want to start processing the downloaded tiles? (y/n)')

    if text == 'y' or text == 'yes':

        tileProcessor = TileProcessor(polygon=nrw_handler.polygon)

        tileProcessor.run()

    elif text == 'n' or text == 'no':

        print('Processing step is skipped')

    else:

        print('Invalid input. Please restart script and enter "y", "yes", "n", "no", when encountering prompt.')


if __name__ == '__main__':

    main()