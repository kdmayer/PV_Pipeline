## OpenNRW Platform:

- The openNRW platform provides exhaustive aerial imagery for the German state of North Rhine-Westphalia (NRW)
- Aerial imagery is characterized by a ground sampling distance (GSD) of 10 cm

## Goal:

- Detect solar panels on aerial imagery to create a database which records all PV system locations

    ![PV_system](https://github.com/kdmayer/PV_Pipeline/blob/master/PV%20system%201.png)
    ![PV_system](https://github.com/kdmayer/PV_Pipeline/blob/master/PV%20system%203.png)
    
## Overview:

- Tiles covering an area of 240x240m are downloaded (4800x4800 pixels), splitted into images of size 16x16m, and then classified in order to record the GPS coordinates of solar panels in a database

## Workflow:

Just set your configuration in config.yml and execute run_pipeline.py. In the background, the following three steps will happen:

* When "run_tile_creator:" is set to "1", Tile_creator.py will automatically create the list of coordinates for all 596,722 tiles covering NRW, if TileCoords.pickle does not yet exist. Alternatively, you can simply upload your own list of tile coordinates for your area of interest. However, please adhere to the format found in TileCoords.pickle in case you choose to upload a specific area of interest.
* When "run_tile_downloader:" is set to "1", Tile_downloader.py will automatically download the tiles specified in TileCoords.pickle in a multi-threaded fashion. If you don't want to download all tiles specified in TileCoords.pickle, feel free to abort this step at anytime.
* When "run_tile_processor:" is set to "1", Tile_processor.py will automatically process all completely downloaded files to identify and locate existing PV panels. To do so, Tile_processor.py splits tiles into images with a resolution of 320x320 pixels and classifies them with a CNN called DeepSolar. Images are classified as positive if they contain solar panels, negative otherwise

If not all tiles have been processed in the first run, just set "run_tile_coords_updater" to "1" and re-run run_pipeline.py. By running "tile_updater.py", all tiles that have already been completely processed will be removed from Tile_coords.pickle, i.e. only tile coordinates not yet processed remain in the Tile_coords.pickle file.

## Hint:

TileCoords.pickle should be splitted into multiple parts. By doing so, you can run multiple pipelines simultaneously, each of which only downloads and processes tiles specified in its respective TileCoords.pickle file.

## License:

[MIT](https://github.com/kdmayer/PV_Pipeline/blob/master/LICENSE)
