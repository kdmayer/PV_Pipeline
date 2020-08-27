## OpenNRW Platform:

- The openNRW platform provides exhaustive aerial imagery for the German state of North Rhine-Westphalia (NRW)
- Aerial imagery is characterized by a ground sampling distance (GSD) of 10 cm

## Goal:

- Detect solar panels on aerial imagery to create a database which records all PV system locations and their respective areas

    ![PV_system](https://github.com/kdmayer/PV_Pipeline/blob/master/PV%20system%201.png)
    ![PV_system](https://github.com/kdmayer/PV_Pipeline/blob/master/PV%20system%203.png)
    
## Overview:

- Tiles covering an area of 240x240m are downloaded (4800x4800 pixels), splitted into images of size 16x16m, and then classified in order to record the GPS coordinates of solar panels in a database

## Workflow:

Just set your configuration in config.yml and execute run_pipeline.py. In the background, the following three steps will happen:

* Tile_creator.py will automatically create the list of coordinates for all 596,722 tiles covering NRW, if TileCoords.pickle does not yet exist.
* Tile_downloader.py will automatically download the tiles specified in TileCoords.pickle in a multi-threaded fashion.
* Tile_processor.py will automatically process all completely downloaded files to identify and locate existing PV panels. To do so, Tile_processor.py splits tiles into images with a resolution of 320x320 pixels and classifies them with a CNN called DeepSolar. Images are classified as positive if they contain solar panels, negative otherwise

If not all tiles have been downloaded in the first run, just execute tile_updater.py to update TileCoords.pickle and re-run run_pipeline.py. By running "tile_updater.py", all tiles that have already been completely downloaded will be removed from Tile_coords.pickle, i.e. only tile coordinates not yet downloaded remain in the Tile_coords.pickle file.

## License:

[MIT](https://github.com/kdmayer/PV_Pipeline/blob/master/LICENSE

## Notes:

- NRW's geojson can be downloaded from https://github.com/kdmayer/deutschlandGeoJSON.git
- DeepSolar can be obtained from https://github.com/wangzhecheng



