## openNRW platform:

- The openNRW platform provides exhaustive aerial imagery for the German state of North Rhine-Westphalia (NRW)
- Aerial imagery is characterized by a ground sampling distance (GSD) of 10 cm

## Goal:

- Detect solar panels on aerial imagery to create a database which records all PV system locations and their respective areas

    ![PV_system](https://github.com/kdmayer/openNRW-Pipeline/blob/master/PV%20system%202.png)
    ![PV_system](https://github.com/kdmayer/openNRW-Pipeline/blob/master/PV%20system%203.png)

## Workflow:

- Tiles covering an area of 240x240m are downloaded (4800x4800 pixels), splitted into images of size 16x16m, and then classified and segmented in order to record the GPS coordinates and the area of solar panels in a database

1. If TileCoords.pickle does not yet exist, tile_creator.py will automatically create the list of coordinates for all 596,722 tiles covering NRW.
2. Download tiles specified in TileCoords.pickle in a multi-threaded fashion with tile_downloader.py
3. Process all completely downloaded files to identify and locate existing PV panels. Tile_processor.py runs in the background, splitting tiles into images (320x320 pixels) and classifying them with a CNN called DeepSolar. Images are classified as positive if they contain solar panels, negative otherwise
4. Run "tile_updater.py" to update TileCoords.pickle. All tiles that have already been completely downloaded will be removed, i.e. only tile coordinates not yet been downloaded remain in the Tile_coords.pickle file. Go back to step 2 until all tiles have been downloaded


## Files:

* Tile_coords_NRW.py: 
  * Creates a pickle file containing a list with the coordinates of all image tiles in NRW
* Tile_Download.py: 
  * Downloads the previously specified tiles by their coordinates in a multi-threaded fashion
* Tile_Processing.py: 
  * Splits tiles into images (320x320 pixels) and classifies them with a CNN called DeepSolar
  * Images are classified as positive if they contain solar panels, negative otherwise
* Update_Tiles_coord.py: 
  * Updates the pickle file of image tile coordinates in case Tile_Download aborts
  * All tiles that have already been completely downloaded will be removed, i.e. only tile coordinates not yet been downloaded remain in the Tile_coords.pickle file

## Notes:

- NRW's geojson can be downloaded from https://github.com/kdmayer/deutschlandGeoJSON.git
- DeepSolar can be obtained from https://github.com/wangzhecheng



