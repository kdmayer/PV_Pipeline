import pickle
from pathlib import Path
import pandas as pd
import os


class TileCoordsUpdater(object):

    def __init__(self, tile_coords):

        self.old_tile_coords = tile_coords

        self.tile_coords_path = os.environ['tile_coords_path']

        self.downloaded_path = os.environ['downloaded_path']

    def update(self):

        Tile_coords_df = pd.DataFrame(self.old_tile_coords)

        if os.path.exists(Path("self.downloaded_path")):

            # Load DownloadedTiles.csv file
            downloadedTiles = pd.read_csv(Path("self.downloaded_path"), header=None)

        else:

            print("DownloadedTiles.csv does not exist. Cannot update TileCoords.pickle by removing already downloaded tiles ...")

        downloadedTiles = downloadedTiles[0].str.split(pat = ',', expand = True)

        downloadedTiles[0] = downloadedTiles[0].apply(lambda x: x.replace("(",""))

        downloadedTiles[3] = downloadedTiles[3].apply(lambda x: x.replace(")",""))

        # Substract all elements in downloadedTiles.csv from Tile_coords_df
        Tile_coords_df = Tile_coords_df[~(Tile_coords_df[0].isin(downloadedTiles[0]) & Tile_coords_df[1].isin(downloadedTiles[1]) & Tile_coords_df[2].isin(downloadedTiles[2]) & Tile_coords_df[3].isin(downloadedTiles[3]))]

        # Convert Tile_coords dataframe to a list of tuples
        new_Tile_coords = list(Tile_coords_df.itertuples(index=False, name=None))

        # Save the new Tile_coords.pickle file
        with open(Path(self.tile_coords_path),'wb') as f:

            pickle.dump(new_Tile_coords, f)

        print(f"Old list of tiles contained {len(self.old_tile_coords)} elements. New list contains {len(new_Tile_coords)}")

        print(f"Successfully updated TileCoords.pickle by removing {len(self.old_tile_coords)-len(new_Tile_coords)} tiles.")

if __name__ == '__main__':

    updater = TileCoordsUpdater()

    updater.update()
