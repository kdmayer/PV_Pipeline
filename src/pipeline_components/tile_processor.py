# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:31:47 2019

@author: Kevin
"""
from __future__ import print_function
from __future__ import division
from pathlib import Path
import torch
from torchvision import datasets, models, transforms, utils
import time
import csv
import os
import numpy as np
from itertools import compress
from shapely.geometry import Point
from PIL import Image
from torch.nn import functional as F
from torchvision.models import Inception3
from torch.utils.data import Dataset, DataLoader
from src.dataset.dataset import NrwDataset
from src.utils.geojson_handler_utils import GeoJsonHandler


class TileProcessor(object):

    def __init__(self, polygon):

        # Execute on gpu, if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ------ Load model configuration ------
        self.threshold = float(os.environ['threshold'])

        # Batch size should be as large as possible to speed up the classification process
        self.batch_size = int(os.environ['batch_size'])

        self.input_size = int(os.environ['input_size'])

        # ------ Specify required input directories ------
        self.checkpoint_path = os.environ['checkpoint_path']

        self.tile_dir = os.environ['tile_dir']

       # ------ Specify required output directories ------
        self.pv_db_path = os.environ['pv_db_path']

        self.processed_path = os.environ['processed_path']

        self.not_processed_path = os.environ['not_processed_path']

        # ------ Load model and dataset ------
        self.model = self.loadModel()

        self.dataset = NrwDataset(self.tile_dir)

        # ------ Set auxiliary instance variables ------
        self.polygon = polygon

        # Avg. earth radius in meters
        self.radius = 6371000

        # Square side length in meters
        self.side = 16

        # dlat spans a distance of 16 meters in north-south direction:
        self.dlat = (self.side * 360) / (2 * np.pi * self.radius)

    def loadModel(self):

        # Specify model architecture
        model = Inception3(num_classes=2, aux_logits=True, transform_input=False)
        model = model.to(self.device)

        # Load old parameters
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if self.checkpoint_path[-4:] == '.tar':  # it is a checkpoint dictionary rather than just model parameters

            model.load_state_dict(checkpoint['model_state_dict'])

        else:

            model.load_state_dict(checkpoint)

        return model

    def splitTile(self, tile, minx, miny, maxx, maxy):

        minx = float(minx)
        miny = float(miny)
        maxx = float(maxx)
        maxy = float(maxy)

        # Takes a 4800x4800 image tile and returns a list of 320x320 pixel images, if they are within
        # the NRW polygon

        tile = np.array(tile)

        images = []
        coords = []

        N = 0
        S = 4800
        W = 0
        E = 4800

        # The first image is taken from the upper left corner, we then slide from left
        # to right and from top to bottom. Since our images shall cover 16x16m, the initial
        # y coordinate is dlat/2 degrees, i.e. 8 meters, south of the maximum y coordinate.
        y_coord = maxy - self.dlat/2

        while N < S:

            W = 0

            x_coord = minx + (((self.side * 360) / (2 * np.pi * self.radius * np.cos(np.deg2rad(maxy))))/2)

            while W < E:

                # The first image is taken from the upper left corner, we then slide from left
                # to right and from top to bottom

                images.append(tile[N:N+320, W:W+320])
                coords.append((x_coord, y_coord))

                x_coord += (((self.side * 360) / (2 * np.pi * self.radius * np.cos(np.deg2rad(y_coord)))))

                W = W + 320

            N = N + 320

            y_coord = y_coord - self.dlat

        # A boolean vector of length 225 indicating whehter a image's centerpoint is within the NRW polygon
        coords_boolean = [self.polygon.intersects(Point(elem)) for elem in coords]

        # A list containing all images from the current tile that lie within NRW
        imagesInNRW = list(compress(images, coords_boolean))
        coordsInNRW = list(compress(coords, coords_boolean))

        return coordsInNRW, imagesInNRW

    def processTiles(self, currentTile, trans):

        # Load image tile
        tile = Image.open(Path(self.tile_dir + "/" + currentTile))

        if not tile.mode == 'RGB':

            tile = tile.convert('RGB')

        print(tile.size)

        currentTile = currentTile[:-13]

        minx, miny, maxx, maxy = currentTile.split(',')

        coords, images = self.splitTile(tile, minx, miny, maxx, maxy)

        length = len(images)

        if length == 0:
            
            pass

        else:

            k = int(length/self.batch_size)

            for i in range(k):

                print("i: ", i, "k:", k)

                images_sub = images[self.batch_size*i:self.batch_size*(i+1)]
                coords_sub = coords[self.batch_size*i:self.batch_size*(i+1)]

                # Image.fromarray() converts a numpy array into a PIL image
                # trans() resizes, normalizes, and converts PIL image to tensor
                # torch.unsqueeze(image tensor,0) adds a new dimension at the specified position (second argument), e.g.
                # converting our image tensor from [3,299,299] to [1,3,299,299]
                images_sub = [torch.unsqueeze(trans(Image.fromarray(image)),0) for image in images_sub]

                # torch.cat(image tensor, dim=0) concatenates our image tensor along dimension 0, i.e. a list
                # of tensors of the form [1,3,299,299] is converted into a tensor of form [N,3,299,299]
                images_sub = torch.cat(images_sub,dim=0)

                # Classify image

                outputs = self.model(images_sub)

                prob = F.softmax(outputs, dim=1)

                # detach() detaches the output from the computational graph.
                # So no gradient will be backproped along this variable.
                # For example if you’re computing some indices from the output of the
                # network and then want to use that to index a tensor. The indexing operation
                # is not differentiable wrt the indices. So you should detach() the indices
                # before providing them.

                prob = prob.detach().numpy()

                # If classification is positive, save coordinates x,y into database

                PV_bool = prob[:, 1] >= self.threshold

                PV_coords = list(compress(coords_sub, PV_bool))

                if PV_coords != []:

                    with open(Path(self.pv_db_path), "a") as csvFile:

                        writer = csv.writer(csvFile, lineterminator="\n")

                        for elem in range(len(PV_coords)):

                            writer.writerow([PV_coords[elem]])

            images_sub = images[self.batch_size*k:length]

            coords_sub = coords[self.batch_size*k:length]

            images_sub = [torch.unsqueeze(trans(Image.fromarray(image)), 0) for image in images_sub]

            images_sub = torch.cat(images_sub, dim=0)

            # Classify image

            outputs = self.model(images_sub)

            prob = F.softmax(outputs, dim=1)

            prob = prob.detach().numpy()

            # If classification is positive, save coordinates x,y into database

            PV_bool = prob[:, 1] >= self.threshold

            PV_coords = list(compress(coords_sub,PV_bool))

            print(prob[:,1])

            print(PV_coords)

            if PV_coords != []:

                with open(Path(self.pv_db_path),"a") as csvFile:

                    writer = csv.writer(csvFile,lineterminator="\n")

                    for elem in range(len(PV_coords)):

                        writer.writerow([PV_coords[elem]])

    def run(self):

        print('Dataset Size:', len(self.dataset))

        dataloader = DataLoader(self.dataset, batch_size=1, num_workers=0)

        self.model.eval()

        trans = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        for i, batch in enumerate(dataloader):

            currentTile = batch[0]

            # Try to process and record it
            try:

                self.processTiles(currentTile, trans)

                with open(Path(self.processed_path), "a") as csvFile:

                    writer = csv.writer(csvFile, lineterminator="\n")

                    writer.writerow([str(currentTile)])

            # Only tiles that weren't fully processed are saved subsequently
            except:

                # Save the tile which could not be processed and continue
                with open(Path(self.not_processed_path), "a") as csvFile:

                    writer = csv.writer(csvFile, lineterminator="\n")

                    writer.writerow([str(currentTile)])

            # Delete processed tile
            os.remove(Path(self.tile_dir + "/" + str(currentTile)))



'''

os.environ['threshold'] = f"{0.5}"
os.environ['batch_size'] = f"{10}"
os.environ['input_size'] = f"{299}"

print(os.getcwd())

os.environ['pv_db_path'] = '../../data/pv_database/PVs_NRW.csv'

os.environ['processed_path'] = '../../logs/processing/Processed.csv'

os.environ['not_processed_path'] = '../../logs/processing/notProcessed.csv'

print(type(float(os.environ['threshold'])))

os.environ['checkpoint_path'] = f"/Users/kevin/PV_Pipeline/models/DeepSolar_openNRW_classification.tar"
os.environ['tile_dir'] = f"/Users/kevin/PV_Pipeline/data/tiles"

nrw_handler = GeoJsonHandler("/Users/kevin/PV_Pipeline/data/deutschlandGeoJSON/2_bundeslaender/1_sehr_hoch.geojson")

tileProcessor = TileProcessor(nrw_handler.polygon)

tileProcessor.run()

'''