import os
import copy
import json
import time
import math
import pickle as pkl

import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import multiprocessing as mp
cpu_count = mp.cpu_count()

from tqdm import tqdm
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
from multiprocessing import Pool, allow_connection_pickling


from eviltransform import gcj2wgs
from shapely.prepared import prep
from shapely.ops import transform, unary_union
from pyproj import CRS, Transformer
from tqdm.contrib.concurrent import process_map
from shapely.geometry import Polygon, LineString, Point
from scipy.sparse import save_npz, load_npz, csr_matrix

from utils.procedure import *

from pprint import pprint

def region_geometries(config):
    IDs, adcodes = [], []
    geometries = []
    with open(config["data_path"] + config["city"] + "/block_level_2.geojson") as f:
        for line in f.readlines():
            line = json.loads(line) # dict_keys(['geometry'(coordinates, type), 'type', 'properties'(adcode, data_type, id, level)])
            Type = line["geometry"]["type"]
            if len(line["geometry"]["coordinates"]) == 1:
                coordinates = line["geometry"]["coordinates"][0]
                polygon = Polygon(coordinates)
                geometries.append(polygon)
            else:
                print("multi-polygon")
            
            IDs.append(int(line["properties"]["id"]))
            adcodes.append(int(line["properties"]["adcode"]))
    regions = pd.DataFrame([IDs, adcodes]).T
    regions.columns = ["ID", "adcode"]
    regions = regions.set_index("ID")
    anchor = regions.index
    return (regions, anchor), geometries

def region_attributes(config):
    gender = {"10":"male", "11":"female"}
    age = {"00":"unknown", "10":"0-17", "11":"18-24", "12":"25-30", "13":"31-35", "14":"36-40", "15":"41-45", "16":"46-60", "17":"61~"}
    poi = {}
    for idx, row in pd.read_excel(config["data_path"] + config["city"] + "/POI.xlsx", engine='openpyxl').iterrows():
        poi[str(row["业态编码"])] = (row["指标名称"], row["业态等级"])

    IDs, homenums, worknums = [], [], []
    for k, v in gender.items(): globals()["homegender_"+k] = []; globals()["workgender_"+k] = []
    for k, v in age.items(): globals()["homeage_"+k] = []; globals()["workage_"+k] = []
    for k, v in poi.items(): globals()["poi_"+k] = []
    with open(config["data_path"] + config["city"] + "/block_profile_202107.txt") as f:
        for line in f.readlines():
            line = json.loads(line)
            IDs.append(int(line["id"]))
            
            ########################################################### 处理home
            if "home" in line.keys():
                homenums.append(line["home"]["num"])
                for k, v in gender.items():
                    if k in line["home"]["sex"].keys():
                        globals()["homegender_"+k].append(line["home"]["sex"][k])
                    else:
                        globals()["homegender_"+k].append(0)
                for k, v in age.items():
                    if k in line["home"]["age"].keys():
                        globals()["homeage_"+k].append(line["home"]["age"][k])
                    else:
                        globals()["homeage_"+k].append(0)
            else:
                homenums.append(0)
                for k, v in gender.items(): globals()["homegender_"+k].append(0)
                for k, v in age.items(): globals()["homeage_"+k].append(0)
            
            ########################################################### 处理work
            if "work" in line.keys():
                worknums.append(line["work"]["num"])
                for k, v in gender.items():
                    if k in line["work"]["sex"].keys():
                        globals()["workgender_"+k].append(line["work"]["sex"][k])
                    else:
                        globals()["workgender_"+k].append(0)
                for k, v in age.items():
                    if k in line["work"]["age"].keys():
                        globals()["workage_"+k].append(line["work"]["age"][k])
                    else:
                        globals()["workage_"+k].append(0)
            else:
                worknums.append(0)
                for k, v in gender.items(): globals()["workgender_"+k].append(0)
                for k, v in age.items(): globals()["workage_"+k].append(0)
            
            if "poi_count" in line.keys():
                for k, v in poi.items():
                    if k in line["poi_count"].keys():
                        globals()["poi_"+k].append(line["poi_count"][k])
                    else:
                        globals()["poi_"+k].append(0)
            else:
                for k, v in poi.items():
                    globals()["poi_"+k].append(0)
    region_attr = pd.DataFrame([IDs, homenums] + 
                            [globals()["homegender_"+x] for x in gender.keys()] + 
                            [globals()["homeage_"+x] for x in age.keys()] + 
                            [worknums] + 
                            [globals()["workgender_"+x] for x in gender.keys()] + 
                            [globals()["workage_"+x] for x in age.keys()] + 
                            [globals()["poi_"+x] for x in poi.keys()]).T
    region_attr.columns = ["ID", "homenum"] + \
                        ["res:"+gender[x] for x in gender.keys()] + \
                        ["res:"+age[x] for x in age.keys()] + \
                        ["worknum"] + \
                        ["work:"+gender[x] for x in gender.keys()] + \
                        ["work:"+age[x] for x in age.keys()] + \
                        ["POI_"+x for x in poi.keys()]
    region_attr["ID"] = IDs
    region_attr = region_attr.set_index("ID")
    return region_attr

def regions_all(config):
    (regions, anchor), geometries = region_geometries(config)
    region_attr = region_attributes(config)
    regions =  pd.concat([regions, region_attr], axis=1, join='outer').fillna(0)
    regions = regions.reindex(anchor)
    regions = gpd.GeoDataFrame(regions, geometry=geometries)
    return regions

def regions_5rings(config):
    regions = regions_all(config)
    # 五环以内
    regions["center"] = regions["geometry"].apply(lambda x:x.centroid)
    lng_min, lat_min, lng_max, lat_max = config["lng_min"], config["lat_min"], config["lng_max"], config["lat_max"]
    regions = regions[regions["center"].apply(lambda x:((x.x >= lng_min) & (x.x <=lng_max) & (x.y >= lat_min) & (x.y <= lat_max)))]
    regions = regions.drop(columns=["center"]).sort_index().reset_index().reset_index().set_index("ID")
    return regions

def region_merge(config):
    merged_regions_path = config["data_path"] + config["city"] + "/region_merge/region_merge.shp"
    merge_origin_dict_path = config["data_path"] + config["city"] + "/region_merge/merge_origin_dict.json"
    if os.path.exists(merged_regions_path):
        regions_need_to_merge = gpd.read_file(merged_regions_path).set_index("ID")
        regions_need_to_merge.index = regions_need_to_merge.index.astype(int)
        with open(merge_origin_dict_path, "r") as f:
            save_merge_origin = json.load(f)
    else:
        regions = regions_5rings(config)
        # gcj2wgs
        def crs_trans_polygon(before):
            after = []
            for p in before.exterior.coords:
                lat, lng = gcj2wgs(p[1], p[0])
                after.append((lng, lat))
            return Polygon(after)
        regions["geometry"] = regions["geometry"].apply(crs_trans_polygon)
        # wgs2utm
        transformer = Transformer.from_crs(4326, 32650, always_xy=True)
        def wgs2utm50n(polygon):
            return transform(transformer.transform, polygon)
        regions["geometry"] = regions["geometry"].apply(wgs2utm50n)

        with tqdm(total=regions.shape[0]) as pbar:
            # area
            regions["area"] = regions["geometry"].apply(lambda x:x.area)
            merge_origin= {}
            id_generator = 0
            regions_need_to_merge = copy.deepcopy(regions)
            while regions_need_to_merge["area"].min() < 60000:
                idxmin = regions_need_to_merge["area"].idxmin()
                region_min = regions_need_to_merge.loc[idxmin]
                min_geometry = region_min["geometry"]
                distance = regions_need_to_merge["geometry"].apply(lambda x:x.distance(min_geometry))
                closed_ID = distance[distance!=0].idxmin()
                target_region = regions_need_to_merge.loc[closed_ID]
                merged_geometry = unary_union([target_region["geometry"], min_geometry])
                if (idxmin not in merge_origin.keys()) and (closed_ID not in merge_origin.keys()):
                    merged_ID = 10000 + id_generator
                    id_generator += 1
                    merge_origin[merged_ID] = [idxmin, closed_ID]
                    regions_need_to_merge = regions_need_to_merge.drop(index=[idxmin, closed_ID], axis=0)
                elif (idxmin not in merge_origin.keys()) and (closed_ID in merge_origin.keys()):
                    already_in_region_IDs = merge_origin[closed_ID]
                    already_in_region_IDs.append(idxmin)
                    merge_origin[closed_ID] = already_in_region_IDs
                    regions_need_to_merge = regions_need_to_merge.drop(index=idxmin, axis=0)
                    merged_ID = closed_ID
                elif (idxmin in merge_origin.keys()) and (closed_ID not in merge_origin.keys()):
                    already_in_region_IDs = merge_origin[idxmin]
                    already_in_region_IDs.append(closed_ID)
                    merge_origin[idxmin] = already_in_region_IDs
                    regions_need_to_merge = regions_need_to_merge.drop(index=closed_ID, axis=0)
                    merged_ID = idxmin
                else:
                    already_in_region_IDs_1 = merge_origin[idxmin]
                    already_in_region_IDs_2 = merge_origin[closed_ID]
                    already_in_region_IDs_1.extend(already_in_region_IDs_2)
                    merge_origin[idxmin] = already_in_region_IDs_1
                    merge_origin.pop(closed_ID)
                    regions_need_to_merge = regions_need_to_merge.drop(index=closed_ID, axis=0)
                    merged_ID = idxmin
                region_min["geometry"], target_region["geometry"] = 0, 0
                merged_series = region_min + target_region
                merged_series["geometry"] = merged_geometry
                regions_need_to_merge.loc[merged_ID] = merged_series
                pbar.update(1)

            # extensibility
            def Polygon2length_width(polygon):
                bbox = polygon.minimum_rotated_rectangle
                a, b ,c, d = list(bbox.exterior.coords)[:4]
                d1 = Point(a).distance(Point(b))
                d2 = Point(a).distance(Point(c))
                d3 = Point(a).distance(Point(d))
                axis = [d1, d2, d3]
                axis.sort()
                return axis[1] / axis[0]
            regions_need_to_merge["extensibility"] = regions_need_to_merge["geometry"].apply(Polygon2length_width)
            while regions_need_to_merge["extensibility"].max() > 2.5:
                idxmax = regions_need_to_merge["extensibility"].idxmax()
                region_max = regions_need_to_merge.loc[idxmax]
                max_geometry = region_max["geometry"]
                distance = regions_need_to_merge["geometry"].apply(lambda x:x.distance(max_geometry))
                closed_ID = distance[distance!=0].idxmin()
                target_region = copy.deepcopy(regions_need_to_merge.loc[closed_ID])
                merged_geometry = unary_union([target_region["geometry"], max_geometry])
                if (idxmax not in merge_origin.keys()) and (closed_ID not in merge_origin.keys()):
                    merged_ID = 10000 + id_generator
                    id_generator += 1
                    merge_origin[merged_ID] = [idxmax, closed_ID]
                    regions_need_to_merge = regions_need_to_merge.drop(index=[idxmax, closed_ID], axis=0)
                elif (idxmax not in merge_origin.keys()) and (closed_ID in merge_origin.keys()):
                    already_in_region_IDs = merge_origin[closed_ID]
                    already_in_region_IDs.append(idxmax)
                    merge_origin[closed_ID] = already_in_region_IDs
                    regions_need_to_merge = regions_need_to_merge.drop(index=idxmax, axis=0)
                    merged_ID = closed_ID
                elif (idxmax in merge_origin.keys()) and (closed_ID not in merge_origin.keys()):
                    already_in_region_IDs = merge_origin[idxmax]
                    already_in_region_IDs.append(closed_ID)
                    merge_origin[idxmax] = already_in_region_IDs
                    regions_need_to_merge = regions_need_to_merge.drop(index=closed_ID, axis=0)
                    merged_ID = idxmax
                else:
                    already_in_region_IDs_1 = merge_origin[idxmax]
                    already_in_region_IDs_2 = merge_origin[closed_ID]
                    already_in_region_IDs_1.extend(already_in_region_IDs_2)
                    merge_origin[idxmax] = already_in_region_IDs_1
                    merge_origin.pop(closed_ID)
                    regions_need_to_merge = regions_need_to_merge.drop(index=closed_ID, axis=0)
                    merged_ID = idxmax
                region_max["geometry"], target_region["geometry"] = 0, 0
                merged_extensibility = Polygon2length_width(merged_geometry)
                merged_series = region_max + target_region
                merged_series["geometry"] = merged_geometry
                merged_series["extensibility"] = merged_extensibility
                regions_need_to_merge.loc[merged_ID] = merged_series
                pbar.update(1)
            
            # compactness
            def Polygon2cycle_area(polygon):
                return polygon.length / polygon.area * 1000
            regions_need_to_merge["compactness"] = regions_need_to_merge["geometry"].apply(Polygon2cycle_area)
            while regions_need_to_merge["compactness"].max() > 7:
                idxmax = regions_need_to_merge["compactness"].idxmax()
                region_max = regions_need_to_merge.loc[idxmax]
                max_geometry = region_max["geometry"]
                distance = regions_need_to_merge["geometry"].apply(lambda x:x.distance(max_geometry))
                closed_ID = distance[distance!=0].idxmin()
                target_region = regions_need_to_merge.loc[closed_ID]
                merged_geometry = unary_union([target_region["geometry"], max_geometry])
                if (idxmax not in merge_origin.keys()) and (closed_ID not in merge_origin.keys()):
                    merged_ID = 10000 + id_generator
                    id_generator += 1
                    merge_origin[merged_ID] = [idxmax, closed_ID]
                    regions_need_to_merge = regions_need_to_merge.drop(index=[idxmax, closed_ID], axis=0)
                elif (idxmax not in merge_origin.keys()) and (closed_ID in merge_origin.keys()):
                    already_in_region_IDs = merge_origin[closed_ID]
                    already_in_region_IDs.append(idxmax)
                    merge_origin[closed_ID] = already_in_region_IDs
                    regions_need_to_merge = regions_need_to_merge.drop(index=idxmax, axis=0)
                    merged_ID = closed_ID
                elif (idxmax in merge_origin.keys()) and (closed_ID not in merge_origin.keys()):
                    already_in_region_IDs = merge_origin[idxmax]
                    already_in_region_IDs.append(closed_ID)
                    merge_origin[idxmax] = already_in_region_IDs
                    regions_need_to_merge = regions_need_to_merge.drop(index=closed_ID, axis=0)
                    merged_ID = idxmax
                else:
                    already_in_region_IDs_1 = merge_origin[idxmax]
                    already_in_region_IDs_2 = merge_origin[closed_ID]
                    already_in_region_IDs_1.extend(already_in_region_IDs_2)
                    merge_origin[idxmax] = already_in_region_IDs_1
                    merge_origin.pop(closed_ID)
                    regions_need_to_merge = regions_need_to_merge.drop(index=closed_ID, axis=0)
                    merged_ID = idxmax
                region_max["geometry"], target_region["geometry"] = 0, 0
                merged_compactness = Polygon2cycle_area(merged_geometry)
                merged_series = region_max + target_region
                merged_series["geometry"] = merged_geometry
                merged_series["compactness"] = merged_compactness
                regions_need_to_merge.loc[merged_ID] = merged_series
                pbar.update(1)
        save_merge_origin = {}
        for k, v in merge_origin.items():
            save_merge_origin[int(k)] = [int(x) for x in v]
        with open(merge_origin_dict_path, "w") as f:
            json.dump(save_merge_origin, f, indent=4)
        regions_need_to_merge = regions_need_to_merge.sort_index()
        regions_need_to_merge.index = regions_need_to_merge.index.astype(str)
        regions_need_to_merge.to_file(merged_regions_path)
    # 算一下每个区域的中心点的位置，作为区域的location特征
    regions_need_to_merge["centroid"] = regions_need_to_merge["geometry"].apply(lambda x : x.centroid)
    regions_need_to_merge["x"] = regions_need_to_merge["centroid"].apply(lambda x : float(x.coords[0][0]))
    regions_need_to_merge["y"] = regions_need_to_merge["centroid"].apply(lambda x : float(x.coords[0][1]))
    regions_need_to_merge = regions_need_to_merge.drop(columns=["centroid"])

    return regions_need_to_merge, save_merge_origin

def adjacency_matrix(config):
    adj_mat_path = config["data_path"] + config["city"] + "/region_distance.npy"
    if os.path.exists(adj_mat_path):
        return np.load(adj_mat_path)
    else:
        print("区域距离矩阵需要整理，正在整理...")
        regions, _ = region_merge(config)

        # # gcj2wgs
        # def crs_trans_polygon(before):
        #     after = []
        #     for p in before.exterior.coords:
        #         lat, lng = gcj2wgs(p[1], p[0])
        #         after.append((lng, lat))
        #     return Polygon(after)
        # regions["geometry"] = regions["geometry"].apply(crs_trans_polygon)

        # # wgs2utm
        # transformer = Transformer.from_crs(4326, 32650, always_xy=True)
        # def wgs2utm50n(polygon):
        #     return transform(transformer.transform, polygon)
        # regions["geometry"] = regions["geometry"].apply(wgs2utm50n)

        # diatance
        def one_multi_disatance(polygon):
            return regions["geometry"].apply(lambda x: x.distance(polygon))
        distance = regions["geometry"].parallel_apply(one_multi_disatance)
        np.save(adj_mat_path, np.array(distance).astype(np.float32))
        
        return distance

def OD(config):
    OD_path = config["data_path"] + config["city"] + "/OD.npy"
    if os.path.exists(OD_path):
        return np.load(OD_path)
    else:
        regions = regions_5rings(config).reset_index().reset_index().set_index("ID")
        merge_region, merge_region_dict = region_merge(config)
        merge_region_dict_reverse = {}
        for k, v in merge_region_dict.items():
            for i in v:
                merge_region_dict_reverse[i] = int(k)
        Os, Ds, Hours, counts = [], [], [], []
        with open(config["data_path"] + config["city"] + "/OD_202107.txt") as f:
            for idx, line in enumerate(f.readlines()):
                if idx % 1000000 == 0:
                    print("OD No. line", idx, end="\r")
                try:
                    line = line.strip().split("\t")
                    origin, hour, destination, count = int(line[0]), int(line[1]), int(line[2]), int(line[5])
                    if (origin not in regions.index) or (destination not in regions.index):
                        continue
                    origin = origin if origin not in merge_region_dict_reverse.keys() else merge_region_dict_reverse[origin]
                    destination = destination if destination not in merge_region_dict_reverse.keys() else merge_region_dict_reverse[destination]
                    if origin == destination:
                        continue
                    Os.append(origin)
                    Ds.append(destination)
                    Hours.append(hour)
                    counts.append(count)
                except:
                    if "null" not in line:
                        print(line)
                    pass
        OD_df = pd.DataFrame([Os, Ds, Hours, counts]).T
        OD_df.columns = ["Origin", "Destination", "Time", "Count"]
        OD_df = OD_df.groupby(["Origin", "Destination", "Time"]).sum().reset_index()
        merge_mapping = merge_region.reset_index().reset_index().set_index("ID")[["index"]]
        OD_df["Origin"] = merge_mapping.loc[OD_df["Origin"]]["index"].values
        OD_df["Destination"] = merge_mapping.loc[OD_df["Destination"]]["index"].values
        OD_mat = np.zeros([merge_region.shape[0], merge_region.shape[0], 24])
        OD_mat[OD_df["Origin"], OD_df["Destination"], OD_df["Time"]] = OD_df["Count"]
        np.save(OD_path, OD_mat)
        return OD_mat

def road_topology(config):
    adj = {}
    with open(config["data_path"] + config["city"] + "/zs_adj_bj_5ring") as f:
        for line in f.readlines(): # 32886条记录，共76449条ID
            line = line.strip().split("_")[:3]
            current, upstream, downstream = int(line[0]), [int(x) for x in line[1].split(",")], [int(x) for x in line[2].split(",")[:-1]]
            adj[current] = {"upstream": upstream, "downstream":downstream}
    return adj

def road_geo(config):
    lnglat = {}
    with open(config["data_path"] + config["city"] + "/zs_lnglats_BJ5ring") as f:
        for line in f.readlines(): # 经纬度有 29418 条
            line = line.strip().split(":")
            ID, lng_start, lat_start = line[0].split(",")
            # 如果需要转换坐标系的话 
            lat_start, lng_start = gcj2wgs(float(lat_start), float(lng_start))
            if ID not in lnglat.keys():
                lnglat[int(ID)] = []
                lnglat[int(ID)].append((float(lat_start), float(lng_start)))
            else:
                raise Exception("出现重复道路")
            for p in line[1:]:
                lng, lat = [float(x) for x in p.split(",")]
                # 如果需要转换坐标系的话
                lat, lng = gcj2wgs(lat, lng)
                lnglat[int(ID)].append((lat, lng))
    return lnglat

def road_graph(config):
    # 返回邻接矩阵，矩阵的index为road的ID从小到大排序
    road_graph_path = config["data_path"] + config["city"] + "/road_adj.npz"
    if os.path.exists(road_graph_path):
        return load_npz(road_graph_path).todense().astype(np.float32)
    else:
        adj = road_topology(config)
        speeds = traffic_speeds(config)
        def filter_road(local_conn):
            mid = local_conn[0]
            if mid not in list(speeds.index):
                return []
            else:
                ups = local_conn[1]["upstream"]
                downs = local_conn[1]["downstream"]
                edges = []
                for up in ups:
                    if up in speeds.index:
                        edges.append((up, mid))
                for down in downs:
                    if down in speeds.index:
                        edges.append((mid, down))
                return edges
        edges = map(filter_road, list(adj.items()))
        edges = sum(edges, [])
        edges = pd.DataFrame(np.array(edges))
        edges.columns = ["O", "D"]
        road_adj = np.zeros([29412, 29412])
        road_adj[speeds.loc[edges["O"]]["index"], speeds.loc[edges["D"]]["index"]] = 1
        save_npz(road_graph_path, csr_matrix(road_adj))
        # np.save(road_graph_path, road_adj)
        return road_adj
    
def traffic_speeds(config):
    speed_path = config["data_path"] + config["city"] + "/speeds.csv"
    if os.path.exists(speed_path):
        return pd.read_csv(speed_path, index_col="ID")
    else:
        IDs = []
        speeds = []
        with open(config["data_path"] + config["city"] + "/zs_BJ5ring_20220305to20220405_simplify") as f:
            for idx, line in enumerate(f.readlines()):# 29412 条路
                line = line.strip().split(":")
                ID = int(line[0].split(",")[0])
                IDs.append(ID)
                info = [list(map(lambda x: float(x.split("_")[1]), x.split("|")[0].split(","))) for x in line[1:]]
                info = list(filter(lambda x: len(x) == 288, info))
                info = np.array(info).mean(0).reshape([24, -1]).mean(1)
                speeds.append(info)
                # if idx % 5000 == 0:
                #     print(idx, end = " ")
                #     print(ID, end=" ")
                #     print(info.shape)
        speeds = pd.DataFrame(np.array(speeds))
        speeds = pd.DataFrame(speeds)
        speeds["ID"] = IDs
        speeds = speeds.set_index("ID").sort_index().reset_index().reset_index().set_index("ID")
        speeds.to_csv(speed_path)
        return speeds

def OD_relate_road(config):
    OD_relate_road_path = config["data_path"] + config["city"] + "/OD_relate_road.npz"
    if os.path.exists(OD_relate_road_path):
        return load_npz(OD_relate_road_path).todense().astype(np.float32)
    print("construct npz matrix of bipartile graph...")
    regions = regions_5rings(config)
    merge_region, merge_region_dict = region_merge(config)
    merge_region_dict_reverse = {}
    for k, v in merge_region_dict.items():
        for i in v:
            merge_region_dict_reverse[i] = int(k)

    # OD多边形
    OD_square_path = config["data_path"] + config["city"] + "/OD_pairs.shp"
    if os.path.exists(OD_square_path):
        OD_df = gpd.read_file(OD_square_path)
        OD_df["Origin"] = OD_df["Origin"].astype(int)
        OD_df["Destinatio"] = OD_df["Destinatio"].astype(int)
    else:
        Os, Ds, Hours, counts = [], [], [], []
        with open(config["data_path"] + config["city"] + "/OD_202107.txt") as f:
            for idx, line in enumerate(f.readlines()):
                if idx % 1000000 == 0:
                    print("OD_polygon No. line", idx, end="\r")
                try:
                    line = line.strip().split("\t")
                    origin, hour, destination, count = int(line[0]), int(line[1]), int(line[2]), int(line[5])
                    if (origin not in regions.index) or (destination not in regions.index):
                        continue
                    origin = origin if origin not in merge_region_dict_reverse.keys() else merge_region_dict_reverse[origin]
                    destination = destination if destination not in merge_region_dict_reverse.keys() else merge_region_dict_reverse[destination]
                    if origin == destination:
                        continue
                    Os.append(origin)
                    Ds.append(destination)
                    Hours.append(hour)
                    counts.append(count)
                except:
                    if "null" not in line:
                        print(line)
                    pass
        OD_df = pd.DataFrame([Os, Ds, Hours, counts]).T
        OD_df.columns = ["Origin", "Destination", "Time", "Count"]
        OD_df = OD_df.drop(columns=["Time"])
        OD_df = OD_df.groupby(["Origin", "Destination"]).sum().reset_index().drop(columns=["Count"])
        def getODSquare(ogn, dst):
            x1, y1 = list(merge_region.loc[ogn]["geometry"].centroid.coords)[0]
            x2, y2 = list(merge_region.loc[dst]["geometry"].centroid.coords)[0]
            return Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
        mid_square = OD_df.parallel_apply(lambda row: getODSquare(row["Origin"], row["Destination"]), axis=1)
        OD_df = gpd.GeoDataFrame(OD_df, geometry=mid_square)
        OD_df.to_file(OD_square_path)

    # OD与道路的关联
    print("OD与道路的关联")
    lnglat = road_geo(config)
    all_road = [(k, LineString([(x[1], x[0]) for x in v])) for k, v in lnglat.items()]
    all_OD = list(zip(list(OD_df.index), list(OD_df["geometry"])))
    def chuncks(arr, num_parts):
        n = int(math.ceil(len(arr) / float(num_parts)))
        return [arr[i:i + n] for i in range(0, len(arr), n)]
    OD_parts = chuncks(all_OD, 50)
    
    transformer = Transformer.from_crs(4326, 32650, always_xy=True)
    global road_relate_OD
    def road_relate_OD(OD):
        idx, polygon = OD
        def IF_in_OD(road):
            return (road[0], polygon.intersects(transform(transformer.transform, road[1])))
        roads = [x[0] for x in list(map(IF_in_OD, all_road)) if x[1]]
        OD_road_pairs = list(zip([idx] * len(roads), roads))
        return OD_road_pairs # roads


    # with Pool(processes=cpu_count) as pool:
    #     OD_road_pairs = list(tqdm(pool.imap(road_relate_OD, all_OD, chunksize= 1000), total=len(all_OD)))
    #     OD_road_pairs = sum(OD_road_pairs, [])
    # print(len(OD_road_pairs))
    # exit(0)

    OD_road_pairs_all_path = config["data_path"] + config["city"] + "/OD_road_pairs_all.pkl"
    if os.path.exists(OD_road_pairs_all_path):
        with open(OD_road_pairs_all_path, "rb") as f:
            OD_road_pairs_all = pkl.load(f)
    else:
        print("开始")
        OD_road_pairs_all = []
        for idx, OD_part in enumerate(OD_parts):
            print("Part No.", idx)
            idx_path = config["data_path"]+config["city"]+"/OD_relate_road/"+str(idx)+".txt"
            if os.path.exists(idx_path):
                print("读取list：")
                OD_road_pairs = []
                with open(idx_path, "r") as f:
                    for line in tqdm(f.readlines()):
                        line = line.strip()[1:-1]
                        OD_idx, road_idx = line.split(", ")
                        OD_idx, road_idx = int(OD_idx), int(road_idx)
                        OD_road_pairs.append((OD_idx, road_idx))
            else:
                print("计算list：")
                with Pool(processes=cpu_count) as pool:
                    OD_road_pairs = list(tqdm(pool.imap(road_relate_OD, OD_part, chunksize= 100), total=len(OD_part)))
                    print("合并list")
                    OD_road_pairs = sum(OD_road_pairs, [])
                    with open(idx_path, "w") as f:
                        print("写文件:")
                        for line in tqdm(OD_road_pairs):
                            f.write(str(line))
                            f.write("\n")
            OD_road_pairs_all.extend(OD_road_pairs)
        with open(OD_road_pairs_all_path, "wb") as f:
            pkl.dump(OD_road_pairs_all, f)
        print("list合成完毕")

    # 搞成npz存下来
    # 1. road ID 的从小到大的统一
    traffic = traffic_speeds(config)[["index"]]
    print("traffic:")
    print(traffic.head(5))
    print(traffic.info())
    # 2. OD id - road id的dataframe
    print("construct DataFrame...") # 这里ID数字很大，必须要使用int64才能处理
    OD_road_pairs_all = pd.DataFrame(np.array(OD_road_pairs_all), columns=["OD", "road"])
    print("construct dask DataFrame...")
    OD_road_pairs_all = dd.from_pandas(OD_road_pairs_all, npartitions=cpu_count*5)
    print(OD_road_pairs_all)
    print(OD_road_pairs_all.info())
    def road_in_traffic(road):
        return road in traffic.index
    print("removing not in road...")
    OD_road_pairs_all = OD_road_pairs_all[OD_road_pairs_all["road"].apply(road_in_traffic)]
    print(OD_road_pairs_all)
    print(OD_road_pairs_all.info())
    print("replacing road id...")

    # try to use dask
    d = traffic.loc[OD_road_pairs_all["road"]]["index"]
    c = np.array(OD_road_pairs_all.index.values)
    b = pd.DataFrame([c, d])
    b = b.T
    b.columns = ["idx", "road"]
    b = b.set_index("idx")
    print(b.head(5))
    print(b.shape)
    b = b["road"]
    a = dd.from_pandas(b, npartitions=cpu_count*5)
    OD_road_pairs_all["road"] = a
    print(OD_road_pairs_all)
    print(OD_road_pairs_all.info())
    print("saving...")
    OD_road_pairs_all.to_csv("OD_road_pairs_all.csv", index=False)
    print("saved.")
    exit()

    return None
    exit(0)
    # OD_df["roads"] = OD_df["geometry"].parallel_apply(road_relate_OD)
    # OD_roads_pairs = process_map(road_relate_OD, all_OD, chunksize = 1, max_workers = 48)
    # print(OD_df["roads"])
    print("完成")
    # with open(config["data_path"] + config["city"] + "/OD_roads_pairs.pkl", "wb") as f:
    #     pkl.dump(OD_roads_pairs, f)
    exit(0)

def preprocess(config):
    starttime = time.time()
    print("****** start preprocessing data ******")

    # # OD与路段的关联
    bipart = OD_relate_road(config)
    print(bipart.shape)
    exit()

    # 城市区域部分
    regions, _ = region_merge(config)
    print("regions:", regions.shape)
    adj_mat = adjacency_matrix(config)
    print("regional distance:", adj_mat.shape)

    # OD 部分
    OD_mat = OD(config)
    print("OD matrix:", OD_mat.shape)

    # 交通路网部分
    road_graph_adjmat = road_graph(config)
    print("road graph adjacency:", road_graph_adjmat.shape)
    speeds = traffic_speeds(config)
    print("speeds:", speeds.shape)

    print("****** time consume ", time.time() - starttime, "s*******", "\n")

    data = {"region_attr" : regions,
            "distance" : adj_mat,
            "OD" : OD_mat,
            "traffic_graph" : road_graph_adjmat,
            "speed" : speeds,
            "OD_relate_road" : bipart}

    return data

if __name__ == "__main__":
    config = get_conifg("/data/rongcan/code/24h-OD/src/config/beijing.json")
    regions_5 = regions_5rings(config)
    print(regions_5.head(5))
    print(regions_5.shape)