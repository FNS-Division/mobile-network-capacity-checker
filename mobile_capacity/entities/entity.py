import pandas as pd
import geopandas as gpd
import numpy as np
from mobile_capacity.spatial import haversine_
from shapely.geometry import Point


class Entity:
    """
    Represents an entity with geographical coordinates.

    Attributes:
    - entity_type (str): Type of the entity (default is 'Unidentified').
    - additional_fields (dict): Additional fields associated with the entity.

    Methods:
    - from_dict(cls, data_dict): Class method to create an Entity instance from a dictionary.
    - __repr__(): Returns a string representation of the Entity.
    """

    def __init__(self, entity_type='Unidentified', **kwargs):

        self.entity_type = entity_type
        self.additional_fields = kwargs

    @classmethod
    def from_dict(cls, data_dict):
        return cls(**data_dict)

    def get_point_geometry(self):
        return Point(self.lon, self.lat)

    def get_distance(self, other):
        if not isinstance(other, Entity):
            raise TypeError(f'other must be an instance of {Entity}')
        return haversine_(lats=[self.lat, other.lat], lons=[self.lon, other.lon])

    def __repr__(self):
        return f"Entity(entity_type= {self.entity_type}, attributes = {self.additional_fields})"


class EntityCollection:
    """
    Represents a collection of Entity instances.

    Attributes:
    - entities (list): List of Entity instances.

    Methods:
    - add_entity(entity): Adds an Entity to the collection.
    - load_entities(entities): Loads a list of entities into the collection.
    - load_from_records(entity_records): Loads entities from a list of records (dictionaries).
    - load_from_csv(file_path, **kwargs): Loads entities from a CSV file.
    - get_entity_types(): Returns a list of unique entity types in the collection.
    - get_entities_by_entity_type(entity_type): Returns entities of a specific entity type.
    - __len__(): Returns the number of entities in the collection.
    - __repr__(): Returns a string representation of the EntityCollection.
    """

    def __init__(self, entities: list = None, entity_records=None, entity_file_path=None):

        self.entities = []

        if entities is not None:
            self.load_entities(entities)

        if entity_records is not None:
            self.load_from_records(entity_records)

        if entity_file_path is not None:
            self.load_from_csv(entity_file_path)

    @property
    def data(self):
        return pd.DataFrame([entity.__dict__ for entity in self.entities])

    def add_entity(self, entity):

        if not isinstance(entity, Entity):
            raise TypeError(f'entity must be an instance of {Entity}')

        self.entities.append(entity)

    def load_entities(self, entities):
        for entity in entities:
            self.add_entity(entity)

    def load_from_records(self, entity_records):

        for row in entity_records:
            entity = Entity.from_dict(row)
            self.add_entity(entity)

    def load_from_csv(self, file_path, **kwargs):
        df = pd.read_csv(file_path, **kwargs)
        self.load_from_records(df.to_dict('records'))

    def get_entities_by_entity_type(self, entity_type):
        matching_entities = [entity for entity in self.entities if entity.entity_type == entity_type]
        return EntityCollection(entities=matching_entities)

    def get_entity_types(self):
        # Extract unique entity types from the entities in the collection
        unique_entity_types = set(entity.entity_type for entity in self.entities)

        return list(unique_entity_types)

    def get_nth_entity(self, index):
        if 0 <= index < len(self.entities):
            return self.entities[index]
        else:
            raise IndexError("Index out of range")

    def get_lat_array(self):
        return np.array([s.lat for s in self.entities])

    def get_lon_array(self):
        return np.array([s.lon for s in self.entities])

    def get_lat_lon_pairs(self):
        n = len(self.entities)
        lat_lon_pairs = np.empty((n, 2))

        for i, entity in enumerate(self.entities):
            lat_lon_pairs[i] = [entity.lat, entity.lon]

        return lat_lon_pairs

    def get_entity_distance_matrix(self):
        lat_lon_pairs = self.get_lat_lon_pairs()
        lats, lons = lat_lon_pairs[:, 0], lat_lon_pairs[:, 1]

        return haversine_(lats=lats, lons=lons)

    def get_geoseries(self):
        lat_lon_pairs = self.get_lat_lon_pairs()
        lats, lons = lat_lon_pairs[:, 0], lat_lon_pairs[:, 1]

        return gpd.GeoSeries(gpd.points_from_xy(lons, lats), crs='EPSG:4326')

    def __len__(self):
        return len(self.entities)

    def __repr__(self):
        return f"{self.__class__.__name__}: {len(self.entities)} entities"
