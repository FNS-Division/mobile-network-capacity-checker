from mobile_capacity.entities.entity import Entity, EntityCollection


class PointOfInterest(Entity):
    """
    Represents a Point of Interest (POI) in geographical space.

    Parameters:
    - poi_id (str): Unique identifier for the POI.
    - lat (float): Latitude of the POI in decimal degrees.
    - lon (float): Longitude of the POI in decimal degrees.
    - entity_type (str, optional): Type of the entity, default is 'Point of Interest'.
    - **kwargs: Additional fields to store in the POI.

    Example:
    ```python
    # Create a PointOfInterest instance
    poi = PointOfInterest(poi_id='poi1', lat=34.0522, lon=-118.2437)

    # Access attributes of the POI
    poi_id = poi.poi_id
    ```
    """

    def __init__(self,
                 poi_id: str,
                 lat: float,
                 lon: float,
                 entity_type='Point of Interest',
                 **kwargs):

        super().__init__(entity_type=entity_type, **kwargs)

        self.poi_id = poi_id
        self.lat = lat
        self.lon = lon

    def __repr__(self):
        return f"PointOfInterest(poi_id={self.poi_id}, lat={self.lat}, lon={self.lon}, attributes={self.additional_fields})"


class PointOfInterestCollection(EntityCollection):
    """
    Represents a collection of Point of Interest entities.

    Parameters:
    - points_of_interest (list, optional): List of PointOfInterest instances.
    - poi_records (list, optional): List of dictionaries representing POI records.
    - poi_file_path (str, optional): Path to a CSV file containing POI records.

    Example:
    ```python
    # Create a PointOfInterestCollection instance
    poi_collection = PointOfInterestCollection(points_of_interest=[poi1, poi2, ...])

    # Load POIs from a CSV file
    poi_collection.load_from_csv('poi_data.csv')
    ```
    """

    def __init__(self, points_of_interest=None, poi_records=None, poi_file_path=None):

        super().__init__(entities=points_of_interest, entity_records=poi_records, entity_file_path=poi_file_path)

    @property
    def points_of_interest(self):
        # Get the list of PointOfInterest instances in the collection.
        return self.entities

    def add_entity(self, poi):
        # Add a PointOfInterest instance to the collection.
        if not isinstance(poi, PointOfInterest):
            raise TypeError(f'entity must be an instance of {PointOfInterest}')

        super().add_entity(entity=poi)

    def load_from_records(self, poi_records):
        """
        Load PointOfInterest instances from a list of dictionaries representing POI records.

        Parameters:
        - poi_records (list): List of dictionaries representing POI records.
        """
        for row in poi_records:
            poi = PointOfInterest.from_dict(row)
            self.add_entity(poi)
