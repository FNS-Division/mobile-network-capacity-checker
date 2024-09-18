from mobile_capacity.entities.entity import Entity, EntityCollection


class VisibilityPair(Entity):
    """
    Represents a pair of POI and cellular site, with visibility information.

    Parameters:
    - poi_id (str): Unique identifier for the point of interest.
    - ict_id (str): Unique identifier for the cellular site.
    - ground_distance (float): Ground distance between the POI and the cellular site in meters.
    - order (int): Order of the visibility pair, according to ground distance, in increasing order (1=closest cell site).
    - **kwargs: Additional fields to store in the cellular site.

    Attributes:
    - radio_type: Type of radio technology used by the cellular site.

    Example:
    ```python
    # Create a VisibilityPair instance
    pair = VisibilityPair(ict_id='site1', poi_id='poi1', ground_distance=100, order=1)

    # Access attributes of the pair
    ict_id = pair.ict_id
    ground_distance = pair.ground_distance
    ```
    """

    def __init__(self,
                 ict_id: str,
                 poi_id: str,
                 ground_distance: float,
                 order: int,
                 is_visible: bool,
                 entity_type: str = 'Visibility Pair',
                 **kwargs):
        super().__init__(entity_type=entity_type, **kwargs)

        self.poi_id = poi_id
        self.ict_id = ict_id
        self.ground_distance = ground_distance
        self.is_visible = is_visible
        self.order = order

    def __repr__(self):
        return f"VisibilityPair(ict_id={self.ict_id}, poi_id={self.poi_id}, attributes={self.additional_fields})"


class VisibilityPairCollection(EntityCollection):
    """
    Represents a collection of visibility pairs.

    Parameters:
    - cellsites (list, optional): List of CellSite instances.
    - cellsite_records (list, optional): List of dictionaries representing cellular site records.
    - cellsite_file_path (str, optional): Path to a CSV file containing cellular site records.

    Example:
    ```python
    # Create a CellSiteCollection instance
    cellsite_collection = CellSiteCollection(cellsites=[site1, site2, ...])

    # Load cellular sites from a CSV file
    cellsite_collection.load_from_csv('cellsite_data.csv')
    ```
    """

    def __init__(self, pairs=None, pair_records=None, pairs_file_path=None):
        super().__init__(entities=pairs, entity_records=pair_records, entity_file_path=pairs_file_path)

    @property
    def pairs(self):
        # Get the list of CellSite instances in the collection.
        return self.entities

    def load_from_records(self, pair_records):
        # Load VisibilityPair instances from a list of dictionaries representing cellular site records.
        for row in pair_records:
            cellsite = VisibilityPair.from_dict(row)
            self.add_entity(cellsite)
