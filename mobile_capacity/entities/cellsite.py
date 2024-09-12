from mobile_capacity.entities.entity import Entity, EntityCollection


class CellSite(Entity):
    """
    Represents a cellular site with antenna details.

    Parameters:
    - ict_id (str): Unique identifier for the cellular site.
    - lat (float): Latitude of the cellular site in decimal degrees.
    - lon (float): Longitude of the cellular site in decimal degrees.
    - radio_type: Type of radio technology used by the cellular site.
    - entity_type (str, optional): Type of the entity, default is 'Cell Site'.
    - **kwargs: Additional fields to store in the cellular site.

    Attributes:
    - radio_type: Type of radio technology used by the cellular site.

    Example:
    ```python
    # Create a CellSite instance
    cellsite = CellSite(ict_id='site1', lat=34.0522, lon=-118.2437, radio_type='4G')

    # Access attributes of the cellular site
    ict_id = cellsite.id
    radio_type = cellsite.radio_type
    ```
    """

    def __init__(self,
                 ict_id: str,
                 lat: float,
                 lon: float,
                 antenna_height: float,
                 radio_type,
                 entity_type: str = 'Cell Site',
                 **kwargs):
        super().__init__(entity_type=entity_type, **kwargs)

        self.ict_id = ict_id
        self.lat = lat
        self.lon = lon
        self.antenna_height = antenna_height
        self.radio_type = radio_type

    def __repr__(self):
        return f"CellSite(ict_id={self.ict_id}, lat={self.lat}, lon={self.lon},\
            radio_type= {self.radio_type}, antenna_height: {self.antenna_height}, attributes={self.additional_fields})"


class CellSiteCollection(EntityCollection):
    """
    Represents a collection of cellular sites.

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

    def __init__(self, cellsites=None, cellsite_records=None, cellsite_file_path=None):

        super().__init__(entities=cellsites, entity_records=cellsite_records, entity_file_path=cellsite_file_path)

    def load_from_records(self, cellsite_records):
        # Load CellSite instances from a list of dictionaries representing cellular site records.
        for row in cellsite_records:
            cellsite = CellSite.from_dict(row)
            self.add_entity(cellsite)

    def get_sites_by_radio_type(self, radio_type):
        # Get a collection of CellSite instances with a specific radio type.
        matching_sites = [site for site in self.entities if site.radio_type == radio_type]

        return CellSiteCollection(cellsites=matching_sites)

    def filter_sites_by_radio_type(self, allowed_radio_types):
        # Filter cellular sites based on allowed radio types.
        filtered_sites = [site for site in self.entities if site.radio_type in allowed_radio_types]

        return CellSiteCollection(cellsites=filtered_sites)
