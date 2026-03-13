import osmnx as ox
import geopandas as gpd
from typing import List, Dict, Any

class OSMIntegration:
    """
    Integration with OpenStreetMap to fetch infrastructure footprints.
    """
    
    def fetch_infrastructure_footprints(self, bbox: List[float], 
                                        tags: Dict[str, Any] = None) -> gpd.GeoDataFrame:
        """
        Fetches infrastructure footprints from OSM within a bounding box.
        
        Args:
            bbox: Bounding box [minx, miny, maxx, maxy].
            tags: OSM tags to filter for (e.g., {"power": "plant"}).
            
        Returns:
            GeoDataFrame containing the infrastructure footprints.
        """
        if tags is None:
            tags = {
                "power": ["plant", "substation", "generator", "transformer"],
                "building": ["industrial", "warehouse"],
                "industrial": True
            }
        
        # osmnx uses [north, south, east, west] or (north, south, east, west)
        # BBOX format in STAC is [minx, miny, maxx, maxy] (west, south, east, north)
        west, south, east, north = bbox
        
        gdf = ox.features_from_bbox(north, south, east, west, tags=tags)
        
        # Ensure we only have polygons or multipolygons
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        return gdf

    def export_to_geojson(self, gdf: gpd.GeoDataFrame, filepath: str):
        """
        Saves the GeoDataFrame to a GeoJSON file.
        
        Args:
            gdf: GeoDataFrame to save.
            filepath: Path to the output GeoJSON file.
        """
        gdf.to_file(filepath, driver='GeoJSON')
        print(f"Exported footprints to: {filepath}")

def fetch_infrastructure_footprints(bbox: List[float], tags: Dict[str, Any] = None) -> gpd.GeoDataFrame:
    """Convenience function to fetch footprints."""
    integration = OSMIntegration()
    return integration.fetch_infrastructure_footprints(bbox, tags)

def export_to_geojson(gdf: gpd.GeoDataFrame, filepath: str):
    """Convenience function to export GeoJSON."""
    integration = OSMIntegration()
    integration.export_to_geojson(gdf, filepath)
