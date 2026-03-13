import pystac_client
import boto3
import aiohttp
import asyncio
import os
import json
from typing import List, Dict, Any

class STACClient:
    """
    Client for interacting with STAC catalogs, specifically optimized for Capella SAR data.
    """
    
    def __init__(self, catalog_url: str):
        """
        Initialize the STAC client.
        
        Args:
            catalog_url: The URL of the STAC catalog.
        """
        self.catalog_url = catalog_url
        self.client = pystac_client.Client.open(catalog_url)

    def fetch_compatible_stacks(self, bbox: List[float], date_range: str, 
                                 max_incidence_diff: float = 5.0) -> List[Dict[str, Any]]:
        """
        Queries the catalog for SLC data compatible for interferometry.
        
        Args:
            bbox: Bounding box [minx, miny, maxx, maxy].
            date_range: Date range string (e.g., "2023-01-01/2023-12-31").
            max_incidence_diff: Maximum allowed difference in incidence angle.
            
        Returns:
            List of compatible STAC items.
        """
        search = self.client.search(
            bbox=bbox,
            datetime=date_range,
            collections=["capella-open-data"], # Example collection
            query={"sar:product_type": {"eq": "SLC"}}
        )
        
        items = list(search.items())
        # Further filtering logic for incidence angles and grazing angles would go here
        # This is a simplified version for the modular structure
        return [item.to_dict() for item in items]

    async def download_assets(self, item_list: List[Dict[str, Any]], output_dir: str):
        """
        Asynchronously downloads .tif and .json assets from the STAC items.
        
        Args:
            item_list: List of STAC items (as dicts).
            output_dir: Directory to save the downloaded assets.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        async with aiohttp.ClientSession() as session:
            tasks = []
            for item in item_list:
                for asset_key, asset in item.get("assets", {}).items():
                    if asset_key in ["data", "metadata"]:
                        url = asset["href"]
                        filename = os.path.basename(url)
                        dest_path = os.path.join(output_dir, filename)
                        tasks.append(self._download_file(session, url, dest_path))
            
            await asyncio.gather(*tasks)

    async def _download_file(self, session: aiohttp.ClientSession, url: str, dest_path: str):
        """Helper to download a single file."""
        async with session.get(url) as response:
            if response.status == 200:
                with open(dest_path, "wb") as f:
                    f.write(await response.read())
                print(f"Downloaded: {dest_path}")
            else:
                print(f"Failed to download {url}: {response.status}")

def initialize_client(catalog_url: str) -> STACClient:
    """Factory function to initialize STACClient."""
    return STACClient(catalog_url)
