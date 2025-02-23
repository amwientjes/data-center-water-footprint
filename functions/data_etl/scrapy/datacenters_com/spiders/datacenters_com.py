"""Basic Scrapy spider to extract data from datacenters.com."""

from collections.abc import Generator
from pathlib import Path

import pandas as pd
from scrapy import Spider
from scrapy.http.response import Response


def get_stat_info_number(response: Response, selector: str) -> str | None:
    """Extracts the first number from a given statInfo CSS selector."""
    return response.css(selector + " > div#statInfo > strong::text").re_first(r"^(\d+(?:,\d{3})*(?:\.\d+)?) ")


# Fetch the master list containing basic information about the data centers
root_dir = Path(__file__).resolve().parents[5]
basic_dc_info = pd.read_csv(root_dir / "data/outputs/0_webscraping/datacenters_com_basic_info.csv")
dc_urls = basic_dc_info["url"].tolist()


class DataCenterSpider(Spider):
    """Scrapy spider to extract data from datacenters.com."""

    name = "datacenters_com"

    start_urls = dc_urls

    def parse(self, response: Response) -> Generator[dict[str, str | None]]:
        """Extracts information for a single data center."""
        yield {
            "url": response.url,
            "total_space_sqft": get_stat_info_number(response, "#totalSpace"),
            "colocation_space_sqft": get_stat_info_number(response, "#colocationSpace"),
            "critical_power_mw": get_stat_info_number(response, "#power"),
        }
