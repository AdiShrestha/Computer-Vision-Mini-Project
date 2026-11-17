# Development product acquisition

The development adapter discovers original Sentinel-1 GRD and Sentinel-2 L2A
products through the Copernicus Data Space OData catalogue and can stage explicitly
selected product archives. A source inventory polygon is optional query context:
query centroids do not establish lake extent or a no-event control.

Use a byte-bound JSON acquisition manifest declaring WGS84 query centroids,
development exposure, products, acquisition interval and timezone-aware cutoff.
The runner accepts ordinary arguments:

    python3 -B source/runners/run_pilot_ingestion.py --manifest MANIFEST.json --manifest-sha256 SHA256 --output-dir NEW_DIRECTORY --mode catalogue

Every attempt uses a new output directory. Catalogue responses, page requests,
product identities, shared-site joins and checksums are retained. Complete
pagination is required; no cloud-filtered subset is presented as complete
observability. Catalogue metadata supplies no measured pixel or ROI quality.

For original-product downloads, configure a valid CDSE access token locally as
CDSE_ACCESS_TOKEN; never put it in arguments, manifests, logs or messages. Use
mode stage with a byte-bound catalogue result and one to four explicit product
IDs. Each product defaults to a 2 GiB bound, configurable up to 4 GiB. The adapter
rechecks current provider identity, streams bytes and checks length when reported,
SAFE identity, archive paths, CRC and required raster assets. Partial failed
downloads are retained. It performs no automatic retry or partial-download resume.

Archive checks are not pixel verification. Pixel decoding, radiometric calibration,
georeferencing, valid support and independently reviewed lake alignment remain
necessary. Raw GRD raster values are not automatically calibrated sigma-zero.
L2A reflectance requires the product's actual quantification and band offsets.
Dynamic water masks are estimated observations with cloud, snow, shadow, terrain
and classification uncertainty; they are not administrative ground truth.

The cutoff uses source-support end times. Event/post-event granules can be staged
but cannot enter pre-event decisions or define pre-event masks. Contemporary
archive retrieval does not establish historical operational availability.
All outputs remain outside scientific admission and do not establish research
readiness or performance.

Provider interfaces and terms:
[CDSE OData documentation](https://documentation.dataspace.copernicus.eu/APIs/OData.html),
[Sentinel data terms](https://dataspace.copernicus.eu/terms-and-conditions).
