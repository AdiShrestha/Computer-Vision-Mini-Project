"""Small constructed SAFE layouts matching retained Sentinel-1 XML scopes."""
from __future__ import annotations

from datetime import datetime


def measurement_stem(scene: str, polarization: str, swath: str = "001") -> str:
    mission, mode, product_code, _, start, stop, orbit, datatake, _ = scene.split("_")
    return (f"{mission.lower()}-{mode.lower()}-{product_code[:3].lower()}-{polarization.lower()}-"
            f"{start.lower()}-{stop.lower()}-{orbit}-{datatake}-{swath}")


def _iso_second(compact: str) -> str:
    return datetime.strptime(compact, "%Y%m%dT%H%M%S").strftime("%Y-%m-%dT%H:%M:%S.000000")


def role_identity_xml(role: str, scene: str, polarization: str, *, mission=None,
                      product_type="GRD", mode=None, start=None, stop=None) -> bytes:
    parts = scene.split("_")
    values = {
        "missionId": mission or parts[0],
        "productType": product_type,
        "polarisation": polarization,
        "mode": mode or parts[1],
        "startTime": start or _iso_second(parts[4]),
        "stopTime": stop or _iso_second(parts[5]),
    }
    body = "".join(f"<{key}>{value}</{key}>" for key, value in values.items())
    extra = "<noiseRangeVector/>" if role.lower() == "noise" else ""
    return f"<{role}><adsHeader>{body}</adsHeader>{extra}</{role}>".encode()


def role_members(scene: str, polarization: str, swath: str = "001", *, include_rfi=True):
    stem = measurement_stem(scene, polarization, swath)
    members = {
        f"measurement/{stem}.tiff": b"constructed measurement bytes",
        f"annotation/{stem}.xml": role_identity_xml("product", scene, polarization),
        f"annotation/calibration/calibration-{stem}.xml":
            role_identity_xml("calibration", scene, polarization),
        f"annotation/calibration/noise-{stem}.xml":
            role_identity_xml("noise", scene, polarization),
    }
    if include_rfi:
        members[f"annotation/rfi/rfi-{stem}.xml"] = role_identity_xml("rfi", scene, polarization)
    return members


def scoped_manifest(scene: str, references, *, platform_number=None, orbit=None,
                    product_type="GRD", mode=None, start=None, stop=None,
                    polarizations=("VV", "VH")) -> bytes:
    parts = scene.split("_")
    platform_number = platform_number or parts[0][2:]
    orbit = orbit or str(int(parts[6]))
    mode = mode or parts[1]
    start = start or _iso_second(parts[4])
    stop = stop or _iso_second(parts[5])
    refs = "".join(f'<xfdu:fileLocation href="./{member}"/>' for member in references)
    pols = "".join(f"<safe:transmitterReceiverPolarisation>{pol}</safe:transmitterReceiverPolarisation>"
                   for pol in polarizations)
    xml = f'''<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1" xmlns:safe="urn:esa:safe:sentinel-1.0">
  <xfdu:metadataSection>
    <xfdu:metadataObject ID="platform"><xfdu:metadataWrap><xfdu:xmlData>
      <safe:platform><safe:familyName>SENTINEL-1</safe:familyName><safe:number>{platform_number}</safe:number>
        <safe:instrument><safe:familyName>Synthetic Aperture Radar</safe:familyName>
          <safe:extension><safe:instrumentMode><safe:mode>{mode}</safe:mode></safe:instrumentMode></safe:extension>
        </safe:instrument>
      </safe:platform>
    </xfdu:xmlData></xfdu:metadataWrap></xfdu:metadataObject>
    <xfdu:metadataObject ID="generalProductInformation"><xfdu:metadataWrap><xfdu:xmlData>
      <safe:standAloneProductInformation><safe:productType>{product_type}</safe:productType>{pols}</safe:standAloneProductInformation>
    </xfdu:xmlData></xfdu:metadataWrap></xfdu:metadataObject>
    <xfdu:metadataObject ID="acquisitionPeriod"><xfdu:metadataWrap><xfdu:xmlData>
      <safe:acquisitionPeriod><safe:startTime>{start}</safe:startTime><safe:stopTime>{stop}</safe:stopTime></safe:acquisitionPeriod>
    </xfdu:xmlData></xfdu:metadataWrap></xfdu:metadataObject>
    <xfdu:metadataObject ID="measurementOrbitReference"><xfdu:metadataWrap><xfdu:xmlData>
      <safe:orbitReference><safe:orbitNumber>{orbit}</safe:orbitNumber><safe:orbitNumber>{orbit}</safe:orbitNumber></safe:orbitReference>
    </xfdu:xmlData></xfdu:metadataWrap></xfdu:metadataObject>
  </xfdu:metadataSection>
  <xfdu:dataObjectSection>{refs}</xfdu:dataObjectSection>
</xfdu:XFDU>'''
    return xml.encode()
