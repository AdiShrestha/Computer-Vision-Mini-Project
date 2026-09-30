# Release Hygiene Policy

This policy defines the boundary between the private rehabilitation workspace and a public `sentinel-gl` release. It is a packaging and review rule: it does not remove files from the active working tree and does not rewrite historical commits.

## Public release rules

Every file proposed for public release must satisfy all of the following:

1. It contains no private machine paths, including `/Users/adi/...`, home-directory paths, local mount points, or credentials.
2. It contains no private control-plane metadata or workflow labels. Public documentation must not expose internal contract identifiers, gatekeeper terminology, or implementor/architect role labels.
3. Internal invariant identifiers are translated into public, reproducible protocol language. For example, the internal event-date control is described publicly as the pre-registered South Lhonak event date, 2023-10-04, rather than by an internal identifier.
4. Data, figures, tables, and model artifacts are either small, inspectable release files or are accompanied by a stable external retrieval record, checksum, license, and reproduction command.
5. Generated files are kept out of the source tree unless they are deliberate release artifacts. Local caches, checkpoints, temporary results, OS metadata, bytecode, and LaTeX build products are ignored by the repository rules.

The release directory is the explicit exception for generated outputs: `results/release/` and its contents may be tracked. A release file still requires provenance and review; the negated ignore rules do not certify its scientific validity.

## Data and large-artifact hosting

Raw observations, feature caches, embeddings, checkpoints, and large generated results are not silently committed. When redistribution is permitted, publish a versioned archive through Zenodo or Figshare and record its DOI, license, acquisition date, provider terms, and SHA-256 manifest in the release metadata. When redistribution is not permitted, publish the provider query or download recipe plus the same integrity metadata. A clean clone must state which external artifacts it needs and how to obtain them.

## Naming and documentation

The public repository identity is `sentinel-gl`; the legacy course-project name is retained only in private history notes and migration records. Public prose must use the current project name and describe the work as a multi-sensor glacier-lake anomaly-scoring case study. It must not imply causal prediction or a broader operational warning system than the verified protocol supports.

## Review checklist

Before a release is cut, scan tracked paths and rendered documentation for private paths, credentials, stale names, internal workflow labels, ignored build products, and missing external-artifact checksums. Verify a clean clone using only the documented release files and external retrieval records. Historical cleanup is a separate release operation and must occur only after the private archive has been verified.

## Current rehabilitation inventory

The forensic inventory identifies seven tracked bytecode/cruft files scheduled for removal from the public release, all under `source/data/acquisition/__pycache__/`:

| Class | Tracked paths | Release disposition |
| --- | --- | --- |
| Python bytecode / cache | `__init__.cpython-312.pyc`, `acquire_era5.cpython-312.pyc`, `acquire_itslive.cpython-312.pyc`, `acquire_landsat.cpython-312.pyc`, `acquire_modis.cpython-312.pyc`, `acquire_sentinel1.cpython-312.pyc`, `acquire_sentinel2.cpython-312.pyc` | Delete from clean release; retain only in the private forensic record if needed |

The separate tracked Factory continuation artifact is also excluded from the clean public release, but no tracked file is physically deleted during this chunk.
