#!/usr/bin/env bash
# Fetch the raw corpora into data/raw/.
#
# Nothing here is committed to the repository: the two archives are 3.2 GB and
# 21 GB. The pipeline streams from them in place and never expands them, so
# ~24 GB of free disk is enough.
#
# Every source has its own licence and access conditions. Read them. Two of
# these require you to agree to terms or register before downloading, and this
# script will not do that for you.

set -euo pipefail

RAW_DIR="${RAW_DIR:-data/raw}"
mkdir -p "$RAW_DIR"

cat <<'NOTES'
FlowCon-X raw data
==================

1. 5G Traffic Datasets  (Archive.zip, 3.2 GB)
   Source:  Kaggle, "5G Traffic Datasets"
            https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets
   Licence: CC BY 4.0 per the Kaggle dataset page. Attribution required.
   Access:  Kaggle account required. Download with the Kaggle CLI:
              kaggle datasets download -d kimdaegyeom/5g-traffic-datasets -p data/raw
            The pipeline expects the file at data/raw/Archive.zip.
   Content: 75 Wireshark packet exports across 15 applications in 6
            categories, captured on a commercial 5G network in Korea, 2022.

2. CESNET-QUIC22  (cesnet-quic22.zip, 21 GB)
   Source:  Zenodo, doi:10.5281/zenodo.7409648
            https://zenodo.org/records/7409648
   Paper:   Luxemburk et al., "CESNET-QUIC22: a large one-month QUIC network
            traffic dataset from backbone lines", Data in Brief, 2023.
   Licence: CC BY 4.0.
   Access:  Direct download, no registration.
   Content: 28 daily flow files over four weeks (2022-10-31 to 2022-11-27),
            with per-flow packet sequences, SNI, addresses and timestamps.

3. MAWI background traffic  (optional, for mixed-condition test sets)
   Source:  MAWI Working Group Traffic Archive, samplepoint-F
            https://mawi.wide.ad.jp/mawi/
   Licence: Free for research use; see the archive's terms. Traces are
            anonymised at source.
   Access:  Direct download of a daily .pcap.gz.

Checksums
---------
Recorded at the time of use in results/data/*_manifest.json under
`archive_sha256`. Verify with:

    shasum -a 256 data/raw/Archive.zip
    shasum -a 256 data/raw/cesnet-quic22.zip

and compare against the manifest committed alongside the results you are
reproducing. A mismatch means you have a different revision of the corpus and
the numbers will not match.
NOTES

if command -v kaggle >/dev/null 2>&1; then
  if [ ! -f "$RAW_DIR/Archive.zip" ]; then
    echo
    echo "==> Downloading 5G Traffic Datasets via the Kaggle CLI"
    kaggle datasets download -d kimdaegyeom/5g-traffic-datasets -p "$RAW_DIR"
  else
    echo "==> $RAW_DIR/Archive.zip already present, skipping"
  fi
else
  echo
  echo "==> kaggle CLI not found; fetch Archive.zip manually (see above)"
fi

if [ ! -f "$RAW_DIR/cesnet-quic22.zip" ]; then
  echo
  echo "==> Downloading CESNET-QUIC22 from Zenodo (21 GB, this takes a while)"
  curl -L --fail --retry 3 -o "$RAW_DIR/cesnet-quic22.zip" \
    "https://zenodo.org/records/7409648/files/cesnet-quic22.zip?download=1"
else
  echo "==> $RAW_DIR/cesnet-quic22.zip already present, skipping"
fi

echo
echo "==> Present in $RAW_DIR:"
ls -lh "$RAW_DIR"
echo
echo "==> Next: make data"
