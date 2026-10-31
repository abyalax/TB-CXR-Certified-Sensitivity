import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from math import cos, pi
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
manifest = list(
    csv.DictReader(
        (ROOT / "data/processed/primary/manifest.csv").open(encoding="utf-8")
    )
)
zipf = zipfile.ZipFile(ROOT / "data/raw/TBX11K.zip")
zip_names = {n.replace("\\", "/").lower(): n for n in zipf.namelist()}
zip_basenames = {n.rsplit("/", 1)[-1].lower(): n for n in zipf.namelist()}
mont = ROOT / "data/raw/montgomery/CXR_png"
shen = ROOT / "data/raw/shenzhen/CXR_png"
counts = defaultdict(Counter)
hashes = defaultdict(lambda: defaultdict(list))
global_hashes = defaultdict(list)
phashes = []
not_found = []
D = np.array(
    [[cos((2 * x + 1) * u * pi / 32) for x in range(32)] for u in range(8)],
    dtype=np.float32,
)
for row in manifest:
    site = row["site"]
    src = row["source_member"].replace("\\", "/")
    if site == "Montgomery":
        data = (mont / Path(src).name).read_bytes()
    elif site == "Shenzhen":
        data = (shen / Path(src).name).read_bytes()
    else:
        name = zip_names.get(src.lower()) or zip_basenames.get(Path(src).name.lower())
        if name is None:
            hits = [
                n for n in zip_names.values() if n.lower().endswith("/" + src.lower())
            ]
            if len(hits) != 1:
                not_found.append(row["image_id"])
                continue
            name = hits[0]
        data = zipf.read(name)
    im = Image.open(io.BytesIO(data))
    im.load()
    bit = (
        8 if im.mode in ("1", "L", "P", "RGB", "RGBA") else int(im.info.get("bits", 16))
    )
    counts[site][
        (
            im.format,
            im.mode,
            bit,
            *im.size,
            str(im.info.get("compression", "PNG DEFLATE; level unavailable")),
        )
    ] += 1
    gray = np.asarray(im.convert("L"))
    pixel_hash = hashlib.sha256(gray.tobytes()).hexdigest()
    rgb = np.asarray(im.convert("RGB"))
    pixel_hash = hashlib.sha256(
        str(rgb.shape).encode("ascii") + rgb.tobytes()
    ).hexdigest()
    hashes[site][pixel_hash].append(row["image_id"])
    global_hashes[pixel_hash].append((site, row["image_id"]))
    sm = np.asarray(
        Image.fromarray(gray).resize((32, 32), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    coeff = (D @ sm @ D.T)[:8, :8].reshape(-1)[1:]
    med = np.median(coeff)
    phashes.append((site, row["image_id"], np.packbits(coeff > med)))
# Compare cross-site perceptual signatures in chunks with a lookup-table popcount.
lookup = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
ids = [x[1] for x in phashes]
P = np.stack([x[2] for x in phashes])
sites = np.array([x[0] for x in phashes])
lookup = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
near = []
for site_a in sorted(set(sites)):
    ia = np.flatnonzero(sites == site_a)
    for site_b in sorted(set(sites)):
        if site_b <= site_a:
            continue
        ib = np.flatnonzero(sites == site_b)
        for start in range(0, len(ia), 128):
            block = ia[start : start + 128]
            d = lookup[np.bitwise_xor(P[block, None, :], P[None, ib, :])].sum(axis=2)
            rr, cc = np.where(d <= 4)
            near.extend(
                {
                    "site_a": site_a,
                    "image_a": ids[block[r]],
                    "site_b": site_b,
                    "image_b": ids[ib[c]],
                    "hamming": int(d[r, c]),
                }
                for r, c in zip(rr, cc, strict=True)
            )
duplicate_pairs_across_sites = [
    records
    for records in global_hashes.values()
    if len({site for site, _ in records}) > 1
]
split_roles = {}
for split_path in (ROOT / "data/splits").glob("*.csv"):
    with split_path.open(encoding="utf-8-sig", newline="") as stream:
        for split_row in csv.DictReader(stream):
            split_roles[(split_row["site"], split_row["image_id"])] = {
                "manifest": split_path.name,
                "role": split_row.get("role", ""),
            }
duplicate_groups_across_roles = []
for pixel_hash, records in global_hashes.items():
    roles = [split_roles.get(record) for record in records]
    if len(records) > 1 and len({role["role"] if role else None for role in roles}) > 1:
        duplicate_groups_across_roles.append(
            {"images": records, "roles": roles, "sha256": pixel_hash}
        )
embedding_audits = {}
for encoder_dir in (ROOT / "artifacts/embeddings").iterdir():
    feature_path = encoder_dir / "embeddings.npy"
    embedding_manifest = encoder_dir / "manifest.csv"
    if not feature_path.is_file() or not embedding_manifest.is_file():
        continue
    features = np.load(feature_path, mmap_mode="r")
    with embedding_manifest.open(encoding="utf-8-sig", newline="") as stream:
        embedding_rows = list(csv.DictReader(stream))
    if len(features) != len(embedding_rows):
        continue
    rng = np.random.default_rng(0)
    projection = rng.normal(size=(features.shape[1], 16)).astype(np.float32)
    projected = np.asarray(features, dtype=np.float32) @ projection
    projected /= np.linalg.norm(projected, axis=1, keepdims=True).clip(1e-12)
    groups = defaultdict(list)
    for index, embedding_row in enumerate(embedding_rows):
        group = (embedding_row.get("site", ""), embedding_row.get("source_split", ""))
        groups[group].append(index)
    neighbors = []
    for source_group, source_indices in groups.items():
        query = projected[source_indices]
        for target_group, target_indices in groups.items():
            if target_group == source_group:
                continue
            target_projection = projected[target_indices]
            for start in range(0, len(source_indices), 128):
                source_block = source_indices[start : start + 128]
                similarities = projected[source_block] @ target_projection.T
                indices = np.argmax(similarities, axis=1)
                distances = 1.0 - similarities[np.arange(len(source_block)), indices]
                for row_index, distance, neighbor_index in zip(
                    source_block, distances, indices, strict=True
                ):
                    neighbor_index = target_indices[int(neighbor_index)]
                    exact_cosine = 1 - np.dot(
                        features[row_index], features[neighbor_index]
                    ) / (
                        np.linalg.norm(features[row_index])
                        * np.linalg.norm(features[neighbor_index])
                    )
                    neighbors.append(
                        {
                            "image_id": embedding_rows[row_index].get("image_id"),
                            "site": source_group[0],
                            "source_split": source_group[1],
                            "neighbor_image_id": embedding_rows[neighbor_index].get(
                                "image_id"
                            ),
                            "neighbor_site": target_group[0],
                            "neighbor_source_split": target_group[1],
                            "projected_cosine_distance": float(distance),
                            "cosine_distance": float(exact_cosine),
                        }
                    )
    embedding_audits[encoder_dir.name] = {
        "method": (
            "16-dimensional seeded Gaussian projection with exact cosine "
            "reranking of nearest projected candidates"
        ),
        "candidate_count": len(neighbors),
        "nearest_neighbors": sorted(
            neighbors, key=lambda item: item["cosine_distance"]
        )[:100],
    }
report = {
    "audit_version": 1,
    "patient_level_supported": False,
    "unit": (
        "image-level; patient linkage is not verified by a crosswalk "
        "for all three datasets"
    ),
    "dataset_identity_documentation": {
        "Montgomery": (
            "The NLM source reports 138 CXRs and radiology readings; the "
            "project has no verified patient crosswalk."
        ),
        "Shenzhen": (
            "The NLM source reports 662 CXRs and radiology readings; the "
            "project has no verified patient crosswalk."
        ),
        "TBX11K": (
            "The dataset paper describes image-level categories; the local "
            "archive provides no project patient crosswalk."
        ),
    },
    "official_sources": {
        "NLM_Montgomery_and_Shenzhen": "https://www.lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayDatasets.html",
        "TBX11K_paper": "https://arxiv.org/abs/2307.02848",
    },
    "source_image_metadata_counts": {
        s: [
            {
                "format": k[0],
                "mode": k[1],
                "bit_depth": k[2],
                "width": k[3],
                "height": k[4],
                "compression": k[5],
                "count": v,
            }
            for k, v in c.items()
        ]
        for s, c in counts.items()
    },
    "high_bit_windowing_assessment": (
        "No native source image exceeded 8 bits per channel; the configured "
        "high-bit percentile window did not apply to these source images."
        if not any(key[2] > 8 for site_counts in counts.values() for key in site_counts)
        else (
            "At least one source image exceeded 8 bits per channel; "
            "see per-site counts."
        )
    ),
    "exact_decoded_rgb_pixel_duplicates": {
        s: {
            "unique_hashes": len(c),
            "duplicate_groups": [list(ids) for ids in c.values() if len(ids) > 1],
        }
        for s, c in hashes.items()
    },
    "exact_duplicate_groups_across_sites": duplicate_pairs_across_sites,
    "exact_duplicate_groups_across_split_roles": duplicate_groups_across_roles,
    "cross_site_perceptual_hash_candidates_hamming_le_4": near,
    "unmatched_archive_members": not_found,
    "embedding_nearest_neighbor_audit": embedding_audits,
    "limitations": [
        "Perceptual hash candidates are a screen, not proof of duplicate identity.",
        "One image per dataset record does not establish one image per person.",
        "No cross-dataset patient crosswalk was available.",
    ],
}
out = ROOT / "artifacts/audits/image_identity.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(
    json.dumps(
        {
            "output": str(out),
            "metadata_sites": {s: sum(c.values()) for s, c in counts.items()},
            "phash_candidates": len(near),
            "unmatched_archive_members": len(not_found),
        }
    )
)
