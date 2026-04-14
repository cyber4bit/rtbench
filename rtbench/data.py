from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd


FINGERPRINT_SIZES = {
    "maccs": 166,
    "ecfp6": 1024,
    "pubchem": 881,
}


@dataclass
class DatasetMatrix:
    dataset_id: str
    ids: list[str]
    mol_keys: list[str]
    X: np.ndarray
    X_mol: np.ndarray
    X_cp: np.ndarray
    y_sec: np.ndarray
    y_scale_sec: float
    t0_sec: float


def _download_file(url: str, out_path: Path, retries: int = 5, sleep_s: float = 1.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                data = resp.read()
            with open(out_path, "wb") as f:
                f.write(data)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed downloading {url}") from exc
            time.sleep(sleep_s * attempt)


def ensure_repo_data(
    repo_url: str,
    commit: str,
    data_root: Path,
    dataset_ids: Iterable[str],
    download: bool,
) -> None:
    processed_root = data_root / "processed_data"
    processed_root.mkdir(parents=True, exist_ok=True)
    if download:
        studies_url = f"{repo_url}/raw/{commit}/processed_data/studies.tsv"
        _download_file(studies_url, processed_root / "studies.tsv")
    for ds in dataset_ids:
        ds_root = processed_root / ds
        ds_root.mkdir(parents=True, exist_ok=True)
        required = required_files_for_dataset(ds)
        for name in required:
            local_file = ds_root / name
            if local_file.exists():
                continue
            if not download:
                continue
            url = f"{repo_url}/raw/{commit}/processed_data/{ds}/{name}"
            try:
                _download_file(url, local_file)
            except RuntimeError:
                # Some canonical/isomeric files may not exist for every dataset; keep strict checks separately.
                pass


def required_files_for_dataset(ds: str) -> list[str]:
    files = [f"{ds}_metadata.tsv", f"{ds}_gradient.tsv", f"{ds}_info.tsv"]
    for mode in ("canonical", "isomeric"):
        files.extend(
            [
                f"{ds}_rtdata_{mode}_success.tsv",
                f"{ds}_descriptors_{mode}_success.tsv",
                f"{ds}_fingerprints_maccs_{mode}_success.tsv",
                f"{ds}_fingerprints_ecfp6_{mode}_success.tsv",
                f"{ds}_fingerprints_pubchem_{mode}_success.tsv",
            ]
        )
    return files


def validate_required_inputs(processed_root: Path, dataset_ids: Iterable[str]) -> None:
    missing: list[str] = []
    for ds in dataset_ids:
        ds_root = processed_root / ds
        if not (ds_root / f"{ds}_metadata.tsv").exists():
            missing.append(f"{ds}: metadata")
        if not (ds_root / f"{ds}_gradient.tsv").exists():
            missing.append(f"{ds}: gradient")
        has_rt = any((ds_root / f"{ds}_rtdata_{m}_success.tsv").exists() for m in ("canonical", "isomeric"))
        has_desc = any((ds_root / f"{ds}_descriptors_{m}_success.tsv").exists() for m in ("canonical", "isomeric"))
        if not has_rt:
            missing.append(f"{ds}: rtdata")
        if not has_desc:
            missing.append(f"{ds}: descriptors")
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def _read_optional_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", encoding="utf-8")


def _canonical_plus_isomeric(ds_root: Path, dataset_id: str, stem: str) -> pd.DataFrame:
    can = _read_optional_tsv(ds_root / f"{dataset_id}_{stem}_canonical_success.tsv")
    iso = _read_optional_tsv(ds_root / f"{dataset_id}_{stem}_isomeric_success.tsv")
    if can.empty and iso.empty:
        return pd.DataFrame()
    if can.empty:
        return iso.drop_duplicates(subset="id", keep="first")
    if iso.empty:
        return can.drop_duplicates(subset="id", keep="first")
    can = can.drop_duplicates(subset="id", keep="first")
    iso = iso.drop_duplicates(subset="id", keep="first")
    add = iso.loc[~iso["id"].isin(can["id"])]
    return pd.concat([can, add], ignore_index=True)


def _parse_bits(bits: str, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    if not isinstance(bits, str) or not bits:
        return out
    for token in bits.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token) - 1
        if 0 <= idx < size:
            out[idx] = 1.0
    return out


def _build_gradient_features(gradient_df: pd.DataFrame, points: int = 20) -> np.ndarray:
    if gradient_df.empty:
        return np.zeros(points * 5, dtype=np.float32)
    time_col = gradient_df.columns[0]
    cols = [c for c in gradient_df.columns if any(k in c.lower() for k in ("a [", "b [", "c [", "d [", "flow"))]
    x = gradient_df[time_col].astype(float).to_numpy()
    if len(np.unique(x)) == 1:
        x = np.array([0.0, float(x[0]) + 1e-6], dtype=np.float64)
        y_mat = np.vstack([gradient_df[cols].iloc[0].to_numpy(dtype=float)] * 2)
    else:
        y_mat = gradient_df[cols].astype(float).to_numpy()
    x_n = (x - x.min()) / (x.max() - x.min())
    xi = np.linspace(0.0, 1.0, points)
    feats = []
    for c in range(y_mat.shape[1]):
        feats.append(np.interp(xi, x_n, y_mat[:, c]))
    return np.concatenate(feats).astype(np.float32)


@dataclass
class FeatureSchema:
    descriptor_cols: list[str]
    meta_num_cols: list[str]
    column_num_cols: list[str]
    eluent_num_cols: list[str]
    gradient_meta_num_cols: list[str]
    gradient_derived_cols: list[str]
    column_name_levels: list[str]
    column_usp_levels: list[str]
    gradient_points: int
    cpvec_dim: int = 0

    @property
    def uses_cpvec(self) -> bool:
        return int(self.cpvec_dim) > 0

    @property
    def group_sizes(self) -> dict[str, int]:
        out = {
            "descriptor": len(self.descriptor_cols),
            "fingerprint": sum(FINGERPRINT_SIZES.values()),
            "meta": len(self.meta_num_cols),
            "column_num": len(self.column_num_cols),
            "column_cat": len(self.column_name_levels) + len(self.column_usp_levels),
            "eluent": len(self.eluent_num_cols),
            # gradient_meta includes both raw metadata (start/end) and derived program summary (duration/steps).
            "gradient_meta": len(self.gradient_meta_num_cols) + len(self.gradient_derived_cols),
            "gradient_program": self.gradient_points * 5,
        }
        if self.uses_cpvec:
            out["cpvec"] = int(self.cpvec_dim)
        return out

    @property
    def mol_size(self) -> int:
        gs = self.group_sizes
        return gs["descriptor"] + gs["fingerprint"]

    @property
    def cp_size(self) -> int:
        return sum(self.group_sizes.values()) - self.mol_size


def build_feature_schema(
    processed_root: Path,
    dataset_ids: Iterable[str],
    gradient_points: int = 20,
    cpvec_dim: int = 0,
) -> FeatureSchema:
    descriptor_cols: set[str] = set()
    meta_num_cols: set[str] = set()
    column_num_cols: set[str] = set()
    eluent_num_cols: set[str] = set()
    gradient_meta_num_cols: set[str] = set()
    col_name_levels: set[str] = set()
    col_usp_levels: set[str] = set()

    for ds in dataset_ids:
        ds_root = processed_root / ds
        desc = _canonical_plus_isomeric(ds_root, ds, "descriptors")
        if not desc.empty:
            descriptor_cols.update([c for c in desc.columns if c != "id"])
        meta = pd.read_csv(ds_root / f"{ds}_metadata.tsv", sep="\t", encoding="utf-8")
        for c in meta.columns:
            if c == "id":
                continue
            as_num = pd.to_numeric(meta[c], errors="coerce")
            if not as_num.isna().all():
                if c.startswith("column."):
                    column_num_cols.add(c)
                elif c.startswith("eluent."):
                    eluent_num_cols.add(c)
                elif c.startswith("gradient."):
                    gradient_meta_num_cols.add(c)
                else:
                    meta_num_cols.add(c)
        col_name_levels.add(str(meta.get("column.name", pd.Series(["__MISSING__"])).iloc[0]))
        col_usp_levels.add(str(meta.get("column.usp.code", pd.Series(["__MISSING__"])).iloc[0]))

    return FeatureSchema(
        descriptor_cols=sorted(descriptor_cols),
        meta_num_cols=sorted(meta_num_cols),
        column_num_cols=sorted(column_num_cols),
        eluent_num_cols=sorted(eluent_num_cols),
        gradient_meta_num_cols=sorted(gradient_meta_num_cols),
        gradient_derived_cols=["gradient.t_end_min", "gradient.n_steps"],
        column_name_levels=sorted(col_name_levels),
        column_usp_levels=sorted(col_usp_levels),
        gradient_points=gradient_points,
        cpvec_dim=int(cpvec_dim),
    )


def _one_hot(value: str, levels: list[str]) -> np.ndarray:
    arr = np.zeros(len(levels), dtype=np.float32)
    if value in levels:
        arr[levels.index(value)] = 1.0
    return arr


def _build_mol_keys(rt_df: pd.DataFrame, ids: list[str]) -> list[str]:
    # Prefer stable cross-study molecular identifiers.
    pref_cols = ["inchikey.std", "inchi.std", "smiles.std", "name"]
    if rt_df.empty:
        return [str(i) for i in ids]
    key_s = pd.Series([""] * len(rt_df), index=rt_df.index, dtype=object)
    for c in pref_cols:
        if c not in rt_df.columns:
            continue
        v = rt_df[c].astype(str).str.strip()
        mask = key_s.eq("") & v.notna() & (~v.isin(["", "nan", "NA", "None"]))
        key_s.loc[mask] = v.loc[mask]
    key_s = key_s.reindex(ids).fillna("")
    out = []
    for sid, k in zip(ids, key_s.astype(str).tolist()):
        if k and k not in ("nan", "NA", "None"):
            out.append(k)
        else:
            out.append(str(sid))
    return out


def build_dataset_matrix(
    processed_root: Path,
    dataset_id: str,
    schema: FeatureSchema,
    cpvec_vec: np.ndarray | None = None,
) -> DatasetMatrix:
    ds_root = processed_root / dataset_id
    rt = _canonical_plus_isomeric(ds_root, dataset_id, "rtdata").set_index("id")
    desc = _canonical_plus_isomeric(ds_root, dataset_id, "descriptors").set_index("id")
    if rt.empty or desc.empty:
        raise ValueError(f"Dataset {dataset_id} is missing rt/descriptors data.")

    fp_parts = {}
    for fp_name in FINGERPRINT_SIZES:
        fp_df = _canonical_plus_isomeric(ds_root, dataset_id, f"fingerprints_{fp_name}")
        if fp_df.empty:
            fp_parts[fp_name] = pd.DataFrame(columns=["id", "bits.on"]).set_index("id")
        else:
            fp_parts[fp_name] = fp_df.set_index("id")[["bits.on"]]

    ids = sorted(set(rt.index).intersection(desc.index))
    if not ids:
        raise ValueError(f"No aligned ids found for dataset {dataset_id}.")
    mol_keys = _build_mol_keys(rt, ids)

    desc_aligned = desc.reindex(ids).reindex(columns=schema.descriptor_cols).fillna(0.0).astype(np.float32)
    y_sec = (rt.reindex(ids)["rt"].astype(float).to_numpy(dtype=np.float32) * 60.0)

    fp_mats = []
    for fp_name, fp_size in FINGERPRINT_SIZES.items():
        fp_df = fp_parts[fp_name]
        vecs = []
        for sid in ids:
            bits = fp_df.loc[sid, "bits.on"] if sid in fp_df.index else ""
            vecs.append(_parse_bits(bits, fp_size))
        fp_mats.append(np.vstack(vecs))
    fp_mat = np.concatenate(fp_mats, axis=1)

    meta = pd.read_csv(ds_root / f"{dataset_id}_metadata.tsv", sep="\t", encoding="utf-8")
    meta_row = meta.iloc[0]
    t0_raw = pd.to_numeric(pd.Series([meta_row.get("column.t0", np.nan)]), errors="coerce").iloc[0]
    t0_min = 0.0 if pd.isna(t0_raw) else float(t0_raw)
    t0_sec = max(1e-6, t0_min * 60.0)

    def _numeric_vec(cols: list[str]) -> np.ndarray:
        vals = []
        for c in cols:
            raw = meta_row.get(c, 0.0)
            val = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
            vals.append(0.0 if pd.isna(val) else float(val))
        return np.array(vals, dtype=np.float32)

    gradient = pd.read_csv(ds_root / f"{dataset_id}_gradient.tsv", sep="\t", encoding="utf-8")
    time_col = gradient.columns[0] if not gradient.empty else ""
    if time_col:
        t_end_min = float(pd.to_numeric(gradient[time_col], errors="coerce").max())
    else:
        t_end_min = 0.0
    meta_vec = _numeric_vec(schema.meta_num_cols)
    column_num_vec = _numeric_vec(schema.column_num_cols)
    eluent_vec = _numeric_vec(schema.eluent_num_cols)
    gradient_meta_vec = _numeric_vec(schema.gradient_meta_num_cols)
    col_name = str(meta_row.get("column.name", "__MISSING__"))
    col_usp = str(meta_row.get("column.usp.code", "__MISSING__"))
    cat_vec = np.concatenate(
        [_one_hot(col_name, schema.column_name_levels), _one_hot(col_usp, schema.column_usp_levels)]
    ).astype(np.float32)
    derived_vec = np.array([t_end_min, float(len(gradient))], dtype=np.float32)
    grad_vec = _build_gradient_features(gradient, points=schema.gradient_points)
    cp_vec_base = np.concatenate(
        [meta_vec, column_num_vec, cat_vec, eluent_vec, gradient_meta_vec, derived_vec, grad_vec],
        axis=0,
    ).astype(np.float32)
    if schema.uses_cpvec:
        if cpvec_vec is None:
            raise ValueError(f"cpvec_vec is required when schema.uses_cpvec=True (dataset={dataset_id})")
        cp_vec = np.asarray(cpvec_vec, dtype=np.float32).reshape(-1)
        if cp_vec.size != int(schema.cpvec_dim):
            raise ValueError(
                f"cpvec_vec dim mismatch for dataset {dataset_id}: got={cp_vec.size}, expected={schema.cpvec_dim}"
            )
        cp_vec_full = np.concatenate([cp_vec_base, cp_vec], axis=0).astype(np.float32)
        cp_mat = np.tile(cp_vec_full, (len(ids), 1))
    else:
        cp_mat = np.tile(cp_vec_base, (len(ids), 1))

    X_mol = np.concatenate([desc_aligned.to_numpy(), fp_mat], axis=1).astype(np.float32)
    X = np.concatenate([X_mol, cp_mat], axis=1).astype(np.float32)
    y_scale_sec = max(1e-6, t_end_min * 60.0)
    return DatasetMatrix(
        dataset_id=dataset_id,
        ids=ids,
        mol_keys=mol_keys,
        X=X,
        X_mol=X_mol,
        X_cp=cp_mat,
        y_sec=y_sec,
        y_scale_sec=y_scale_sec,
        t0_sec=t0_sec,
    )


def build_all_matrices(
    processed_root: Path,
    dataset_ids: Iterable[str],
    gradient_points: int = 20,
    cpvec_map: dict[str, np.ndarray] | None = None,
) -> tuple[FeatureSchema, dict[str, DatasetMatrix]]:
    ds_list = list(dataset_ids)
    cpvec_dim = 0
    if cpvec_map:
        first = next(iter(cpvec_map.values()))
        cpvec_dim = int(np.asarray(first, dtype=np.float32).reshape(-1).size)
    schema = build_feature_schema(processed_root, ds_list, gradient_points=gradient_points, cpvec_dim=cpvec_dim)
    mats: dict[str, DatasetMatrix] = {}
    for ds in ds_list:
        vec = None
        if schema.uses_cpvec:
            if not cpvec_map or ds not in cpvec_map:
                raise KeyError(f"cpvec_map missing dataset '{ds}' while cpvec is enabled")
            vec = cpvec_map[ds]
        mats[ds] = build_dataset_matrix(processed_root, ds, schema, cpvec_vec=vec)
    return schema, mats


def pretrain_count_14(mats: dict[str, DatasetMatrix], pretrain_ids: Iterable[str]) -> int:
    return int(sum(len(mats[ds].ids) for ds in pretrain_ids))
