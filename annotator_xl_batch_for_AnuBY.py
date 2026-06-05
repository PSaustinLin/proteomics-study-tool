import os
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (required in worker processes too)
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from annotator_xl import CrosslinkedMS2Annotator


def parse_modifications(mod_string):
    """
    Parse modification string with format: "site_mass,site_mass"
    e.g., "2_15.9949" or "2_15.9949,9_15.9949"
    Returns dictionary with 1-indexed positions as keys and mass deltas as values.
    """
    modifications = {}
    if not mod_string or pd.isna(mod_string) or mod_string.strip() == '':
        return modifications
    for mod_part in mod_string.strip().split(','):
        mod_part = mod_part.strip()
        if '_' in mod_part:
            try:
                site_str, mass_str = mod_part.split('_', 1)
                modifications[int(site_str)] = float(mass_str)
            except ValueError:
                print(f"Warning: Could not parse modification '{mod_part}'")
    return modifications


def _process_one_scan(task: dict) -> tuple[int, str | None]:
    """
    Worker function executed in a subprocess.
    Receives all arguments as a plain dict (must be picklable).
    Returns (scan_number, error_message_or_None).
    """
    # matplotlib must be set to non-interactive in every worker process
    matplotlib.use('Agg')

    scan_number = task['scan_number']
    try:
        annotator = CrosslinkedMS2Annotator(
            ms2_file=task['ms2_file_path'],
            alpha_sequence=task['alpha_sequence'],
            beta_sequence=task['beta_sequence'],
            scan_number=scan_number,
            alpha_modifications=task['alpha_modifications'],
            beta_modifications=task['beta_modifications'],
            alpha_crosslink_site=task['alpha_crosslink_site'],
            beta_crosslink_site=task['beta_crosslink_site'],
            crosslinker_mass=-2.01565,
            tolerance='20ppm',
        )
        annotator.annotate_crosslinked_spectrum(
            output_file=task['output_file'],
            csv_output=task['csv_output'],
            show_ann_text=False,
        )
        plt.close('all')
        gc.collect()
        return scan_number, None
    except Exception as exc:
        plt.close('all')
        gc.collect()
        return scan_number, str(exc)


def batch_process_csv(input_file: str, ms2_file_path: str, output_dir: str,
                      n_workers: int | None = None) -> None:
    """
    Process all rows in the CSV file and annotate spectra for each entry.
    Scans are dispatched to a ProcessPoolExecutor so that all CPU cores are
    used (one process per scan, up to n_workers at a time).

    n_workers defaults to max(1, cpu_count - 1).
    """
    # ── Load input ────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows from {input_file}")
    except Exception as exc:
        print(f"Error reading CSV file: {exc}")
        return

    required_columns = ['Scan', 'Alpha Peptide', 'Beta Peptide', 'Alpha XL Site', 'Beta XL Site']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}. Available: {df.columns.tolist()}")
        return

    alpha_mod_col = ('Alpha Mods' if 'Alpha Mods' in df.columns
                     else ('A mod' if 'A mod' in df.columns else None))
    beta_mod_col  = ('Beta Mods'  if 'Beta Mods'  in df.columns
                     else ('B mod'  if 'B mod'  in df.columns else None))

    os.makedirs(output_dir, exist_ok=True)

    # ── Build task list ───────────────────────────────────────────────────────
    tasks = []
    for _, row in df.iterrows():
        scan_number = int(row['Scan'])
        tasks.append({
            'ms2_file_path':       ms2_file_path,
            'scan_number':         scan_number,
            'alpha_sequence':      row['Alpha Peptide'],
            'beta_sequence':       row['Beta Peptide'],
            'alpha_crosslink_site': int(row['Alpha XL Site']),
            'beta_crosslink_site':  int(row['Beta XL Site']),
            'alpha_modifications': (parse_modifications(row[alpha_mod_col])
                                    if alpha_mod_col else {}),
            'beta_modifications':  (parse_modifications(row[beta_mod_col])
                                    if beta_mod_col else {}),
            'output_file':  os.path.join(output_dir, f"{scan_number}.png"),
            'csv_output':   os.path.join(output_dir, f"{scan_number}_matched_ions.csv"),
        })

    # ── Dispatch to worker processes ──────────────────────────────────────────
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Processing {len(tasks)} scans with {n_workers} worker processes...")

    processed_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one_scan, task): task['scan_number']
                   for task in tasks}
        for future in as_completed(futures):
            scan_number, error = future.result()
            if error:
                print(f"  [ERROR] scan {scan_number}: {error}")
                error_count += 1
            else:
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"  Processed {processed_count}/{len(tasks)} scans...")

    print(f"Done. {processed_count} succeeded, {error_count} failed.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # IMPORTANT: on Windows, ProcessPoolExecutor requires the entry point to be
    # guarded by `if __name__ == "__main__":` (already done here).

    input_files = [
        r"C:\env\test\IGHG3\xl_ds_20260511-TYG_1-HS1-150.csv",
        r"C:\env\test\IGHG3\xl_ds_20260511-TYG_2-HCC1-150.csv",
        r"C:\env\test\IGHG3\xl_ds_20260511-TYG_3-HS2-150.csv",
        r"C:\env\test\IGHG3\xl_ds_20260511-TYG_4-HCC2-150.csv",
        r"C:\env\test\IGHG3\xl_ds_20260511-TYG_5-HS2-245.csv",
        r"C:\env\test\IGHG3\xl_ds_20260511-TYG_6-HCC2-245.csv",
    ]

    ms2_file_paths = [
        r"C:\Crux\data\20260511\20260511-TYG_1-HS1-150.ms2",
        r"C:\Crux\data\20260511\20260511-TYG_2-HCC1-150.ms2",
        r"C:\Crux\data\20260511\20260511-TYG_3-HS2-150.ms2",
        r"C:\Crux\data\20260511\20260511-TYG_4-HCC2-150.ms2",
        r"C:\Crux\data\20260511\20260511-TYG_5-HS2-245.ms2",
        r"C:\Crux\data\20260511\20260511-TYG_6-HCC2-245.ms2",
    ]

    output_dirs = [
        r'C:\Crux\Output\20260601\HS1-150',
        r'C:\Crux\Output\20260601\HCC1-150',
        r'C:\Crux\Output\20260601\HS2-150',
        r'C:\Crux\Output\20260601\HCC2-150',
        r'C:\Crux\Output\20260601\HS2-245',
        r'C:\Crux\Output\20260601\HCC2-245',
    ]

    for input_file, ms2_file_path, output_dir in zip(input_files, ms2_file_paths, output_dirs):
        batch_process_csv(input_file, ms2_file_path, output_dir)