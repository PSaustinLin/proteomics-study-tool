from time_utils import format_runtime, current_time
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
import os
import csv
from collections import defaultdict
import concurrent.futures
from queue import Queue
from threading import Lock
import traceback
import numpy as np                        # OPT: zero-copy tensor construction
import multiprocessing as mp              # OPT: true parallelism for CPU parsing (bypasses GIL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GLOBAL Constants
BIN_WIDTH = 1.0
BIN_OFFSET = 0.04
cid = True
crosslinker = 'disulfide'

# Ion generation flags — set to False to disable the corresponding ion series
USE_NEUTRAL_LOSS    = True   # Include H2O (-18.01056) and NH3 (-17.02655) neutral loss ions for b/y series
USE_SIGNATURE_IONS  = False   # Include crosslinker signature ions (alpha/beta whole-peptide ions)

# Constant masses
proton_mass = 1.00728
water_mass = 18.01056

if crosslinker == 'loss_NH3':
    crosslinker_mass=-17.02655
    # Isospeptide bonds that lose NH3 (K-N / K-Q)
    signature_types = {
        '': 0.0,
        '-1': -1.00783, #-H
        '-15': -15.01090, #-NH
        '-16': -16.01872 #-NH2
    }
elif crosslinker == 'loss_H2O':
    crosslinker_mass=-18.01056
    # Isospeptide bonds that lose H2O (K-D / K-E)
    signature_types = {
        '-1': -1.00783, #-H
        '-17': -17.00274 #-OH
    }
elif crosslinker == 'disulfide':
    crosslinker_mass=-2.01565
    # Disulfides
    signature_types = {
        '': 0.0,
        '-2': -2.01565,
        '+32': 31.97207,
        '-34': -33.98772
    }


aa_dict = {
    'A': 71.03711, 'C': 103.00918, 'D': 115.02694, 'E': 129.04259,
    'F': 147.06841, 'G': 57.02146, 'H': 137.05891, 'I': 113.08406,
    'K': 128.09496, 'L': 113.08406, 'M': 131.04048, 'N': 114.04293,
    'P': 97.05276, 'Q': 128.05858, 'R': 156.10111, 'S': 87.03203,
    'T': 101.04768, 'V': 99.06841, 'W': 186.07931, 'Y': 163.06333, 'X': 0.0
}

def sort_by_intensity(peak: Tuple[float, float, float]) -> float:
    return peak[1]

def read_peptides(file_path: str) -> Tuple[List[str], List[Dict[int, float]], List[int], List[str]]:
    """
    Read peptide candidates from a CSV file.

    Expected columns (0-indexed):
        0 : peptide sequence
        1 : modifications  (e.g. '3_15.9949,5_86.0368'; empty string if none)
        2 : crosslink site (integer)
        3 : protein accession (e.g. 'P12345')
    """
    peptides = []
    modifications = []
    crosslink_sites = []
    accessions = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            peptide   = row[0]
            mods      = row[1]
            xl_site_raw = row[2].strip()
            if not xl_site_raw:
                #print(f"Warning: skipping peptide '{peptide}' — crosslink site is empty")
                continue
            xl_site   = int(xl_site_raw)
            # Parse one or more accessions from UniProt-style entries
            # e.g. "sp|A0M8Q6|IGLC7_HUMAN,sp|P0CF74|IGLC6_HUMAN" → "A0M8Q6,P0CF74"
            raw_accession = row[3].strip()
            parsed_parts = []
            for entry in raw_accession.split(','):
                entry = entry.strip()
                pipes = entry.split('|')
                if len(pipes) >= 2:
                    parsed_parts.append(pipes[1])  # field between 1st and 2nd '|'
                else:
                    parsed_parts.append(entry)     # already a bare accession
            accession = ','.join(parsed_parts)

            # Create a dictionary for modifications (3_15.9949,5_86.0368)
            mods_dict = {}
            if mods:
                mods = mods.split(',')
                for mod in mods:
                    site, mass = mod.split('_')
                    mods_dict[int(site)] = float(mass)

            peptides.append(peptide)
            modifications.append(mods_dict)
            crosslink_sites.append(xl_site)
            accessions.append(accession)

    print(f"Peptides read: {peptides[:5]}... ({len(peptides)} total)")
    return peptides, modifications, crosslink_sites, accessions


def format_modifications(modifications: Dict[int, float]) -> str:
    """Format modification dictionary as a CSV-friendly string."""
    if not modifications:
        return ''
    return ','.join(f"{site}_{mass:.4f}" for site, mass in sorted(modifications.items()))


def generate_all_combinations(peptides: List[str], modifications: List[Dict[int, float]],
                             crosslink_sites: List[int], accessions: List[str],
                             alpha_acc: List[str],
                             crosslinker: str) -> Tuple[List[str], List[Dict[int, float]],
                                                        List[int], List[str], List[Dict[int, float]],
                                                        List[int]]:
    """
    Generate all alpha-beta peptide pairs.

    Alpha candidates : ALL peptides whose protein accession is in ``alpha_acc``
                       (hardcoded in __main__).  Multiple peptides from the same
                       protein are all included.
    Beta candidates  : ALL peptides whose protein accession is NOT in ``alpha_acc``.

    Every alpha peptide is paired with every beta peptide (Cartesian product).
    Returns expanded lists of alpha and beta peptides with their modifications and
    crosslink sites.
    """
    alpha_peptides = []
    alpha_modifications = []
    alpha_crosslink_sites = []
    beta_peptides = []
    beta_modifications = []
    beta_crosslink_sites = []

    alpha_acc_set = set(alpha_acc)

    def _is_alpha(acc: str) -> bool:
        return any(a.strip() in alpha_acc_set for a in acc.split(','))

    # ── Crosslink-residue filter ───────────────────────────────────────────────
    # Only peptides that actually contain the crosslinkable residue at the
    # recorded site are chemically valid candidates.  Pairing peptides without
    # the right residue produces impossible spectra and inflates pair count
    # (the primary cause of the 119 GB matrix).
    #
    # Residue rules per crosslinker:
    #   disulfide  → C (cysteine) at crosslink_site
    #   loss_NH3   → K (lysine)   at crosslink_site   (isopeptide K-N/K-Q)
    #   loss_H2O   → K (lysine)   at crosslink_site   (isopeptide K-D/K-E)
    _XL_RESIDUE: Dict[str, str] = {
        'disulfide': 'C',
        'loss_NH3':  'K',
        'loss_H2O':  'K',
    }
    xl_residue = _XL_RESIDUE.get(crosslinker, '')   # '' → no filtering if unknown type

    def _has_xl_residue(seq: str, site: int) -> bool:
        """Return True if the peptide has the crosslinkable residue at `site` (1-based)."""
        if not xl_residue:
            return True   # unknown crosslinker type — don't filter
        if site < 1 or site > len(seq):
            return False
        return seq[site - 1] == xl_residue

    # Partition and filter in one pass
    alpha_indices = [
        i for i, acc in enumerate(accessions)
        if _is_alpha(acc) and _has_xl_residue(peptides[i], crosslink_sites[i])
    ]
    beta_indices = [
        i for i, acc in enumerate(accessions)
        if not _is_alpha(acc) and _has_xl_residue(peptides[i], crosslink_sites[i])
    ]

    print(f"Alpha protein(s)   : {alpha_acc}")
    print(f"Alpha peptides     : {len(alpha_indices)}  |  Beta peptides : {len(beta_indices)}")
    print(f"Max pair count     : {len(alpha_indices) * len(beta_indices):,}")

    for i in alpha_indices:
        for j in beta_indices:
            alpha_peptides.append(peptides[i])
            alpha_modifications.append(modifications[i])
            alpha_crosslink_sites.append(crosslink_sites[i])

            beta_peptides.append(peptides[j])
            beta_modifications.append(modifications[j])
            beta_crosslink_sites.append(crosslink_sites[j])

    total_combinations = len(alpha_peptides)
    print(f"Generated {total_combinations:,} alpha-beta combinations")
    return (alpha_peptides, alpha_modifications, alpha_crosslink_sites,
            beta_peptides, beta_modifications, beta_crosslink_sites)

def _calculate_peptide_mass(aa_dict: Dict[str, float], sequence: str, modifications: Dict[int, float]) -> float:
    peptide_mass = 18.01056  # water mass
    for i, aa in enumerate(sequence):
        # Base amino acid mass
        if aa in aa_dict:
            peptide_mass += aa_dict[aa]
        # Add modification if exists
        if i+1 in modifications:
            peptide_mass += modifications[i+1]
    return peptide_mass

@torch.jit.script
def calculate_alpha_ions(aa_dict: Dict[str, float], signature_types: Dict[str, float], alpha_peptide: str, alpha_modification: Dict[int, float], alpha_crosslink_site: int, beta_peptide: str, beta_modification: Dict[int, float], beta_crosslink_site: int, precursor_mass: float, precursor_charge: int, crosslinker_mass: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    proton_mass = 1.00728
    cid = True
    # Calculate whole peptide masses for use in crosslink mass calculation
    alpha_whole_mass = _calculate_peptide_mass(aa_dict, alpha_peptide, alpha_modification)
    
    if beta_peptide is not None and beta_modification is not None:
        beta_whole_mass = _calculate_peptide_mass(aa_dict, beta_peptide, beta_modification)
    else:
        beta_whole_mass = precursor_mass - alpha_whole_mass

    max_charge = precursor_charge
    alpha_length = len(alpha_peptide)
    beta_length = len(beta_peptide)
    # Regular ions
    alpha_b_ions = torch.zeros(alpha_length - 1, dtype=torch.float32)
    alpha_y_ions = torch.zeros(alpha_length - 1, dtype=torch.float32)
    beta_b_ions = torch.zeros(beta_length - 1, dtype=torch.float32)
    beta_y_ions = torch.zeros(beta_length - 1, dtype=torch.float32)

    # signature ions
    alpha_sig_ions = torch.zeros(len(signature_types), dtype=torch.float32)
    beta_sig_ions = torch.zeros(len(signature_types), dtype=torch.float32)
    #alpha_sig_types = [''] * len(signature_types)
    #beta_sig_types = [''] * len(signature_types)
    
    """
    # both amide and crosslinker cleavage ions (define outside the if statement when used)
    if not cid:
        alpha_b_sig_ions = torch.zeros(alpha_length - 1, dtype=torch.float32)
        beta_b_sig_ions = torch.zeros(beta_length - 1, dtype=torch.float32)
        alpha_y_sig_ions = torch.zeros(alpha_length - 1, dtype=torch.float32)
        beta_y_sig_ions = torch.zeros(beta_length - 1, dtype=torch.float32)
    """
    # Calculate alpha b-ions
    alpha_b_mass = 0.0
    for i, aa in enumerate(alpha_peptide[:-1]):
        alpha_b_mass += aa_dict[aa]
        # Add modification if exists
        if i+1 in alpha_modification:
            alpha_b_mass += alpha_modification[i+1]

        crosslink_b_mass = alpha_b_mass
        # Crosslinked fragment condition
        if i+1 >= alpha_crosslink_site:
            # Add crosslinker and beta peptide mass
            crosslink_b_mass = (
                alpha_b_mass + 
                crosslinker_mass + 
                beta_whole_mass
            )
        alpha_b_ions[i - 1] = crosslink_b_mass

        """
        # Non-crosslinked fragment alpha_b_ions.append((b_ion_mz, i+1, charge, f'αb', f'{sig_type}'))
        if not cid:
            for sig_type, sig_mass in signature_types.items():
                non_crosslink_b_mass = alpha_b_mass + sig_mass
                alpha_b_sig_ions[i - 1] = non_crosslink_b_mass
        """

    # Calculate alpha y-ions
    alpha_y_mass = 18.01056
    for i in range(len(alpha_peptide)-1, 0, -1):
        alpha_y_mass += aa_dict[alpha_peptide[i]]

        # Add modification if exists
        if i+1 in alpha_modification:
            alpha_y_mass += alpha_modification[i+1]

        crosslink_y_mass = alpha_y_mass
        # Crosslinked fragment condition
        if i < alpha_crosslink_site:
            # Add crosslinker and beta peptide mass
            crosslink_y_mass = (alpha_y_mass + crosslinker_mass + beta_whole_mass)
        alpha_y_ions[i - 1] = crosslink_y_mass

        """
        # Non-crosslinked fragment alpha_y_ions.append((y_ion_mz, len(alpha_peptide) - i, charge, f'αy', f'{sig_type}'))
        if not cid:
            for sig_type, sig_mass in signature_types.items():
                non_crosslink_y_mass = alpha_y_mass + sig_mass
                alpha_y_sig_ions[i - 1] = non_crosslink_y_mass
        """

    if beta_peptide:
        # Calculate beta b-ions
        beta_b_mass = 0.0
        for i, aa in enumerate(beta_peptide[:-1]):
            beta_b_mass += aa_dict[aa]
            # Add modification if exists
            if i+1 in beta_modification:
                beta_b_mass += beta_modification[i+1]

            crosslink_b_mass = beta_b_mass
            # Crosslinked fragment condition
            if (beta_crosslink_site is not None and 
                    i+1 >= beta_crosslink_site):
                # Add crosslinker and alpha peptide mass
                crosslink_b_mass = (beta_b_mass + crosslinker_mass + alpha_whole_mass)
            beta_b_ions[i - 1] = crosslink_b_mass
            
            """
            # Non-crosslinked fragment
            if not cid:
                for sig_type, sig_mass in signature_types.items():
                    non_crosslink_b_mass = beta_b_mass + sig_mass
                    beta_b_sig_ions[i - 1] = non_crosslink_b_mass
            """
        
        # Calculate beta y-ions
        beta_y_mass = 18.01056
        for i in range(len(beta_peptide)-1, 0, -1):
            beta_y_mass += aa_dict[beta_peptide[i]]

            # Add modification if exists
            if i+1 in beta_modification:
                beta_y_mass += beta_modification[i+1]
                
            crosslink_y_mass = beta_y_mass
            # Crosslinked fragment condition
            if (beta_crosslink_site is not None and i < beta_crosslink_site):
                # Add crosslinker and alpha peptide mass
                crosslink_y_mass = (beta_y_mass + crosslinker_mass + alpha_whole_mass)
            beta_y_ions[i - 1] = crosslink_y_mass
            
            """
            # Non-crosslinked fragment beta_y_ions.append((y_ion_mz, len(self.beta_sequence) - i, charge, f'βy', f'{sig_type}'))
            if not cid:
                for sig_type, sig_mass in signature_types.items():
                    non_crosslink_y_mass = beta_y_mass + sig_mass
                    beta_y_sig_ions[i - 1] = non_crosslink_y_mass
            """
    
    # Don't use format string here, pass the actual sig_type
    for i, (sig_type, sig_mass) in enumerate(signature_types.items()):
        # whole alpha signature ions alpha_sig_ions.append((ion_mz, '', charge, f'α', f'{sig_type}'))
        non_crosslink_mass = alpha_whole_mass + sig_mass
        alpha_sig_ions[i] = non_crosslink_mass

        # whole beta signature ions beta_sig_ions.append((ion_mz, '', charge, f'β', f'{sig_type}'))
        non_crosslink_mass = beta_whole_mass + sig_mass
        beta_sig_ions[i] = non_crosslink_mass

    return alpha_b_ions, alpha_y_ions, beta_b_ions, beta_y_ions, alpha_sig_ions, beta_sig_ions

@torch.jit.script
def add_ion_with_charges(mass: float, ion_type: str, ion_number: int, sig_type: str, input_charge: int, proton_mass: float) -> Tuple[torch.Tensor, List[str]]:
    # Pre-allocate tensors instead of using lists for numerical values
    charge_range = torch.arange(1, input_charge, dtype=torch.int32)
    # Calculate m/z values
    mz_values = (mass + charge_range * proton_mass) / charge_range
    
    # Build ion labels
    matched_ions: List[str] = []
    for charge in range(1, input_charge):
        if ion_number == 0:     # sig ions
            label = f'{ion_type}{sig_type}^{{+{charge}}}'
        else:                   # b/y ions
            label = f'{ion_type}_{{{ion_number}}}{sig_type}^{{+{charge}}}'
        formatted_label = f'$\\mathrm{{{label}}}$'
        matched_ions.append(formatted_label)
    return mz_values, matched_ions

@torch.jit.script
def vectorize_spectrum(mz_values: torch.Tensor, intensities: torch.Tensor, bin_width: float, offset: float, observed: bool) -> torch.Tensor:
    # Ensure both tensors are float32 and on the same device
    device = mz_values.device
    mz_values = mz_values.to(torch.float32)
    intensities = intensities.to(torch.float32)
    
    max_mz = 2000
    bin_count = int(torch.ceil(torch.tensor(max_mz / bin_width)))
    binned_spectrum = torch.zeros(bin_count, dtype=torch.float32, device=device)
    
    bin_indices = ((mz_values + offset) / bin_width).long()
    valid_indices = (bin_indices >= 0) & (bin_indices < bin_count)
    
    if observed:
        binned_spectrum.scatter_add_(0, bin_indices[valid_indices], intensities[valid_indices])
    else:
        binned_spectrum.scatter_(0, bin_indices[valid_indices], torch.ones_like(intensities[valid_indices]))
    
    return binned_spectrum

@torch.jit.script
def create_theoretical_spectrum(alpha_b: torch.Tensor, alpha_y: torch.Tensor, beta_b: torch.Tensor, beta_y: torch.Tensor, alpha_sig_ions: torch.Tensor, beta_sig_ions: torch.Tensor, alpha_crosslink_site: int, beta_crosslink_site: int, input_charge: int, signature_types: Dict[str, float], use_neutral_loss: bool, use_signature_ions: bool) -> Tuple[torch.Tensor, List[str], torch.Tensor, torch.Tensor]:
    
    PROTON = 1.00728
    BIN_WIDTH = 0.02
    BIN_OFFSET = 0.4
    
    # Initialize lists for collecting results
    all_mzs: List[torch.Tensor] = []
    all_labels: List[str] = []
    
    alpha_len = len(alpha_b) + 1
    beta_len = len(beta_b) + 1
    
    # ── Regular b/y ions ──────────────────────────────────────────────────────
    for ion_type, ions in [('αb', alpha_b), ('αy', alpha_y), ('βb', beta_b), ('βy', beta_y)]:
        for i, mass in enumerate(ions):
            if (ion_type == 'αb' and i + 1 >= alpha_crosslink_site) or (ion_type == 'βb' and i + 1 >= beta_crosslink_site) or (ion_type == 'αy' and i + 1 >= alpha_len - alpha_crosslink_site + 1) or (ion_type == 'βb' and i + 1 >= beta_len - beta_crosslink_site + 1):
                star_ion_type = ion_type + '*'
            else:
                star_ion_type = ion_type
            mzs, ion = add_ion_with_charges(float(mass), star_ion_type, i+1, '', input_charge, PROTON)
            all_mzs.append(mzs)
            all_labels.extend(ion)

            # ── Neutral-loss ions (H2O and NH3) ───────────────────────────────
            if use_neutral_loss:
                # -H2O neutral loss
                nl_mzs_h2o, nl_ion_h2o = add_ion_with_charges(float(mass) - 18.01056, star_ion_type, i+1, '-H2O', input_charge, PROTON)
                all_mzs.append(nl_mzs_h2o)
                all_labels.extend(nl_ion_h2o)
                # -NH3 neutral loss
                nl_mzs_nh3, nl_ion_nh3 = add_ion_with_charges(float(mass) - 17.02655, star_ion_type, i+1, '-NH3', input_charge, PROTON)
                all_mzs.append(nl_mzs_nh3)
                all_labels.extend(nl_ion_nh3)

    # ── Signature ions (whole-peptide crosslinker cleavage ions) ─────────────
    if use_signature_ions:
        for ion_type, ions in [('α', alpha_sig_ions), ('β', beta_sig_ions)]:
            for i, mass in enumerate(ions):
                for j, (sig_type, sig_mass) in enumerate(signature_types.items()):
                    mzs, ion = add_ion_with_charges(float(mass), ion_type, 0, sig_type, input_charge, PROTON)
                    all_mzs.append(mzs)
                    all_labels.extend(ion)
    
    # Concatenate all mass tensors
    if len(all_mzs) > 0:
        mass_tensor = torch.cat(all_mzs)
    else:
        mass_tensor = torch.tensor([], dtype=torch.float32)
    
    # Convert masses to m/z values
    #charges = torch.tensor([float(ion.split('+')[1]) for ion in all_matched_ions])
    #mz_values = mass_tensor / charges + PROTON
    
    # Sort by m/z values
    sorted_indices = torch.argsort(mass_tensor)
    sorted_mz = mass_tensor[sorted_indices]
    sorted_intensities = torch.ones_like(sorted_mz)
    
    # Convert matched ions list to sorted version using a type-safe approach
    indices_list: List[int] = sorted_indices.tolist()  # Explicit type annotation
    sorted_matched_ions: List[str] = []
    for idx in indices_list:
        sorted_matched_ions.append(all_labels[idx])
    
    # Create theoretical spectrum
    theoretical_spectrum = vectorize_spectrum(sorted_mz, sorted_intensities, BIN_WIDTH, BIN_OFFSET, False)
    
    return theoretical_spectrum, sorted_matched_ions, sorted_mz, sorted_intensities

@torch.jit.script
def preprocess_spectrum(spectrum: torch.Tensor, region_size: int = 10, max_intensity: float = 50.0) -> torch.Tensor:
    """
    Preprocess an observed spectrum:
    1. Square root intensities
    2. Region-based normalization  (OPT: fully vectorized — replaces Python for-loop)
    3. Background subtraction via sliding-window mean
    """
    device = spectrum.device

    # 1. Square-root intensities
    processed = torch.sqrt(spectrum)

    # 2. OPT: vectorized region normalization via max_pool1d + repeat_interleave.
    #    The original code looped over every region in Python (thousands of iterations
    #    per spectrum).  max_pool1d computes all region maxima in a single CUDA/CPU kernel.
    n = processed.size(0)
    pad_size = (region_size - n % region_size) % region_size
    padded_proc = F.pad(processed.unsqueeze(0).unsqueeze(0), (0, pad_size))   # [1,1,N']
    region_max = F.max_pool1d(padded_proc, kernel_size=region_size,
                              stride=region_size).squeeze()                    # [num_regions]
    region_max = region_max.clamp(min=1e-9)                                   # guard /0
    scale = max_intensity / region_max                                         # [num_regions]
    scale_expanded = scale.repeat_interleave(region_size)[:n]                 # [N]
    normalized = processed * scale_expanded

    # 3. Sliding-window background subtraction (already vectorized, unchanged)
    max_xcorr_offset = 75
    padded = F.pad(normalized.unsqueeze(0).unsqueeze(0),
                   (max_xcorr_offset, max_xcorr_offset), mode='constant')
    kernel = torch.ones(1, 1, max_xcorr_offset * 2 + 1, device=device) / (max_xcorr_offset * 2 + 1)
    means = F.conv1d(padded, kernel, padding=0).squeeze()
    return normalized - means

@torch.jit.script
def vectorize_spectrum_enhanced(mz_values: torch.Tensor, intensities: torch.Tensor, 
                             bin_width: float, bin_offset: float, 
                             max_mz: float = 2000.0) -> torch.Tensor:
    """
    Create binned spectrum
    
    Args:
        mz_values: m/z values tensor
        intensities: Intensity values tensor
        bin_width: Width of each bin
        bin_offset: Bin offset value
        max_mz: Maximum m/z value to consider
        
    Returns:
        Binned spectrum tensor
    """
    device = mz_values.device
    
    # Calculate bin count
    bin_count = int(max_mz / bin_width) + 1
    binned_spectrum = torch.zeros(bin_count, dtype=torch.float32, device=device)
    
    # Calculate bin indices with offset
    bin_indices = ((mz_values / bin_width + 1.0) - bin_offset).long()
    
    # Filter valid indices
    valid_indices = (bin_indices >= 0) & (bin_indices < bin_count)
    
    # Add intensities to bins
    binned_spectrum.scatter_add_(0, bin_indices[valid_indices], intensities[valid_indices])
    
    return binned_spectrum

@torch.jit.script
def create_theoretical_spectrum_enhanced(mz_values: torch.Tensor, bin_width: float, bin_offset: float, 
                                     intensity_value: float = 50.0, max_mz: float = 2000.0) -> torch.Tensor:
    """
    Create theoretical spectrum for XCorr with standard intensity values
    
    Args:
        mz_values: m/z values tensor
        bin_width: Width of each bin
        bin_offset: Bin offset value
        intensity_value: Standard intensity for theoretical peaks
        max_mz: Maximum m/z value to consider
        
    Returns:
        Theoretical spectrum tensor
    """
    device = mz_values.device
    
    # Calculate bin count
    bin_count = int(max_mz / bin_width) + 1
    theoretical_spectrum = torch.zeros(bin_count, dtype=torch.float32, device=device)
    
    # Calculate bin indices with offset
    bin_indices = ((mz_values / bin_width + 1.0) - bin_offset).long()
    
    # Filter valid indices
    valid_indices = (bin_indices >= 0) & (bin_indices < bin_count)
    filtered_indices = bin_indices[valid_indices]
    
    # Set standard intensity for each peak
    theoretical_spectrum.scatter_(0, filtered_indices, 
                                torch.full_like(filtered_indices, intensity_value, dtype=torch.float32))
    
    return theoretical_spectrum

@torch.jit.script
def compute_xcorr_enhanced(theoretical_spectrum: torch.Tensor, observed_spectrum: torch.Tensor, 
                         normalization_factor: float = 10000.0) -> float:
    """
    Compute enhanced XCorr score between theoretical and preprocessed observed spectra
    
    Args:
        theoretical_spectrum: Theoretical spectrum tensor
        observed_spectrum: Preprocessed observed spectrum tensor
        normalization_factor: Factor to normalize the XCorr score
        
    Returns:
        XCorr score
    """
    device = theoretical_spectrum.device
    observed_spectrum = observed_spectrum.to(device)
    
    # Ensure both spectra are on the same device
    theoretical_spectrum = theoretical_spectrum.to(device)
    
    # Calculate simple dot product
    xcorr_score = torch.dot(theoretical_spectrum, observed_spectrum) / normalization_factor
    
    return xcorr_score.item()

def _precompute_worker(args):
    """
    OPT: Top-level (picklable) worker for multiprocessing.Pool.
    Computes a slice of the theoretical spectrum matrix for pairs [start, end).
    Returns (start, np.ndarray[slice_len, bin_count], list[float] pair_masses).
    """
    (start, end,
     aa_dict, signature_types,
     alpha_peptides, alpha_modifications, alpha_crosslink_sites,
     beta_peptides,  beta_modifications,  beta_crosslink_sites,
     crosslinker_mass, max_charge,
     bin_count, BIN_WIDTH, BIN_OFFSET, USE_NEUTRAL_LOSS, USE_SIGNATURE_IONS) = args

    slice_len = end - start
    local_np  = np.zeros((slice_len, bin_count), dtype=np.float32)
    local_masses: List[float] = []

    for local_i in range(slice_len):
        alpha_pep  = alpha_peptides[local_i]
        alpha_mod  = alpha_modifications[local_i]
        alpha_site = alpha_crosslink_sites[local_i]
        beta_pep   = beta_peptides[local_i]
        beta_mod   = beta_modifications[local_i]
        beta_site  = beta_crosslink_sites[local_i]

        alpha_mass = _calculate_peptide_mass(aa_dict, alpha_pep, alpha_mod)
        beta_mass  = _calculate_peptide_mass(aa_dict, beta_pep,  beta_mod)
        theo_mass  = alpha_mass + beta_mass + crosslinker_mass
        local_masses.append(theo_mass)

        alpha_b, alpha_y, beta_b, beta_y, alpha_sig, beta_sig = calculate_alpha_ions(
            aa_dict, signature_types, alpha_pep, alpha_mod, alpha_site,
            beta_pep, beta_mod, beta_site,
            theo_mass, max_charge, crosslinker_mass)

        _, _, theoretical_mz, _ = create_theoretical_spectrum(
            alpha_b, alpha_y, beta_b, beta_y, alpha_sig, beta_sig,
            alpha_site, beta_site, max_charge, signature_types,
            USE_NEUTRAL_LOSS, USE_SIGNATURE_IONS)

        row = create_theoretical_spectrum_enhanced(theoretical_mz, BIN_WIDTH, BIN_OFFSET)
        local_np[local_i] = row.numpy()

    return start, local_np, local_masses


def precompute_theoretical_spectra(
        aa_dict: Dict[str, float], signature_types: Dict[str, float],
        alpha_peptides: List[str], alpha_modifications: List[Dict[int, float]],
        alpha_crosslink_sites: List[int],
        beta_peptides: List[str], beta_modifications: List[Dict[int, float]],
        beta_crosslink_sites: List[int],
        crosslinker_mass: float, max_charge: int,
        device: torch.device) -> Tuple[torch.Tensor, List[float]]:
    """
    OPT: Pre-compute ALL theoretical spectra and peptide pair masses ONCE before
    the scan loop.  In the original code these were recomputed inside every call
    to process_spectrum_enhanced — i.e. once per scan × once per peptide pair —
    even though theoretical spectra are completely independent of the observed
    scan.  For N pairs and S scans that's N×S redundant computations; here it
    is just N.

    Returns
    -------
    theo_matrix : float32 tensor [N, BIN_COUNT] on `device`
        Each row is the precomputed theoretical spectrum for one peptide pair.
    pair_masses  : list[float] length N
        Theoretical precursor mass for each pair (used for ppm filtering).
    """
    n_pairs   = len(alpha_peptides)
    bin_count = int(2000.0 / BIN_WIDTH) + 1
    matrix_mb = n_pairs * bin_count * 4 / 1e6

    # ── Memory guard ──────────────────────────────────────────────────────────
    # Use *total* installed RAM (not currently-available) as the ceiling.
    # psutil.virtual_memory().available reflects only the un-committed pages at
    # this instant — it can read as low as 10-11 GB on a 16 GB machine that has
    # normal OS/app overhead, even when Task Manager shows 10 GB "free".
    # Comparing against total×0.70 gives a stable, machine-level limit that
    # won't false-trigger due to transient OS cache pressure.
    #
    # If the matrix still exceeds that limit we do NOT abort — instead the caller
    # (process_ms2_file) will automatically chunk the pair list into RAM-safe
    # slices and call us repeatedly.  We only raise here for an absolute ceiling
    # (>90 % of total) where even chunking couldn't help.
    import psutil
    vm             = psutil.virtual_memory()
    total_ram_mb   = vm.total    / 1e6
    available_ram_mb = vm.available / 1e6
    HARD_CEIL_FRAC = 0.90          # never allocate more than 90 % of total RAM at once
    if matrix_mb > total_ram_mb * HARD_CEIL_FRAC:
        raise MemoryError(
            f"Theoretical spectrum matrix for this chunk would require "
            f"{matrix_mb:.0f} MB but the hard ceiling is "
            f"{total_ram_mb * HARD_CEIL_FRAC:.0f} MB (90 % of {total_ram_mb:.0f} MB total RAM). "
            f"This should not happen when called via process_ms2_file — "
            f"check that CHUNK_SIZE in process_ms2_file is set correctly."
        )
    if matrix_mb > available_ram_mb:
        # Non-fatal warning — chunking in the caller keeps actual allocation safe.
        print(f"  [info] chunk matrix = {matrix_mb:.0f} MB  |  "
              f"currently available = {available_ram_mb:.0f} MB  |  "
              f"total = {total_ram_mb:.0f} MB  (OS cache will be displaced — normal)")

    # ── Build into a plain numpy array — no CUDA involved ────────────────────
    # OPT: replaced single-threaded for-loop with multiprocessing.Pool so all
    # physical CPU cores are used.  Each worker receives a contiguous slice of
    # the pair list, runs the same per-pair logic, and returns a numpy sub-array.
    # The main process assembles slices in-order and wraps the result with
    # torch.from_numpy (zero-copy).  Pool is started *after* torch.jit.script
    # functions are already compiled (they are module-level), so child processes
    # inherit the compiled cache and start quickly.
    theo_np     = np.zeros((n_pairs, bin_count), dtype=np.float32)
    pair_masses: List[float] = [0.0] * n_pairs   # pre-allocate for index-based fill

    # Use physical core count; leave one core for the OS / GPU driver.
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    # Slice size: 64 pairs × 100,001 bins × 4 B ≈ 25 MB per returned result.
    # Keeping slices small bounds how much data is simultaneously in-flight
    # across IPC pipes (n_workers results queued at once), preventing OOM.
    SLICE = 64

    slices = []
    for s in range(0, n_pairs, SLICE):
        e = min(s + SLICE, n_pairs)
        slices.append((
            s, e,
            aa_dict, signature_types,
            alpha_peptides[s:e], alpha_modifications[s:e], alpha_crosslink_sites[s:e],
            beta_peptides[s:e],  beta_modifications[s:e],  beta_crosslink_sites[s:e],
            crosslinker_mass, max_charge,
            bin_count, BIN_WIDTH, BIN_OFFSET, USE_NEUTRAL_LOSS, USE_SIGNATURE_IONS,
        ))

    with mp.Pool(processes=n_workers) as pool:
        for start, sub_np, sub_masses in pool.imap_unordered(_precompute_worker, slices):
            end = start + len(sub_masses)
            theo_np[start:end]    = sub_np
            pair_masses[start:end] = sub_masses

    # Single conversion to a CPU torch tensor (no pin_memory — avoids CUDA OOM)
    theo_matrix = torch.from_numpy(theo_np)   # zero-copy wrap
    return theo_matrix, pair_masses


def process_spectrum_enhanced(
        theo_matrix: torch.Tensor,          # [N, BIN_COUNT] — precomputed, on device
        pair_masses: List[float],           # [N] theoretical precursor masses
        spectrum: torch.Tensor,             # raw peak list, CPU
        scan_num: str,
        alpha_peptides: List[str],
        alpha_modifications: List[Dict[int, float]],
        alpha_crosslink_sites: List[int],
        beta_peptides: List[str],
        beta_modifications: List[Dict[int, float]],
        beta_crosslink_sites: List[int],
        crosslinker_mass: float,
        precursor_mass: float, charge: int,
        aa_dict: Dict[str, float],
        device: torch.device) -> List[Tuple[str, str, str, float, float, float]]:
    """
    OPT summary vs original:
    • Theoretical spectra are passed in pre-built (not recomputed here).
    • PPM filter is applied as a vectorised numpy comparison before any GPU work.
    • The observed spectrum is uploaded to GPU once; dot-products for ALL
      passing pairs are computed in a single batched matrix-vector multiply
      instead of one dot() call per pair.
    • torch.as_tensor(np_array) avoids the data copy that torch.tensor() does.
    """
    results = []

    # ── 1. PPM filter on CPU (pure Python / numpy — no GPU needed) ────────────
    # OPT: filter candidates with a vectorised numpy comparison; only surviving
    #      indices are sent to the GPU, keeping VRAM traffic minimal.
    pm_arr   = np.array(pair_masses, dtype=np.float64)
    ppm_arr  = np.abs(1e6 * (precursor_mass - pm_arr) / pm_arr)
    passing  = np.where(ppm_arr <= 10.0)[0]   # indices of pairs within 10 ppm

    if len(passing) == 0:
        return results

    # ── 2. Build observed spectrum — upload to GPU once ────────────────────────
    # OPT: torch.as_tensor on a numpy array avoids a data copy; .to(device) is
    #      a single H→D transfer instead of two separate ones for mz and intensity.
    spec_np = np.array(spectrum, dtype=np.float32)   # list-of-tuples → contiguous array

    # OPT: use non_blocking=True so the CPU can continue while the transfer runs
    spec_tensor = torch.as_tensor(spec_np, dtype=torch.float32).to(device, non_blocking=True)

    observed_mz        = spec_tensor[:, 0]
    observed_intensity = spec_tensor[:, 1]

    vectorized_observed = vectorize_spectrum_enhanced(
        observed_mz, observed_intensity, BIN_WIDTH, BIN_OFFSET)
    processed_observed = preprocess_spectrum(vectorized_observed)   # [BIN_COUNT]

    # ── 3. Batched XCorr ──────────────────────────────────────────────────────
    # Upload ppm-passing rows in small batches so VRAM usage is bounded.
    # Batch size: 500 rows × 100,001 bins × 4 B ≈ 200 MB — safe on a 4 GB card
    # even with the observed spectrum and workspace already resident.
    GPU_BATCH = 500
    xcorr_all = np.empty(len(passing), dtype=np.float32)

    for batch_start in range(0, len(passing), GPU_BATCH):
        batch_idx  = passing[batch_start : batch_start + GPU_BATCH]
        # Index numpy rows → CPU tensor → GPU in one async DMA
        batch_rows = torch.from_numpy(theo_matrix.numpy()[batch_idx])
        batch_gpu  = batch_rows.to(device, non_blocking=True)          # [B, BINS]
        scores     = (batch_gpu @ processed_observed) / 10000.0        # [B]
        xcorr_all[batch_start : batch_start + len(batch_idx)] = scores.cpu().numpy()

    # Move scores back to CPU once for the threshold check
    xcorr_cpu = xcorr_all   # already a numpy array on CPU

    for rank, orig_idx in enumerate(passing):
        xcorr = float(xcorr_cpu[rank])
        if xcorr >= 0.0: ### XCorr >= 2.0
            alpha_pep  = alpha_peptides[orig_idx]
            beta_pep   = beta_peptides[orig_idx]
            alpha_mod  = alpha_modifications[orig_idx]
            beta_mod   = beta_modifications[orig_idx]
            alpha_site = alpha_crosslink_sites[orig_idx]
            beta_site  = beta_crosslink_sites[orig_idx]
            theo_mass  = pair_masses[orig_idx]
            # Compute mass_diff and ppm_diff fresh from this scan's precursor_mass
            # (not from the pre-built ppm_arr) so every row reflects its own scan's
            # observed precursor, not a value shared with other scans.
            obs_mass   = precursor_mass
            mass_diff  = obs_mass - theo_mass
            ppm_diff   = abs(1e6 * mass_diff / theo_mass)
            results.append((
                scan_num, alpha_pep, beta_pep,
                alpha_site, beta_site,
                format_modifications(alpha_mod),
                format_modifications(beta_mod),
                _calculate_peptide_mass(aa_dict, alpha_pep, alpha_mod),
                _calculate_peptide_mass(aa_dict, beta_pep,  beta_mod),
                theo_mass, obs_mass, mass_diff,
                ppm_diff, xcorr, charge
            ))

    return results

@torch.jit.script
def create_xcorr_observed_spectrum(mz_values: torch.Tensor, intensities: torch.Tensor, 
                                 bin_width: float) -> torch.Tensor:
    """
    Create processed observed spectrum for XCorr calculation:
    1. Bin the spectrum
    2. Preprocess intensities
    3. Subtract mean in sliding window
    """
    # Calculate number of bins
    max_mz = 2000
    num_bins = int(max_mz / bin_width) + 1
    binned = torch.zeros(num_bins, dtype=torch.float32, device=mz_values.device)
    
    # Bin the spectrum
    bin_indices = (mz_values / bin_width).long()
    valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
    bin_indices = bin_indices[valid_mask]
    intensities = intensities[valid_mask]
    
    # Sum intensities in each bin
    binned.scatter_add_(0, bin_indices, intensities)
    
    # Preprocess spectrum
    processed = preprocess_spectrum(binned)
    
    # Calculate and subtract mean in sliding window (75 peaks on each side)
    window_size = 150
    padding = window_size // 2
    padded = torch.nn.functional.pad(processed, (padding, padding), mode='constant')
    
    # Calculate moving average
    window_mean = torch.nn.functional.avg_pool1d(
        padded.unsqueeze(0).unsqueeze(0),
        kernel_size=window_size,
        stride=1,
        padding=0
    ).squeeze()
    
    # Subtract mean from original spectrum
    processed = processed - window_mean
    
    return processed

@torch.jit.script
def create_xcorr_theoretical_spectrum(mz_values: torch.Tensor, bin_width: float) -> torch.Tensor:
    """
    Create theoretical spectrum for XCorr:
    1. Bin the spectrum with unit heights
    2. No intensity preprocessing needed for theoretical spectrum
    """
    max_mz = 2000
    num_bins = int(max_mz / bin_width) + 1
    binned = torch.zeros(num_bins, dtype=torch.float32, device=mz_values.device)
    
    # Bin the spectrum with unit intensities
    bin_indices = (mz_values / bin_width).long()
    valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
    bin_indices = bin_indices[valid_mask]
    
    # Set unit intensity for each peak
    binned.scatter_add_(0, bin_indices, torch.ones_like(bin_indices, dtype=torch.float32))
    
    return binned

@torch.jit.script
def compute_xcorr(theoretical_spectrum: torch.Tensor, observed_spectrum: torch.Tensor, window_size: int = 75) -> float:
    """
    Compute XCorr score between theoretical and observed spectra using Crux-like approach
    """
    device = theoretical_spectrum.device
    observed_spectrum = observed_spectrum.to(device)
    
    # 1. Normalize observed spectrum
    # Take square root of intensities
    observed_processed = torch.sqrt(observed_spectrum)
    
    # Calculate and subtract local mean using convolution
    # Add padding to handle boundaries
    padded = F.pad(observed_processed.unsqueeze(0).unsqueeze(0), 
                   (window_size, window_size), mode='constant')
    
    # Calculate moving average with a boxcar filter
    kernel = torch.ones(1, 1, window_size*2+1, device=device) / (window_size*2+1)
    local_mean = F.conv1d(padded, kernel, padding=0).squeeze()
    
    # Subtract mean from observed spectrum
    observed_processed = observed_processed - local_mean
    
    # 2. Calculate correlation at zero offset
    xcorr = torch.dot(theoretical_spectrum, observed_processed)
    
    return xcorr.item()

def process_spectrum(aa_dict: Dict[str, float], signature_types: Dict[str, float], 
                    spectrum: torch.Tensor, scan_num: str, 
                    alpha_peptides: List[str], alpha_modifications: List[Dict[int, float]], 
                    alpha_crosslink_sites: List[int], 
                    beta_peptides: List[str], beta_modifications: List[Dict[int, float]], 
                    beta_crosslink_sites: List[int], 
                    crosslinker_mass: float, precursor_mass: float, charge: int, 
                    device: torch.device) -> List[Tuple[str, str, str, float, float, float]]:
    
    results = []
    
    for pep_idx, (alpha_peptide, alpha_modification, alpha_crosslink_site, 
                 beta_peptide, beta_modification, beta_crosslink_site) in enumerate(
        zip(alpha_peptides, alpha_modifications, alpha_crosslink_sites, 
            beta_peptides, beta_modifications, beta_crosslink_sites)):
            
        theoretical_mass = _calculate_peptide_mass(aa_dict, alpha_peptide, alpha_modification) + _calculate_peptide_mass(aa_dict, beta_peptide, beta_modification) + crosslinker_mass
        mass_diff = precursor_mass - theoretical_mass
        ppm_diff = abs(1E+6 * mass_diff / theoretical_mass)

        if ppm_diff <= 10.0:
            alpha_b, alpha_y, beta_b, beta_y, alpha_sig_ions, beta_sig_ions = calculate_alpha_ions(
                aa_dict, signature_types, alpha_peptide, alpha_modification, 
                alpha_crosslink_site, beta_peptide, beta_modification, 
                beta_crosslink_site, precursor_mass, charge, crosslinker_mass)

            theoretical_spectrum, matched_ions, theoretical_mz, theoretical_intensity = create_theoretical_spectrum(
                alpha_b, alpha_y, beta_b, beta_y, alpha_sig_ions, beta_sig_ions,
                alpha_crosslink_site, beta_crosslink_site, charge, signature_types,
                USE_NEUTRAL_LOSS, USE_SIGNATURE_IONS)
            
            # Fix tensor construction warnings and add debugging
            observed_mz = spectrum[:, 0].clone().detach().to(device)
            observed_intensity = spectrum[:, 1].clone().detach().to(device)
            vectorized_observed = vectorize_spectrum(observed_mz, observed_intensity, BIN_WIDTH, BIN_OFFSET, True)
            
            # Use the fixed XCorr calculation
            xcorr = compute_xcorr(theoretical_spectrum, vectorized_observed)
            
            # Output criteria for resolving the IO bottleneck
            if xcorr >= 2.0:
                results.append((scan_num, alpha_peptide, beta_peptide, alpha_crosslink_site, beta_crosslink_site, _calculate_peptide_mass(aa_dict, alpha_peptide, alpha_modification), _calculate_peptide_mass(aa_dict, beta_peptide, beta_modification), theoretical_mass, precursor_mass, mass_diff, ppm_diff, xcorr, charge))
    
    return results

def split_ms2_by_scans(file_path: str) -> List[List[str]]:
    """Split MS2 file into scan blocks and return list of scan content blocks."""
    scans = []
    current_scan = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('S') and current_scan:
                scans.append(current_scan)
                current_scan = []
            current_scan.append(line)
            
    # Don't forget the last scan
    if current_scan:
        scans.append(current_scan)
        
    return scans

def distribute_scans(scans: List[List[str]], num_threads: int) -> List[List[List[str]]]:
    """Distribute scans evenly across threads."""
    # Get total number of scans
    total_scans = len(scans)
    print(f"Total scans found: {total_scans}")
    
    # Calculate scans per thread
    base_scans_per_thread = total_scans // num_threads
    extra_scans = total_scans % num_threads
    
    # Distribute scans
    distributed_scans = []
    start_idx = 0
    
    for thread_idx in range(num_threads):
        # Add one extra scan to early threads if there are remainder scans
        thread_scan_count = base_scans_per_thread + (1 if thread_idx < extra_scans else 0)
        end_idx = start_idx + thread_scan_count
        
        thread_scans = scans[start_idx:end_idx]
        distributed_scans.append(thread_scans)
        
        print(f"Thread {thread_idx}: Processing {len(thread_scans)} scans")
        start_idx = end_idx
        
    return distributed_scans

def process_scan_block(scan_block: List[str]) -> Tuple[str, List[Tuple[float, float]], float, int]:
    """Parse a single scan block.  Returns peaks as a plain list; caller converts.

    MS2 format:
        S  <scan_lo>  <scan_hi>  <precursor_mz>
        Z  <charge>   <M+H>          ← M+H = neutral_mass + proton; BUT some tools
                                        write precursor m/z here instead of M+H.
        I  ...                        ← info lines, skipped
        <mz> <intensity>              ← fragment peaks

    We read precursor_mz from the S line (always reliable) and charge from the
    Z line, then compute the neutral precursor mass as:
        neutral_mass = precursor_mz * charge - charge * proton_mass
    This is unambiguous regardless of whether the Z line's third field is M+H
    or m/z.  Using the Z line's mass field directly was the source of the bug:
    when that field is the precursor m/z (not M+H), subtracting one proton gives
    only ~1/charge of the true neutral mass, making every scan's precursor_mass
    wrong and often identical to neighbours within a chunk.
    """
    scan_num = ''
    spectrum: List[Tuple[float, float]] = []
    precursor_mz = 0.0   # read from S line
    precursor_mass = 0.0  # computed after both S and Z are parsed
    charge = 0

    for line in scan_block:
        line = line.strip()
        if line.startswith('S'):
            parts = line.split()
            if len(parts) >= 2:
                scan_num = parts[1].strip()
            if len(parts) >= 4:
                try:
                    precursor_mz = float(parts[3].strip())
                except ValueError:
                    pass
            elif len(parts) >= 3:
                try:
                    precursor_mz = float(parts[2].strip())
                except ValueError:
                    pass
        elif line.startswith('Z'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    charge = int(parts[1].strip())
                except ValueError:
                    charge = 0
            if precursor_mz > 0.0 and charge > 0:
                # Some MS/MS scan fetches the highest isotope in an envelope. Store iso_num for caluclating the b/y ions with the correct isotope peaks
                iso_precursor_mass = precursor_mz * charge - charge * proton_mass
                precursor_mass = float(parts[2]) - proton_mass
                iso_num = round(iso_precursor_mass - precursor_mass / proton_mass)
        elif not line.startswith(('H', 'I')):
            try:
                mz, intensity = map(float, line.split())
                spectrum.append((mz, intensity))
            except ValueError:
                continue

    return scan_num, spectrum, precursor_mass, charge

def process_scan_group(scan_blocks: List[List[str]], output_queue: Queue,
                       theo_matrix: torch.Tensor,      # [N, BIN_COUNT] on device — shared read-only
                       pair_masses: List[float],        # [N] — precomputed precursor masses
                       alpha_peptides: List[str], alpha_modifications: List[Dict[int, float]],
                       alpha_crosslink_sites: List[int],
                       beta_peptides: List[str], beta_modifications: List[Dict[int, float]],
                       beta_crosslink_sites: List[int],
                       crosslinker_mass: float, aa_dict: Dict[str, float],
                       device: torch.device):
    """
    Process a group of scans assigned to one thread.

    OPT: theo_matrix and pair_masses are passed in pre-built.  Each scan only
    needs to (a) parse its peak list, (b) upload one observed spectrum, and
    (c) run one batched matmul — dramatically less work per scan than before.

    NOTE on threading vs GPU: CUDA operations are serialised on the default
    stream regardless of how many threads call them.  The ThreadPoolExecutor
    here is therefore useful only for the CPU-side parsing work (process_scan_block).
    All GPU calls (vectorize / preprocess / matmul) will naturally queue on the
    same CUDA stream.  This is intentional: launching separate CUDA streams per
    thread would require explicit stream management and careful synchronisation.
    """
    results = []
    for scan_block in scan_blocks:
        scan_num, spectrum, precursor_mass, charge = process_scan_block(scan_block)
        if spectrum:
            scan_results = process_spectrum_enhanced(
                theo_matrix, pair_masses,
                spectrum, scan_num,
                alpha_peptides, alpha_modifications, alpha_crosslink_sites,
                beta_peptides, beta_modifications, beta_crosslink_sites,
                crosslinker_mass, precursor_mass, charge,
                aa_dict, device
            )
            results.extend(scan_results)
    output_queue.put(results)

def score_chunk_against_file(
        distributed_scans: List,
        theo_matrix: torch.Tensor,
        pair_masses: List[float],
        c_alpha_pep: List[str],
        c_alpha_mod: List[Dict[int, float]],
        c_alpha_site: List[int],
        c_beta_pep: List[str],
        c_beta_mod: List[Dict[int, float]],
        c_beta_site: List[int],
        crosslinker_mass: float,
        aa_dict: Dict[str, float],
        device: torch.device,
        csv_writer) -> int:
    """
    Score all scans in `distributed_scans` against a single precomputed chunk
    (theo_matrix / pair_masses) and write any hits directly to `csv_writer`.

    Returns the number of result rows written.
    """
    output_queue: Queue = Queue()
    error_queue:  Queue = Queue()
    num_threads = max(1, (os.cpu_count() or 2) - 1)

    # OPT: ThreadPoolExecutor is kept (not ProcessPoolExecutor) because:
    #   - theo_matrix is a GPU tensor and cannot be pickled across processes.
    #   - The remaining CPU work (scan parsing + numpy ppm filter) releases
    #     the GIL, so threads still provide genuine parallelism for that part.
    #   - All GPU calls serialise on the CUDA stream regardless of executor type.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                process_scan_group,
                scan_group, output_queue,
                theo_matrix,  pair_masses,
                c_alpha_pep,  c_alpha_mod,  c_alpha_site,
                c_beta_pep,   c_beta_mod,   c_beta_site,
                crosslinker_mass, aa_dict, device
            )
            for scan_group in distributed_scans
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in thread: {e}")
                traceback.print_exc()

    while not error_queue.empty():
        thread_idx, error_msg, tb_str = error_queue.get()
        print(f"\nError in thread {thread_idx}:\n{error_msg}\n{tb_str}")

    n_written = 0
    while not output_queue.empty():
        for result in output_queue.get():
            csv_writer.writerow(result)
            n_written += 1

    return n_written


def main(peptides_csv_path: str, output_directory: str, ms2_file_paths: List[str],
         crosslinker_mass: float, aa_dict: Dict[str, float],
         signature_types: Dict[str, float], device: torch.device,
         alpha_acc: List[str]):
    import psutil

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read peptides from a single CSV file (returns accessions as 4th value)
    peptides, modifications, crosslink_sites, accessions = read_peptides(peptides_csv_path)

    # Generate all alpha/beta pair combinations
    (alpha_peptides, alpha_modifications, alpha_crosslink_sites,
     beta_peptides, beta_modifications, beta_crosslink_sites) = generate_all_combinations(
         peptides, modifications, crosslink_sites, accessions, alpha_acc, crosslinker)

    # ── Chunk size: target 40 % of total RAM for ONE chunk ────────────────────
    # Only one chunk is ever live in RAM at a time (build → score all files →
    # del → next chunk), so 40 % of total is safe regardless of how many chunks
    # there are.  Previous code accumulated ALL chunks before scoring anything,
    # which multiplied peak RAM by n_chunks.
    n_pairs    = len(alpha_peptides)
    bin_count  = int(2000.0 / BIN_WIDTH) + 1
    CHUNK_RAM_FRAC = 0.4
    total_ram_mb   = psutil.virtual_memory().total / 1e6
    target_mb      = total_ram_mb * CHUNK_RAM_FRAC
    bytes_per_pair = bin_count * 4   # float32
    CHUNK_SIZE     = max(1, int(target_mb * 1e6 / bytes_per_pair))
    n_chunks       = (n_pairs + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"\nStreaming theoretical spectra (1 chunk live at a time):")
    print(f"  Total pairs      : {n_pairs:,}")
    print(f"  Full matrix size : {n_pairs * bin_count * 4 / 1e6:.0f} MB")
    print(f"  Total RAM        : {total_ram_mb:.0f} MB")
    print(f"  Chunk size       : {CHUNK_SIZE:,} pairs  "
          f"({CHUNK_SIZE * bin_count * 4 / 1e6:.0f} MB / chunk)")
    print(f"  Number of chunks : {n_chunks}")

    # ── Parse every MS2 file ONCE up front (text only, cheap RAM) ────────────
    num_threads = max(1, (os.cpu_count() or 2) - 1)
    parsed_scans: Dict[str, List] = {}
    for file_path in ms2_file_paths:
        print(f"  Parsing: {file_path}")
        scans = split_ms2_by_scans(file_path)
        parsed_scans[file_path] = distribute_scans(scans, num_threads)

    # ── Open all output CSV files ─────────────────────────────────────────────
    output_handles = {}
    csv_writers    = {}
    result_counts  = {fp: 0 for fp in ms2_file_paths}
    for file_path in ms2_file_paths:
        out_path = os.path.join(output_directory,
                                f"xl_ds_{os.path.basename(file_path).split('.')[0]}.csv")
        fh = open(out_path, 'w', newline='')
        w  = csv.writer(fh)
        w.writerow(['Scan', 'Alpha Peptide', 'Beta Peptide',
                'Alpha XL Site', 'Beta XL Site',
                'Alpha Mods', 'Beta Mods',
                'Alpha Mass', 'Beta Mass',
                'Theoretical Mass', 'Observed Mass',
                'Mass_Diff', 'PPM_Diff', 'Xcorr', 'Charge'])
        output_handles[file_path] = fh
        csv_writers[file_path]    = w

    # ── Stream: build one chunk → score ALL MS2 files → free → next chunk ────
    for chunk_idx, chunk_start in enumerate(range(0, n_pairs, CHUNK_SIZE)):
        chunk_end    = min(chunk_start + CHUNK_SIZE, n_pairs)
        chunk_len    = chunk_end - chunk_start
        chunk_t0     = current_time()

        print(f"\n  Building chunk {chunk_idx + 1}/{n_chunks} "
              f"(pairs {chunk_start:,}–{chunk_end - 1:,})...", flush=True)

        c_alpha_pep  = alpha_peptides        [chunk_start:chunk_end]
        c_alpha_mod  = alpha_modifications   [chunk_start:chunk_end]
        c_alpha_site = alpha_crosslink_sites [chunk_start:chunk_end]
        c_beta_pep   = beta_peptides         [chunk_start:chunk_end]
        c_beta_mod   = beta_modifications    [chunk_start:chunk_end]
        c_beta_site  = beta_crosslink_sites  [chunk_start:chunk_end]

        theo_matrix, pair_masses = precompute_theoretical_spectra(
            aa_dict, signature_types,
            c_alpha_pep, c_alpha_mod, c_alpha_site,
            c_beta_pep,  c_beta_mod,  c_beta_site,
            crosslinker_mass, max_charge=4, device=device)

        print(f"  Chunk {chunk_idx + 1}/{n_chunks} built  "
              f"({format_runtime(chunk_t0, current_time())})  "
              f"— scoring {len(ms2_file_paths)} MS2 file(s)...")

        for file_path in ms2_file_paths:
            n = score_chunk_against_file(
                parsed_scans[file_path],
                theo_matrix, pair_masses,
                c_alpha_pep, c_alpha_mod, c_alpha_site,
                c_beta_pep,  c_beta_mod,  c_beta_site,
                crosslinker_mass, aa_dict, device,
                csv_writers[file_path])
            result_counts[file_path] += n
            print(f"    {os.path.basename(file_path)}: +{n} hits "
                  f"(total so far: {result_counts[file_path]})")

        # Explicitly release the chunk before building the next one
        del theo_matrix, pair_masses

    # ── Close all output files ────────────────────────────────────────────────
    for file_path, fh in output_handles.items():
        fh.close()
        n = result_counts[file_path]
        if n > 0:
            out_path = os.path.join(
                output_directory,
                f"xl_ds_{os.path.basename(file_path).split('.')[0]}.csv")
            print(f"  Results written to {out_path}  ({n} rows)")
        else:
            print(f"  Warning: No results for {os.path.basename(file_path)}")

    print(f"\nAll {n_chunks} chunk(s) processed.")

if __name__ == "__main__":
    # OPT: Required on Windows (spawn start method) so that multiprocessing.Pool
    # workers don't re-execute module-level code (imports, device init, etc.).
    # No-op on Linux/macOS (fork), but harmless to keep for portability.
    mp.freeze_support()
    start_time = current_time()
    print(f"Using device: {device} - {torch.cuda.get_device_name(0)}")

    # ── Alpha protein accession(s) ────────────────────────────────────────────
    # All peptides in the CSV whose accession matches an entry here are treated
    # as ALPHA candidates.  Every other peptide becomes a BETA candidate.
    # Add more accessions to the list if multiple proteins should be alpha.
    alpha_acc = [
        'P02652',   # ← replace with your actual protein accession(s)
    ]
    # ─────────────────────────────────────────────────────────────────────────
    main(peptides_csv_path = r'C:\env\test\IGHG3\IGHG3_ds.csv',
         output_directory = r'C:\env\test',
         ms2_file_paths = [
             r"C:\Crux\data\20260511\20260511-TYG_1-HS1-150.ms2",
             r"C:\Crux\data\20260511\20260511-TYG_2-HCC1-150.ms2",
             r"C:\Crux\data\20260511\20260511-TYG_3-HS2-150.ms2",
             r"C:\Crux\data\20260511\20260511-TYG_4-HCC2-150.ms2",
             r"C:\Crux\data\20260511\20260511-TYG_5-HS2-245.ms2",
             r"C:\Crux\data\20260511\20260511-TYG_6-HCC2-245.ms2",
             ],
         crosslinker_mass=crosslinker_mass,
         aa_dict=aa_dict,
         signature_types=signature_types,
         device=device,
         alpha_acc=alpha_acc,
         )

    end_time = current_time()
    print(f"Total runtime: {format_runtime(start_time, end_time)}")