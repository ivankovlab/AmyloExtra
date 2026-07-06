#!/usr/bin/env python3
"""
Amyloid Fibril Classifier for PDB Structures
This script analyzes PDB files to identify potential amyloid fibril structures
based on structural and sequence characteristics.
"""

import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import sys

warnings.filterwarnings('ignore')

try:
    from Bio import PDB
    from Bio.PDB import PDBParser, PPBuilder, DSSP, HSExposure
    from Bio.SeqUtils import IUPACData
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    import numpy as np
except ImportError:
    print("Error: Required biopython module not found.")
    print("Install with: pip install biopython")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None
    print("Note: pandas not installed. Some features may be limited.")

class AmyloidFibrilClassifier:
    """Classify PDB structures as amyloid fibrils or not."""

    # Known amyloidogenic hexapeptides (from amyloid databases)
    AMYLOIDOGENIC_MOTIFS = [
        "GNNQQNY",  # Yeast prion
        "NFGAIL",   # Human islet amyloid polypeptide
        "LVFFAE",   # Aβ(16-21)
        "KLVFFA",   # Aβ(16-21)
        "STVIIE",   # Tau protein
        "MVGGVV",   # α-synuclein
        "CGNITVQ",  # TDP-43
    ]

    # Amyloid-prone amino acids (hydrophobic/aromatic)
    AMYLOIDOGENIC_RESIDUES = set("FYWLIVAMC")

    # Typical amyloid properties
    MIN_BETA_STRAND_CONTENT = 0.20  # At least 20% beta-sheet for amyloid
    MIN_CHAINS = 2                  # Amyloids typically have multiple chains
    MAX_SOLVENT_ACCESSIBILITY = 0.3 # Buried residues in fibril core

    def __init__(self, pdb_file: str):
        """Initialize classifier with PDB file."""
        self.pdb_file = pdb_file
        self.structure = None
        self.dssp = None
        self.results = {}

    def parse_structure(self) -> bool:
        """Parse PDB file and extract structure."""
        try:
            parser = PDBParser(QUIET=True)
            self.structure = parser.get_structure("structure", self.pdb_file)
            return True
        except Exception as e:
            print(f"Error parsing PDB file: {e}")
            return False

    def run_dssp(self) -> bool:
        """Run DSSP for secondary structure assignment."""
        try:
            model = self.structure[0]
            self.dssp = DSSP(model, self.pdb_file)
            return True
        except Exception as e:
            print(f"Error running DSSP: {e}")
            return False

    def calculate_structural_features(self) -> Dict:
        """Calculate structural features relevant to amyloid fibrils."""
        features = {
            'beta_sheet_content': 0.0,
            'parallel_beta_sheet': False,
            'cross_beta_pattern': False,
            'strand_length': 0,
            'interchain_hbonds': 0,
            'total_residues': 0,
            'total_chains': 0,
        }

        if not self.dssp:
            return features

        # Analyze DSSP results
        ss_counts = Counter()
        total_residues = 0

        for dssp_tuple in self.dssp:
            ss = dssp_tuple[2]
            ss_counts[ss] += 1
            total_residues += 1

        features['total_residues'] = total_residues

        # Calculate secondary structure percentages
        if total_residues > 0:
            beta_count = ss_counts.get('E', 0) + ss_counts.get('B', 0)
            features['beta_sheet_content'] = beta_count / total_residues

            # Check for extended beta-strands (amyloid characteristic)
            if beta_count > 0:
                features['strand_length'] = beta_count / ss_counts.get('E', 1)

        # Count chains
        features['total_chains'] = len(list(self.structure[0].get_chains()))

        # Check for cross-beta pattern (hallmark of amyloid)
        # Simplified check: look for beta-sheets perpendicular to fibril axis
        if features['beta_sheet_content'] > self.MIN_BETA_STRAND_CONTENT:
            if features['total_chains'] >= self.MIN_CHAINS:
                features['cross_beta_pattern'] = True

        return features

    def analyze_sequence(self) -> Dict:
        """Analyze sequence for amyloidogenic properties."""
        features = {
            'amyloid_motif_found': False,
            'amyloidogenic_residue_content': 0.0,
            'hydrophobicity': 0.0,
            'charge': 0.0,
            'sequence_length': 0,
        }

        ppb = PPBuilder()
        sequences = []

        for chain in self.structure[0]:
            for pp in ppb.build_peptides(chain):
                sequences.append(str(pp.get_sequence()))

        if not sequences:
            return features

        # Combine sequences if multiple chains
        full_sequence = "".join(sequences)
        features['sequence_length'] = len(full_sequence)

        # Check for amyloidogenic motifs
        for motif in self.AMYLOIDOGENIC_MOTIFS:
            if motif in full_sequence:
                features['amyloid_motif_found'] = True
                break

        # Calculate amyloidogenic residue content
        amyloid_count = sum(1 for aa in full_sequence
                          if aa in self.AMYLOIDOGENIC_RESIDUES)
        if full_sequence:
            features['amyloidogenic_residue_content'] = amyloid_count / len(full_sequence)

        # Calculate hydrophobicity (GRAVY score)
        try:
            analysis = ProteinAnalysis(full_sequence)
            features['hydrophobicity'] = analysis.gravy()
            features['charge'] = analysis.charge_at_pH(7.0)
        except:
            pass

        return features

    def analyze_solvent_accessibility(self) -> Dict:
        """Analyze solvent accessibility (amyloid fibrils often have buried cores)."""
        features = {
            'avg_solvent_accessibility': 1.0,
            'buried_residues_ratio': 0.0,
        }

        if not self.dssp:
            return features

        accessibilities = []
        buried_count = 0

        for dssp_tuple in self.dssp:
            # DSSP relative accessibility (0-1, normalized by residue type)
            rel_acc = dssp_tuple[3]

            if type(rel_acc) == float:
                accessibilities.append(rel_acc)
                if rel_acc < self.MAX_SOLVENT_ACCESSIBILITY:
                    buried_count += 1

        if accessibilities:
            features['avg_solvent_accessibility'] = np.mean(accessibilities)
            features['buried_residues_ratio'] = buried_count / len(accessibilities)

        return features

    def check_fibril_morphology(self) -> Dict:
        """Check for fibril-like morphology."""
        features = {
            'fibril_like': False,
            'longitudinal_repeat': 0.0,
            'helical_symmetry': False,
        }

        # Simplified check: look for elongated structure
        try:
            coords = []
            for atom in self.structure.get_atoms():
                if atom.name == "CA":  # Alpha carbons only
                    coords.append(atom.get_coord())

            if len(coords) >= 10:
                coords = np.array(coords)

                # Calculate principal axes
                centroid = np.mean(coords, axis=0)
                centered = coords - centroid
                cov_matrix = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

                # Check for elongation (one long axis)
                sorted_eigenvalues = np.sort(eigenvalues)[::-1]
                aspect_ratio = sorted_eigenvalues[0] / sorted_eigenvalues[1]

                if aspect_ratio > 3.0:  # Highly elongated
                    features['fibril_like'] = True

                    # Estimate repeat distance (typical amyloid: ~4.7Å)
                    if len(coords) > 100:
                        # Simple distance histogram analysis
                        distances = []
                        for i in range(min(100, len(coords))):
                            for j in range(i+1, min(i+20, len(coords))):
                                dist = np.linalg.norm(coords[i] - coords[j])
                                if 4.0 < dist < 6.0:
                                    distances.append(dist)

                        if distances:
                            hist, bins = np.histogram(distances, bins=20)
                            max_bin = bins[np.argmax(hist)]
                            features['longitudinal_repeat'] = max_bin

                            if 4.6 < max_bin < 4.8:  # Typical amyloid repeat
                                features['helical_symmetry'] = True
        except:
            pass

        return features

    def calculate_amyloid_score(self) -> float:
        """Calculate overall amyloid propensity score."""

        scores, weights = [], []

        # Beta-sheet content score.
        beta_content = self.results['structural']['beta_sheet_content']
        if beta_content > self.MIN_BETA_STRAND_CONTENT:
            scores.append(min(beta_content * 2, 1.0))
            weights.append(0.3)

        # Cross-beta pattern
        if self.results['structural']['cross_beta_pattern']:
            scores.append(1.0)
            weights.append(0.2)

        # Amyloidogenic motifs
        #if self.results['sequence']['amyloid_motif_found']:
            #scores.append(1.0)
            #weights.append(0.15)

        # Amyloidogenic residue content
        amyloid_res = self.results['sequence']['amyloidogenic_residue_content']
        scores.append(amyloid_res)
        weights.append(0.1)

        # Buried residues (fibril core)
        buried = self.results['solvent']['buried_residues_ratio']
        scores.append(buried)
        weights.append(0.15)

        # Fibril-like morphology
        if self.results['morphology']['fibril_like']:
            scores.append(1.0)
            weights.append(0.1)

        # Calculate weighted score
        if weights:
            return np.average(scores, weights=weights)
        return 0.0

    def classify(self, threshold: float = 0.6) -> Tuple[bool, float, Dict]:
        """
        Classify the structure as amyloid or not.

        Args:
            threshold: Score threshold for amyloid classification

        Returns:
            Tuple of (is_amyloid, score, detailed_results)
        """

        if not self.parse_structure():
            return False, 0.0, {}

        self.run_dssp()

        # Collect all features
        self.results = {
            'structural': self.calculate_structural_features(),
            'sequence': self.analyze_sequence(),
            'solvent': self.analyze_solvent_accessibility(),
            'morphology': self.check_fibril_morphology(),
        }

        # Calculate final score
        score = self.calculate_amyloid_score()

        # Determine classification
        is_amyloid = score >= threshold

        return is_amyloid, score, self.results

    def print_report(self, is_amyloid: bool, score: float, results: Dict):
        """Print detailed classification report."""

        print("\n" + "="*60)
        print("AMYLOID FIBRIL CLASSIFICATION REPORT")
        print("="*60)
        print(f"PDB File: {self.pdb_file}")
        print(f"Classification: \
{'AMYLOID FIBRIL' if is_amyloid else 'NOT AMYLOID'}")
        print(f"Confidence Score: {score:.3f}")

        print("\nDETAILED ANALYSIS:")
        print("-"*60)

        # Structural features.
        struct = results['structural']
        print(f"\nStructural Features:")
        print(f"  Beta-sheet content: \
{struct['beta_sheet_content']:.2%}")
        print(f"  Cross-beta pattern: \
{'Yes' if struct['cross_beta_pattern'] else 'No'}")
        print(f"  Number of chains: \
{struct['total_chains']}")
        print(f"  Total residues: \
{struct['total_residues']}")

        # Sequence features.
        seq = results['sequence']
        print(f"\nSequence Features:")
        print(f"  Amyloid motif found: \
{'Yes' if seq['amyloid_motif_found'] else 'No'}")
        print(f"  Amyloidogenic residue content: \
{seq['amyloidogenic_residue_content']:.2%}")
        print(f"  Hydrophobicity (GRAVY): \
{seq['hydrophobicity']:.3f}")
        print(f"  Net charge at pH 7: \
{seq['charge']:.2f}")

        # Solvent accessibility.
        solv = results['solvent']
        print(f"\nSolvent Accessibility:")
        print(f"  Average relative accessibility: \
{solv['avg_solvent_accessibility']:.3f}")
        print(f"  Buried residues ratio: \
{solv['buried_residues_ratio']:.2%}")

        # Morphology
        morph = results['morphology']
        print(f"\nMorphological Features:")
        print(f"  Fibril-like elongated structure: \
{'Yes' if morph['fibril_like'] else 'No'}")
        if morph['longitudinal_repeat'] > 0:
            print(f"  Longitudinal repeat distance: \
{morph['longitudinal_repeat']:.2f} Å")
        print(f"  Helical symmetry pattern: \
{'Yes' if morph['helical_symmetry'] else 'No'}")

        print("\n" + "="*60)

        if is_amyloid:
            print("\nINDICATORS OF AMYLOID STRUCTURE:")
            indicators = []
            if struct['beta_sheet_content'] > self.MIN_BETA_STRAND_CONTENT:
                indicators.append("High beta-sheet content")
            if struct['cross_beta_pattern']:
                indicators.append("Cross-beta pattern")
            if seq['amyloid_motif_found']:
                indicators.append("Known amyloid motif")
            if morph['fibril_like']:
                indicators.append("Fibril-like morphology")

            for i, indicator in enumerate(indicators, 1):
                print(f"  ({i}) {indicator}.")
        else:
            print("\nSUGGESTIONS FOR FURTHER ANALYSIS:")
            print("  (1) Consider NMR or cryo-EM for definitive confirmation.")
            print("  (2) Check for low-resolution regions in the structure.")
            print("  (3) Verify with specialized amyloid prediction servers.")

        print("\n" + "="*60)

def main():
    """Main function to run the classifier."""

    if len(sys.argv) != 2:
        print("Usage: python amyloid_classifier.py <pdb_file>")
        print("Example: python amyloid_classifier.py 2m4j.pdb")
        sys.exit(1)

    pdb_file = sys.argv[1]

    print(f"Analyzing PDB file: {pdb_file}")

    # Create classifier.
    classifier = AmyloidFibrilClassifier(pdb_file)

    # Perform classification.
    is_amyloid, score, results = classifier.classify(threshold=0.8)

    # Print report.
    classifier.print_report(is_amyloid, score, results)

if __name__ == "__main__":
    main()
