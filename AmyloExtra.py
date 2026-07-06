from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from typing import Dict, List, Tuple

import numpy as np
from Bio import PDB
from scipy.spatial import KDTree


def run_dssp(filepath: str) -> Dict[Tuple[str, str], str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".dssp", delete=False)
    tmp.close()

    subprocess.run(
        ["mkdssp", filepath, tmp.name],
        capture_output=True,
        text=True
    )

    if not os.path.exists(tmp.name):
        return {}

    ss_map = {}
    try:
        with open(tmp.name, 'r') as f:
            lines = f.readlines()
        os.unlink(tmp.name)
    except:
        return {}

    # Process DSSP output.
    for line in lines:
        if line and len(line) > 10:
            if line[5:11].strip().isdigit() or \
               (line[5] == ' ' and line[6:11].strip().isdigit()):
                try:
                    res_num_str = line[5:11].strip()
                    if not res_num_str:
                        continue

                    chain_id = line[11] if len(line) > 11 else ' '
                    if chain_id == ' ' and len(line) > 12:
                        chain_id = line[12]
                    if chain_id == ' ':
                        chain_id = 'A'

                    ss_code = line[16] if len(line) > 16 else ' '

                    if ss_code in ['H', 'E', 'T', 'S', 'G', 'I', 'B']:
                        ss_map[(chain_id, res_num_str)] = ss_code
                except:
                    continue

    return ss_map


def fit_plane(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(coords) < 3:
        raise ValueError("Need at least 3 points")

    centroid = coords.mean(axis=0)
    centered = coords - centroid

    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normal = Vt[-1]
        return centroid, normal / np.linalg.norm(normal)
    except:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, np.argmin(eigvals)]
        return centroid, normal / np.linalg.norm(normal)


def best_approximation_line_direction(points: list) -> np.ndarray:
    """
    Returns the unit direction vector of the best-approximation line
    for a list of 3D points.

    The best-approximation line minimises the sum of squared perpendicular
    distances from the points to the line. It passes through the centroid
    of the points, and its direction is given by the first principal component
    (the eigenvector of the covariance matrix with the largest eigenvalue).

    Parameters
    ----------
    points : list of np.ndarray
        Each array must have shape (3,) and contain the (x, y, z) coordinates
        of one point.

    Returns
    -------
    direction : np.ndarray
        A unit vector (norm = 1) of shape (3,) giving the direction of the line.

    Raises
    ------
    ValueError
        If the input list is empty.
    """
    if len(points) == 0:
        raise ValueError("Input list of points must not be empty.")

    # Convert list of arrays to a 2D array of shape (n, 3)
    pts = np.array(points, dtype=np.float64)

    # Center the points around the mean
    centered = pts - pts.mean(axis=0)

    # If all points are identical, any direction is equally valid.
    # Return an arbitrary unit vector (here, x-axis) to avoid degeneracy.
    if np.allclose(centered, 0.0):
        return np.array([1.0, 0.0, 0.0])

    # Singular Value Decomposition: columns of Vt are the right singular vectors
    # (principal directions). The first row of Vt corresponds to the largest
    # singular value (most variance) -> best line direction.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]

    # Ensure it is a unit vector (SVD already returns unit vectors)
    # and fix orientation if desired (e.g., make first non-zero component positive)
    if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
        direction = -direction

    return direction


def find_all_beta_strands(chain, ss_map: Dict, chain_id: str,
                             min_beta_length: int = 3):
    '''Alpha-carbon atoms of the longest beta-strand.'''

    beta_residues = []
    for r in chain:
        if not r.has_id("CA"):
            continue
        res_id = str(r.id[1])
        ss = ss_map.get((chain_id, res_id), "-")
        if ss == "E":
            try:
                res_num = int(res_id)
                beta_residues.append((res_num,
                                     r["CA"].get_vector().get_array()))
            except:
                beta_residues.append((0, r["CA"].get_vector().get_array()))

    if not beta_residues:
        return []

    beta_residues.sort(key=lambda x: x[0])

    runs = []
    current_run = [beta_residues[0]]

    for i in range(1, len(beta_residues)):
        if beta_residues[i][0] == current_run[-1][0] + 1:
            current_run.append(beta_residues[i])
        else:
            if len(current_run) >= min_beta_length:
                runs.append(current_run)
            current_run = [beta_residues[i]]

    if len(current_run) >= min_beta_length:
        runs.append(current_run)

    if not runs:
        return []

    #longest = max(runs, key=len)

    coords = []

    for run in runs:
        coords += [coord for _, coord in run]

    #return [coord for _, coord in longest]

    return coords


def best_ca_coords(chain, ss_map: Dict, chain_id: str, min_residues: int,
                   min_beta_length: int = 3) -> np.ndarray:
    coords = find_all_beta_strands(chain, ss_map, chain_id, min_beta_length)
    if len(coords) >= min_residues:
        return np.array(coords, dtype=float)

    coords = []
    for r in chain:
        if not r.has_id("CA"):
            continue
        res_id = str(r.id[1])
        ss = ss_map.get((chain_id, res_id), "-")
        if ss == "E":
            coords.append(r["CA"].get_vector().get_array())

    if len(coords) >= min_residues:
        return np.array(coords, dtype=float)

    all_coords = []
    for r in chain:
        if r.has_id("CA"):
            all_coords.append(r["CA"].get_vector().get_array())

    if len(all_coords) >= min_residues:
        return np.array(all_coords, dtype=float)

    return np.array([], dtype=float)


def extract_chain_planes(filepath: str, min_residues: int = 3,
                         min_beta_length: int = 3,
                         min_chain_size: int = 10,
                         verbose: bool = True) -> Dict:
    '''For each chain, calculate its centroid and normal.'''

    ext = os.path.splitext(filepath)[1].lower()

    parser = (
        PDB.MMCIFParser(QUIET=True)
        if ext in (".cif", ".mmcif")
        else PDB.PDBParser(QUIET=True)
    )

    try:
        structure = parser.get_structure("s", filepath)
    except:
        return {}

    first_model = next(iter(structure))
    ss_map = run_dssp(filepath)

    chain_planes = {}
    for chain in first_model:
        ca_count = sum(1 for r in chain if r.has_id("CA"))

        if ca_count < min_chain_size:
            continue

        ca_coords = best_ca_coords(chain, ss_map, chain.id,
                                   min_residues, min_beta_length)

        if verbose:
            print(chain.id)
            print(ca_coords)

        # If the number of amino acid residues is not less than the minimal
        # polypeptide length.
        if len(ca_coords) >= min_residues:
            try:
                centroid, normal = fit_plane(ca_coords)
                # Store the plane through its point and normal.
                chain_planes[chain.id] = (centroid, normal)
            except:
                continue

    return chain_planes


def find_neighbors(chain_planes: Dict,
                   neighbor_dist: float) -> List[Tuple[str, str]]:
    '''Find pairs of adjacent monomers.
        chain_planes: dict containing centers and normals
        neighbor_dist: float threshold
    '''

    # Extract IDs of the monomers.
    chain_ids = list(chain_planes.keys())

    # Pairs need at least two monomers.
    if len(chain_ids) < 2:
        return []

    # Extract centroids of all monomers.
    centroids = np.array([chain_planes[cid][0] for cid in chain_ids])

    # Build a KDTree from the centroids for faster neighbors search.
    tree = KDTree(centroids)

    # Find the pairs.
    pairs_idx = tree.query_pairs(r=neighbor_dist)

    return [(chain_ids[i], chain_ids[j]) for i, j in pairs_idx]


class UnionFind:
    def __init__(self, elements):
        self._parent = {e: e for e in elements}
        self._rank = {e: 0 for e in elements}

    def find(self, x):
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x, y):
        '''Merge sets of elements with names x and y.'''

        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def n_components(self) -> int:
        '''Number of unions.'''

        return len({self.find(e) for e in self._parent})

    def components(self) -> dict:
        '''Lists of element names contained in each component.'''

        components = dict()

        for e in self._parent:
            p = self.find(e)

            if p not in components:
                components[p] = []

            components[p].append(e)

        return tuple(tuple(components[p]) for p in components)


def count_protofilaments(
    filepath: str,
    dot_threshold: float = 0.8,
    min_residues: int = 3,
    neighbor_dist: float = 20.0,
    max_lateral_dist: float = 30.0,
    min_beta_length: int = 3,
    min_chain_size: int = 10,
    verbose: bool = True
) -> int:
    # Calculate centers and normals of each amyloid-like chain.
    chain_planes = extract_chain_planes(filepath, min_residues, min_beta_length,
                                        min_chain_size)

    if verbose:
        print(chain_planes)

    # Store IDs of all found amyloid-like chains.
    chain_ids = list(chain_planes.keys())

    # If no chain found.
    if not chain_ids:
        raise ValueError('This path does not exist.')

    # Initiate the union of sets, where each monomer is initially in its own
    # set.
    uf = UnionFind(chain_ids)

    # Check all neighbor amyloid-like chains, whether they belong to one
    # protofilament.
    for id1, id2 in find_neighbors(chain_planes, neighbor_dist):
        if verbose:
            print(id1, id2, 'are neighbours.')

        # Centers and their normal vectors.
        c1, n1 = chain_planes[id1]
        c2, n2 = chain_planes[id2]

        # Vector connecting centers of two monomers.
        v = c2 - c1

        # If the distance between centers is too short, this is one monomer.
        dist = np.linalg.norm(v)
        if dist < 1e-6:
            continue

        # Normalize the vector.
        v_unit = v / dist

        # Check whether connecting vector and normal vectors of the planes
        # are nearly parallel.
        if abs(np.dot(v_unit, n1)) < dot_threshold or \
           abs(np.dot(v_unit, n2)) < dot_threshold:
            continue

        # Check distance deviation of the first chain normal from connecting
        # vector.
        if np.linalg.norm(v - np.dot(v, n1) * n1) > max_lateral_dist:
            continue

        # Check distance deviation of the second chain normal from connecting
        # vector.
        if np.linalg.norm(v - np.dot(v, n2) * n2) > max_lateral_dist:
            continue

        # If all conditions are satisfied, merge the protofilaments of the
        # chains with IDs id1 and id2.
        uf.union(id1, id2)

    components, n_components = uf.components(), uf.n_components()

    selected_protofilament, selected_protofilament_centers = components[0], []
    for chain_id in selected_protofilament:
        selected_protofilament_centers.append(chain_planes[chain_id][0])

    axis = best_approximation_line_direction(selected_protofilament_centers)

    ordered_protofilaments = []
    for protofilament in components:
        ordered_protofilament = []
        for chain_id in protofilament:
            center = chain_planes[chain_id][0]
            projection = np.dot(center, axis)
            ordered_protofilament.append((projection, chain_id))
        ordered_protofilament = sorted(ordered_protofilament)
        ordered_protofilament = [pair[1] for pair in ordered_protofilament]
        ordered_protofilaments.append(tuple(ordered_protofilament))

    return tuple(ordered_protofilaments)


def _cli() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("filepath")
    parser.add_argument("--dot-threshold", type=float, default=0.8)
    parser.add_argument("--min-residues", type=int, default=3)
    parser.add_argument("--neighbor-dist", type=float, default=20.0)
    parser.add_argument("--max-lateral-dist", type=float, default=30.0)
    parser.add_argument("--min-beta-length", type=int, default=3)
    parser.add_argument("--min-chain-size", type=int, default=10,
                       help="Minimum number of CA atoms in chain to consider\
                             (default: 10)")

    args = parser.parse_args()

    result = count_protofilaments(
        args.filepath,
        args.dot_threshold,
        args.min_residues,
        args.neighbor_dist,
        args.max_lateral_dist,
        args.min_beta_length,
        args.min_chain_size,
    )

    print(result)


# Parse CLI if run as main program.
if __name__ == "__main__":
    _cli()
