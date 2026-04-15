"""
3D-3D correspondence matching using feature similarity.
Matches query object features to target proposal features.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CorrespondenceMatcher:
    """Match 3D features between query and target."""

    def __init__(self,
                 distance_threshold: float = 0.1,
                 ratio_test: float = 0.8,
                 min_correspondences: int = 10):
        """
        Initialize correspondence matcher.

        Args:
            distance_threshold: Max feature distance for valid correspondence
            ratio_test: Lowe's ratio test threshold (nearest/second-nearest)
            min_correspondences: Minimum correspondences required
        """
        self.distance_threshold = distance_threshold
        self.ratio_test = ratio_test
        self.min_correspondences = min_correspondences

        logger.info(f"CorrespondenceMatcher initialized: "
                   f"dist_thresh={distance_threshold}, ratio={ratio_test}")

    def match_features(self,
                      query_features: np.ndarray,
                      target_features: np.ndarray,
                      mutual: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match features using nearest neighbor search.

        Args:
            query_features: (N, D) query feature matrix
            target_features: (M, D) target feature matrix
            mutual: Whether to enforce mutual nearest neighbors

        Returns:
            query_indices: (K,) indices into query features
            target_indices: (K,) indices into target features
        """
        # Check for NaN/Inf
        if np.any(~np.isfinite(query_features)) or np.any(~np.isfinite(target_features)):
            logger.warning("NaN or Inf detected in features, skipping")
            return np.array([], dtype=int), np.array([], dtype=int)

        # Normalize features
        query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
        target_norm = target_features / (np.linalg.norm(target_features, axis=1, keepdims=True) + 1e-8)

        # Compute similarity matrix (cosine similarity)
        similarity = query_norm @ target_norm.T

        # Find nearest neighbors for each query feature
        matches_q2t = []

        for q_idx in range(len(query_features)):
            # Get distances to all target features
            sims = similarity[q_idx]

            # Find top 2 nearest neighbors
            if len(sims) < 2:
                continue

            top2_indices = np.argpartition(sims, -2)[-2:]
            top2_sims = sims[top2_indices]

            # Sort top 2
            sort_idx = np.argsort(top2_sims)[::-1]
            nearest_idx = top2_indices[sort_idx[0]]
            second_idx = top2_indices[sort_idx[1]]
            nearest_sim = top2_sims[sort_idx[0]]
            second_sim = top2_sims[sort_idx[1]]

            # Distance threshold (convert similarity to distance)
            nearest_dist = 1.0 - nearest_sim
            if nearest_dist > self.distance_threshold:
                continue

            # Ratio test (Lowe's ratio test)
            # For distances: nearest_dist / second_dist should be < threshold
            second_dist = 1.0 - second_sim
            if second_dist > 1e-8:  # Avoid division by zero
                ratio = nearest_dist / second_dist
                if ratio > self.ratio_test:  # Reject if nearest isn't clearly better
                    continue

            matches_q2t.append((q_idx, nearest_idx, nearest_sim))

        if len(matches_q2t) == 0:
            logger.warning("No valid matches found after ratio test")
            return np.array([], dtype=int), np.array([], dtype=int)

        logger.debug(f"After ratio test: {len(matches_q2t)} matches")

        # Mutual nearest neighbor filtering
        if mutual:
            matches_t2q = {}
            for t_idx in range(len(target_features)):
                sims = similarity[:, t_idx]
                if len(sims) == 0:
                    continue
                q_idx = np.argmax(sims)
                matches_t2q[t_idx] = q_idx

            # Keep only mutual matches
            mutual_matches = []
            for q_idx, t_idx, sim in matches_q2t:
                if matches_t2q.get(t_idx) == q_idx:
                    mutual_matches.append((q_idx, t_idx))

            matches = mutual_matches
        else:
            matches = [(q, t) for q, t, s in matches_q2t]

        if len(matches) < self.min_correspondences:
            logger.warning(f"Too few correspondences: {len(matches)} < {self.min_correspondences}")
            return np.array([], dtype=int), np.array([], dtype=int)

        query_indices = np.array([m[0] for m in matches])
        target_indices = np.array([m[1] for m in matches])

        logger.info(f"Found {len(matches)} correspondences")

        return query_indices, target_indices

    def match_to_proposal(self,
                         query_data: Dict,
                         proposal_data: Dict,
                         feature_type: str = 'geometric') -> Optional[Dict]:
        """
        Match query to a single target proposal.

        Args:
            query_data: Query dict with 'points', 'features', 'geometric_features', 'visual_features'
            proposal_data: Proposal dict with 'points', 'features'
            feature_type: 'geometric', 'visual', or 'fused'

        Returns:
            Match dict with correspondences or None if too few matches
        """
        # Select features
        if feature_type == 'geometric':
            if 'geometric_features' in query_data:
                query_feats = query_data['geometric_features']
            else:
                logger.warning("No geometric features in query, using 'features'")
                query_feats = query_data['features']
            target_feats = proposal_data['features']

        elif feature_type == 'visual':
            if 'visual_features' not in query_data:
                logger.error("No visual features in query")
                return None
            query_feats = query_data['visual_features']
            target_feats = proposal_data.get('visual_features', proposal_data['features'])

        elif feature_type == 'fused':
            query_feats = query_data['features']
            target_feats = proposal_data.get('fused_features', proposal_data['features'])
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        # Match features (mutual=True reduces spurious correspondences)
        query_indices, target_indices = self.match_features(query_feats, target_feats, mutual=True)

        if len(query_indices) == 0:
            return None

        # Get corresponding 3D points
        query_points = query_data['points'][query_indices]
        target_points = proposal_data['points'][target_indices]

        match_result = {
            'query_indices': query_indices,
            'target_indices': target_indices,
            'query_points': query_points,
            'target_points': target_points,
            'num_correspondences': len(query_indices),
            'proposal_id': proposal_data.get('proposal_id', -1)
        }

        logger.debug(f"Matched query to proposal {proposal_data.get('proposal_id', -1)}: "
                    f"{len(query_indices)} correspondences")

        return match_result

    def match_to_scene(self,
                      query_data: Dict,
                      proposals: List[Dict],
                      feature_type: str = 'geometric',
                      top_k: int = 5) -> List[Dict]:
        """
        Match query to all proposals in scene, return top-k.

        Args:
            query_data: Query feature dict
            proposals: List of proposal feature dicts
            feature_type: Feature type to use
            top_k: Return top K proposals by number of correspondences

        Returns:
            List of match dicts, sorted by num_correspondences
        """
        matches = []

        for proposal in proposals:
            match = self.match_to_proposal(query_data, proposal, feature_type)
            if match is not None:
                matches.append(match)

        # Sort by number of correspondences
        matches.sort(key=lambda x: x['num_correspondences'], reverse=True)

        # Take top K
        matches = matches[:top_k]

        logger.info(f"Matched to {len(matches)}/{len(proposals)} proposals")
        if len(matches) > 0:
            logger.info(f"Top match: {matches[0]['num_correspondences']} correspondences")

        return matches
