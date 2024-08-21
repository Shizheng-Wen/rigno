from typing import Tuple, Union, NamedTuple

import flax.typing
import jax.numpy as jnp
import jax.random
import numpy as np
from flax import linen as nn
from scipy.spatial import Delaunay
from time import time
from rigno.graph.typed_graph import (
    TypedGraph, EdgeSet, EdgeSetKey,
    EdgesIndices, NodeSet, Context)
from rigno.models.graphnet import DeepTypedGraphNet
from rigno.models.operator import AbstractOperator, Inputs
from rigno.utils import Array, shuffle_arrays


class RegionInteractionGraphs(NamedTuple):
  p2r: TypedGraph
  r2r: TypedGraph
  r2p: TypedGraph

class RegionInteractionGraphBuilder:

  # TODO: Optimize by avoiding for loops

  def __init__(self,
    periodic: bool,
    rmesh_levels: int,
    subsample_factor: float,
    overlap_factor_p2r: float,
    overlap_factor_r2p: float,
    node_coordinate_freqs: int,
    mode:str = "fast" # ["debug","fast","slow"]
  ):

    # Set attributes
    self.periodic = periodic
    self.overlap_factor_p2r = overlap_factor_p2r
    self.overlap_factor_r2p = overlap_factor_r2p
    self.node_coordinate_freqs = node_coordinate_freqs
    self.rmesh_levels = rmesh_levels
    self.subsample_factor = subsample_factor
    assert mode in ["debug","fast","slow"], f"mode should in `debug`, `fast`, `slow`, got {mode}"
    self.mode = mode

    # Domain shifts for periodic BC
    self._domain_shifts = [
      np.array([[0., 0.]]),  # C
      np.array([[-2, 0.]]),  # W
      np.array([[-2, +2]]),  # NW
      np.array([[0., +2]]),  # N
      np.array([[+2, +2]]),  # NE
      np.array([[+2, 0.]]),  # E
      np.array([[+2, -2]]),  # SE
      np.array([[0., -2]]),  # S
      np.array([[-2, -2]]),  # SW
    ]

    if self.mode in ["fast", "debug"]:
      self._domain_shifts = np.concatenate(self._domain_shifts,0)

  def _get_supported_points(self,
    centers: Array,
    points: Array,
    radii: Array,
    ord_distance: int = 2,
  ) -> Array:
    """ord_distance can be 1, 2, or np.inf"""

    # Get relative coordinates
    rel = points[:, None] - centers
    # Mirror relative positions because of periodic boudnary conditions
    if self.periodic:
      rel = jnp.where(rel >= 1., (rel - 2.), rel)
      rel = jnp.where(rel < -1., (rel + 2.), rel)

    # Compute distance
    # NOTE: Order of the norm determines the shape of the sub-regions
    distance = jnp.linalg.norm(rel, ord=ord_distance, axis=-1)

    # Get indices
    # -> [idx_point, idx_center]
    idx_nodes = jnp.stack(jnp.where(distance <= radii), axis=-1)

    return idx_nodes

  def _init_structural_features(self,
    x_sen: Array,
    x_rec: Array,
    idx_sen: list[int],
    idx_rec: list[int],
    max_edge_length: float,
    feats_sen: Array = None,
    feats_rec: Array = None,
    domain_sen: list[int] = None,
    domain_rec: list[int] = None,
    shifts: list[Array] = None,
  ) -> Tuple[EdgeSet, NodeSet, NodeSet]:

    # Get number of nodes and the edges
    num_sen = x_sen.shape[0]
    num_rec = x_rec.shape[0]
    assert len(idx_sen) == len(idx_rec)
    num_edg = len(idx_sen)

    # Process coordinates
    phi_sen = np.pi * (x_sen + 1)  # [0, 2pi]
    phi_rec = np.pi * (x_rec + 1)  # [0, 2pi]

    # Define node features
    # NOTE: Sinusoidal features don't need normalization
    if self.periodic:
      # NOTE: no need to optimize here, small loop
      sender_node_feats = np.concatenate([
          np.concatenate([np.sin((k+1) * phi_sen), np.cos((k+1) * phi_sen)], axis=-1)
          for k in range(self.node_coordinate_freqs)
        ], axis=-1)
      receiver_node_feats = np.concatenate([
          np.concatenate([np.sin((k+1) * phi_rec), np.cos((k+1) * phi_rec)], axis=-1)
          for k in range(self.node_coordinate_freqs)
        ], axis=-1)
    else:
      sender_node_feats = np.concatenate([x_sen], axis=-1)
      receiver_node_feats = np.concatenate([x_rec], axis=-1)
    # Concatenate with forced features
    if feats_sen is not None:
      sender_node_feats = np.concatenate([sender_node_feats, feats_sen], axis=-1)
    if feats_rec is not None:
      receiver_node_feats = np.concatenate([receiver_node_feats, feats_rec], axis=-1)

    # Build node sets
    sender_node_set = NodeSet(
      n_node=jnp.array([num_sen]),
      features=jnp.array(sender_node_feats)
    )
    receiver_node_set = NodeSet(
      n_node=jnp.array([num_rec]),
      features=jnp.array(receiver_node_feats)
    )

    # Define edge features
    if self.mode == "fast":
      z_ij = x_sen[np.array(idx_sen)] - x_rec[np.array(idx_rec)]
    elif self.mode == "slow":
      z_ij = np.stack([
        (x_sen[s] - x_rec[r])
        for s, r in zip(idx_sen, idx_rec)
      ], axis=0)
    else:
      z_ij = np.stack([
        (x_sen[s] - x_rec[r])
        for s, r in zip(idx_sen, idx_rec)
      ], axis=0)
      
      z_ij_ = x_sen[np.array(idx_sen)] - x_rec[np.array(idx_rec)]
      np.testing.assert_allclose(z_ij, z_ij_)
     
    assert np.all(np.abs(z_ij) <= 2.)
    if self.periodic:
      # NOTE: For p2r and r2p, mirror the large relative coordinates
      # MODIFY: Unify the mirroring with the below method in r2r
      if shifts is None:
        z_ij = np.where(z_ij < -1.0, z_ij + 2, z_ij)
        z_ij = np.where(z_ij >= 1.0, z_ij - 2, z_ij)
      # NOTE: For the r2r multi-mesh, use extended domain indices and shifts
      else:
        # NOTE: Need to be further optimized
        if self.mode == "fast":
          z_ij = (x_sen[np.array(idx_sen)] + shifts[np.array(np.array(domain_sen))]) - (x_rec[np.array(idx_rec)]+shifts[np.array(domain_rec)])
        elif self.mode == "slow":
          z_ij = np.stack([
          ((x_sen[s] + shifts[dom_s]) - (x_rec[r] + shifts[dom_r]))
          for s, r, dom_s, dom_r in zip(idx_sen, idx_rec, domain_sen, domain_rec)
          ], axis=0)
        else:
          z_ij = np.stack([
          ((x_sen[s] + shifts[dom_s]) - (x_rec[r] + shifts[dom_r]))
          for s, r, dom_s, dom_r in zip(idx_sen, idx_rec, domain_sen, domain_rec)
          ], axis=0)
          z_ij_ = (x_sen[np.array(idx_sen)] + shifts[np.array(np.array(domain_sen))]) - (x_rec[np.array(idx_rec)]+shifts[np.array(domain_rec)])
          np.testing.assert_allclose(z_ij, z_ij_)
    d_ij = np.linalg.norm(z_ij, axis=-1, keepdims=True)
    # Normalize and concatenate edge features
    assert np.all(np.abs(z_ij) <= max_edge_length), np.max(np.abs(z_ij))
    assert np.all(np.abs(d_ij) <= max_edge_length), np.max(np.abs(d_ij))
    z_ij = z_ij / max_edge_length
    d_ij = d_ij / max_edge_length
    edge_feats = np.concatenate([z_ij, d_ij], axis=-1)

    # Build edge set
    edge_set = EdgeSet(
      n_edge=jnp.array([num_edg]),
      indices=EdgesIndices(
        senders=jnp.array(idx_sen),
        receivers=jnp.array(idx_rec)
      ),
      features=jnp.array(edge_feats),
    )

    return edge_set, sender_node_set, receiver_node_set

  def _compute_minimum_support_radii(self, x: Array) -> Array:
      if self.periodic:
        if self.mode == "fast":
          x_extended = (x[None, :, :] + self._domain_shifts[:, None, :]).reshape(-1, 2)
        elif self.mode == "slow":
          x_extended = np.concatenate(
            [x + self._domain_shifts[idx] for idx in range(len(self._domain_shifts))], axis=0)
        else:
          x_extended = np.concatenate(
            [x + self._domain_shifts[None,idx] for idx in range(len(self._domain_shifts))], axis=0)
          x_extended_ = (x[None, :, :] + self._domain_shifts[:, None, :]).reshape(-1, 2)
         
          np.testing.assert_allclose(
            x_extended,
            x_extended_
          )
        tri = Delaunay(points=x_extended)
      else:
        tri = Delaunay(points=x)

      medians = _compute_triangulation_medians(tri)
      radii = np.zeros(shape=(x.shape[0],))
      if self.mode == "fast":
        mask = tri.simplices < x.shape[0] # [N, 3]
        values = medians[mask]
        indices  = tri.simplices[mask] 
        sorted_idx = np.argsort(indices)
        sorted_indices = indices[sorted_idx]
        sorted_values  = values[sorted_idx]
        unique_indices, idx_start = np.unique(sorted_indices, return_index=True)
        radii[unique_indices] = np.maximum.reduceat(sorted_values,idx_start)
      elif self.mode == "slow":
        for s, simplex in enumerate(tri.simplices):
          for v in range(simplex.shape[0]):
            if simplex[v] < x.shape[0]:
              m = medians[s, v]
              radii[simplex[v]] = max(m, radii[simplex[v]])
      else:
        for s, simplex in enumerate(tri.simplices):
          for v in range(simplex.shape[0]):
            if simplex[v] < x.shape[0]:
              m = medians[s, v]
              radii[simplex[v]] = max(m, radii[simplex[v]])
        radii_ = np.zeros_like(radii)
        mask = tri.simplices < x.shape[0] # [N, 3]
        values = medians[mask]
        indices  = tri.simplices[mask] 
        sorted_idx = np.argsort(indices)
        sorted_indices = indices[sorted_idx]
        sorted_values  = values[sorted_idx]
        unique_indices, idx_start = np.unique(sorted_indices, return_index=True)
        radii_[unique_indices] = np.maximum.reduceat(sorted_values,idx_start)
        np.testing.assert_allclose(radii, radii_)
      return radii

  def _build_p2r_graph(self, x_inp: Array, x_rmesh: Array, r_min: Array) -> TypedGraph:
    """Constructrs the encoder graph (pmesh to rmesh)"""

    # Set the sub-region radii
    radius = self.overlap_factor_p2r * r_min

    # Get indices of supported points
    idx_nodes = self._get_supported_points(
      centers=x_rmesh,
      points=x_inp,
      radii=radius,
    )

    # Get the initial features
    edge_set, pmesh_node_set, rmesh_node_set = self._init_structural_features(
      x_sen=x_inp,
      x_rec=x_rmesh,
      idx_sen=idx_nodes[:, 0],
      idx_rec=idx_nodes[:, 1],
      max_edge_length=(2. * jnp.sqrt(x_rmesh.shape[1])),
      feats_rec=radius.reshape(-1, 1),
    )

    # Construct the graph
    graph = TypedGraph(
      context=Context(n_graph=jnp.array([1]), features=()),
      nodes={'pnodes': pmesh_node_set, 'rnodes': rmesh_node_set},
      edges={EdgeSetKey('p2r', ('pnodes', 'rnodes')): edge_set},
    )

    return graph

  def _build_r2r_graph(self, x_rmesh: Array, r_min: Array) -> TypedGraph:
    """Constructrs the processor graph (rmesh to rmesh)"""

    # Set the sub-region radii
    radius = self.overlap_factor_p2r * r_min

    # Define edges and their corresponding -extended- domain
    if self.mode in ["fast", "debug"]:
      edges = None 
      domains = None 
    else:
      edges = []
      domains = []
    for level in range(self.rmesh_levels):
      # Sub-sample the rmesh
      _rmesh_size = int(x_rmesh.shape[0] / (self.subsample_factor ** level))
      _x_rmesh = x_rmesh[:_rmesh_size]
      if self.periodic:
        # Repeat the rmesh in periodic directions
      
        if self.mode == "fast":
          _x_rmesh_extended = (_x_rmesh[None, :, :] + self._domain_shifts[:, None, :]).reshape(-1, 2)
        elif self.mode == "slow":
          _x_rmesh_extended = jnp.concatenate(
          [_x_rmesh + self._domain_shifts[idx] for idx in range(len(self._domain_shifts))], axis=0)
        else:
          _x_rmesh_extended = jnp.concatenate(
          [_x_rmesh + self._domain_shifts[idx] for idx in range(len(self._domain_shifts))], axis=0)
          _x_rmesh_extended_ = (_x_rmesh[None, :, :] + self._domain_shifts[:, None, :]).reshape(-1, 2)
          np.testing.assert_allclose(_x_rmesh_extended, _x_rmesh_extended_)
        tri = Delaunay(points=_x_rmesh_extended)

      else:
        tri = Delaunay(points=_x_rmesh)

      if self.mode == "fast":
        indptr, cols = tri.vertex_neighbor_vertices
        rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
        assert cols.shape == rows.shape
        _extended_edges = np.stack([rows, cols], -1) # [e,2]
        domain = _extended_edges // _rmesh_size # [e,2]
        edge   = _extended_edges % _rmesh_size # [e,2]
        is_row_0 = domain[:,0] == 0
        is_col_0 = domain[:,1] == 0
        is_all_0 = is_row_0 & is_col_0
        is_any_0 = is_row_0 | is_col_0 
        mask     = is_any_0 if self.periodic else is_all_0 # [e]
        edge     = (edge[:,0] << 32) & edge[:, 1]
        edge, index = np.unique(edge[mask], axis=0, return_index=True)
        edge     = np.stack([edge >> 32, edge & 0xFFFFFFFF], -1)
        domain  = domain[mask][index]
        try:
          edges   = edge   if edges   is None else np.concatenate([edges, edge   ], 0)
          domains = domain if domains is None else np.concatenate([domains, domain], 0)
        except:
          breakpoint()
      elif self.mode == "slow":
        # Construct a triangulation and get the edges
        _extended_edges = _get_edges_from_triangulation(tri)
        # Keep the relevant edges
        for edge in _extended_edges:
          domain = tuple([i // _rmesh_size for i in edge])
          edge = tuple([i % _rmesh_size for i in edge])
          if (domain == (0, 0)) or (self.periodic and (0 in domain)):
            if edge not in edges:
              domains.append(domain)
              edges.append(edge)

      else:
        # Construct a triangulation and get the edges
        _extended_edges = _get_edges_from_triangulation(tri)
        # Keep the relevant edges
        start_ptr = 0 if edges is None else len(edges)
        end_ptr   = 0 if edges is None else len(edges)
        for edge in _extended_edges:
          domain = tuple([i // _rmesh_size for i in edge])
          edge = tuple([i % _rmesh_size for i in edge])
          if (domain == (0, 0)) or (self.periodic and (0 in domain)):
            if edges is None or edge not in edges:
              if domains is None:
                domains = [domain]
              else:
                domains.append(domain)
              if edges is None:
                edges = [edge]
              else:
                edges.append(edge)
              end_ptr += 1
        
        indptr, cols = tri.vertex_neighbor_vertices
        rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
        assert cols.shape == rows.shape
        _extended_edges_ = np.stack([rows, cols], -1) # [e,2]
        domain_ = _extended_edges_ // _rmesh_size # [e,2]
        edge_   = _extended_edges_ % _rmesh_size # [e,2]
        is_row_0 = domain_[:,0] == 0
        is_col_0 = domain_[:,1] == 0
        is_all_0 = is_row_0 & is_col_0
        is_any_0 = is_row_0 | is_col_0 
        mask     = is_any_0 if self.periodic else is_all_0 # [e]
        
        edge_ = edge_[:, 0] << 32 | edge_[:, 1]
        edge_, index = np.unique(edge_[mask], axis=0, return_index=True)
        edge_ = np.stack([edge_ >> 32, edge_ & 0xFFFFFFFF], -1)
        domain_ = domain_[mask][index]
       
        try:
          edge   = np.array(edges[start_ptr:end_ptr])
          domain = np.array(domains[start_ptr:end_ptr])
          edge_id  = edge[:, 0] << 32 + edge[:, 1]
          edge_id_ = edge_[:,0] << 32 + edge_[:, 1]
          domain_id  = domain[:,0] << 32 + domain[:,1]
          domain_id_ = domain_[:,0] << 32 + domain[:,1] 
          edge_id  = np.sort(edge_id)
          edge_id_ = np.sort(edge_id_)
          domain_id= np.sort(domain_id)
          domain_id_=np.sort(domain_id_)
          np.testing.assert_allclose(edge_id,edge_id_)
          np.testing.assert_allclose(domain_id, domain_id_) 
        except:
          breakpoint()
    # Set the initial features
    if self.mode == "fast":
      edge_set, rmesh_node_set, _ = self._init_structural_features(
      x_sen=x_rmesh,
      x_rec=x_rmesh,
      idx_sen=edges[:,0],
      idx_rec=edges[:,1],
      max_edge_length=(2. * jnp.sqrt(x_rmesh.shape[1])),
      feats_sen=radius.reshape(-1, 1),
      feats_rec=radius.reshape(-1, 1),
      shifts=jnp.array(self._domain_shifts),
      domain_sen=domains[:,0],
      domain_rec=domains[:,1],
      )
    elif self.mode == "slow":
      edge_set, rmesh_node_set, _ = self._init_structural_features(
        x_sen=x_rmesh,
        x_rec=x_rmesh,
        idx_sen=[i for (i, j) in edges],
        idx_rec=[j for (i, j) in edges],
        max_edge_length=(2. * jnp.sqrt(x_rmesh.shape[1])),
        feats_sen=radius.reshape(-1, 1),
        feats_rec=radius.reshape(-1, 1),
        shifts=jnp.array(self._domain_shifts).squeeze(1),
        domain_sen=[i for (i, j) in domains],
        domain_rec=[j for (i, j) in domains],
      )
    else:
      edge_set, rmesh_node_set, _ = self._init_structural_features(
        x_sen=x_rmesh,
        x_rec=x_rmesh,
        idx_sen=[i for (i, j) in edges],
        idx_rec=[j for (i, j) in edges],
        max_edge_length=(2. * jnp.sqrt(x_rmesh.shape[1])),
        feats_sen=radius.reshape(-1, 1),
        feats_rec=radius.reshape(-1, 1),
        shifts=jnp.array(self._domain_shifts),
        domain_sen=[i for (i, j) in domains],
        domain_rec=[j for (i, j) in domains],
      )
    
      

    # Construct the graph
    graph = TypedGraph(
      context=Context(n_graph=jnp.array([1]), features=()),
      nodes={'rnodes': rmesh_node_set},
      edges={EdgeSetKey('r2r', ('rnodes', 'rnodes')): edge_set},
    )

    return graph

  def _build_r2p_graph(self, x_out: Array, x_rmesh: Array, r_min: Array) -> TypedGraph:
    """Constructrs the decoder graph (rmesh to pmesh)"""

    # Set the sub-region radii
    radius = self.overlap_factor_r2p * r_min

    # Get indices of supported points
    idx_nodes = self._get_supported_points(
      centers=x_rmesh,
      points=x_out,
      radii=radius,
    )

    # Get the initial features
    edge_set, rmesh_node_set, pmesh_node_set = self._init_structural_features(
      x_sen=x_rmesh,
      x_rec=x_out,
      idx_sen=idx_nodes[:, 1],
      idx_rec=idx_nodes[:, 0],
      max_edge_length=(2. * jnp.sqrt(x_rmesh.shape[1])),
      feats_sen=radius.reshape(-1, 1),
    )

    # Construct the graph
    graph = TypedGraph(
      context=Context(n_graph=jnp.array([1]), features=()),
      nodes={'pnodes': pmesh_node_set, 'rnodes': rmesh_node_set},
      edges={EdgeSetKey('r2p', ('rnodes', 'pnodes')): edge_set},
    )

    return graph

  def build(self, x_inp: Array, x_out: Array, domain: Array, key: Union[flax.typing.PRNGKey, None] = None) -> RegionInteractionGraphs:

    # Normalize coordinates in [-1, +1)
    x_inp = 2 * (x_inp - domain[0]) / (domain[1] - domain[0]) - 1
    x_out = 2 * (x_out - domain[0]) / (domain[1] - domain[0]) - 1

    # Randomly sub-sample pmesh to get rmesh
    if key is None: key = jax.random.PRNGKey(0)
    if self.periodic:
      x_rmesh = _subsample_pointset(key=key, x=x_inp, factor=self.subsample_factor)
    else:
      # NOTE: Always keep boundary nodes for non-periodic BC
      idx_bound = np.where((x_inp[:, 0] == -1) | (x_inp[:, 0] == +1) | (x_inp[:, 1] == -1) | (x_inp[:, 1] == +1))
      x_boundary = x_inp[idx_bound]
      x_internal = np.delete(x_inp, idx_bound, axis=0)
      x_rmesh_internal = _subsample_pointset(key=key, x=x_internal, factor=self.subsample_factor)
      x_rmesh, = shuffle_arrays(key=key, arrays=(jnp.concatenate([x_boundary, x_rmesh_internal]),))

    # Compute minimum support radius of each rmesh node
    r_min = self._compute_minimum_support_radii(x_rmesh)

    # Build the graphs
    graphs = RegionInteractionGraphs(
      p2r=self._build_p2r_graph(x_inp, x_rmesh, r_min),
      r2r=self._build_r2r_graph(x_rmesh, r_min),
      r2p=self._build_r2p_graph(x_out, x_rmesh, r_min),
    )

    return graphs

class Encoder(nn.Module):
  node_latent_size: int
  edge_latent_size: int
  mlp_hidden_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True
  p_edge_masking: float = .0

  def setup(self):
    self.gnn = DeepTypedGraphNet(
      embed_nodes=True,  # Embed raw features of all nodes
      embed_edges=True,  # Embed raw features of the edges
      edge_latent_size=dict(p2r=self.edge_latent_size),
      node_latent_size=dict(rnodes=self.node_latent_size, pnodes=self.node_latent_size),
      mlp_hidden_size=self.mlp_hidden_size,
      mlp_num_hidden_layers=self.mlp_hidden_layers,
      num_message_passing_steps=1,
      use_layer_norm=self.use_layer_norm,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=True,
      aggregate_edges_for_nodes_fn='segment_mean',
      aggregate_normalization=None,
    )

  def __call__(self,
    graph: TypedGraph,
    pnode_features: Array,
    tau: Union[None, float],
    key: Union[flax.typing.PRNGKey, None] = None,
  ) -> tuple[Array, Array]:
    """Runs the p2r GNN, extracting latent physical and regional nodes."""

    # Get batch size
    batch_size = pnode_features.shape[1]

    # Concatenate node structural features with input features
    pnodes = graph.nodes['pnodes']
    rnodes = graph.nodes['rnodes']
    new_pnodes = pnodes._replace(
      features=jnp.concatenate([
        pnode_features,
        _add_batch_second_axis(pnodes.features.astype(pnode_features.dtype), batch_size)
      ], axis=-1)
    )
    # To make sure capacity of the embedded is identical for the physical nodes and
    # the regional nodes, we also append some dummy zero input features for the
    # regional nodes.
    dummy_rnode_features = jnp.zeros(
        (rnodes.features.shape[0],) + pnode_features.shape[1:],
        dtype=pnode_features.dtype)
    new_rnodes = rnodes._replace(
      features=jnp.concatenate([
        dummy_rnode_features,
        _add_batch_second_axis(
          rnodes.features.astype(dummy_rnode_features.dtype), batch_size)
      ], axis=-1)
    )

    # Get edges
    p2r_edges_key = graph.edge_key_by_name('p2r')
    edges = graph.edges[p2r_edges_key]
    # Drop out edges randomly with the given probability
    if key is not None:
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[0])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        key=key, arrays=[edges.features, edges.indices.senders, edges.indices.receivers])
      new_edge_features = new_edge_features[:n_edges_after]
      new_edge_senders = new_edge_senders[:n_edges_after]
      new_edge_receivers = new_edge_receivers[:n_edges_after]
    else:
      n_edges_after = edges.features.shape[0]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    # Change edge feature dtype
    new_edge_features = new_edge_features.astype(dummy_rnode_features.dtype)
    # Broadcast edge structural features to the required batch size
    new_edge_features = _add_batch_second_axis(new_edge_features, batch_size)
    # Build new edge set
    new_edges = EdgeSet(
      n_edge=jnp.array([n_edges_after]),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    input_graph = graph._replace(
      edges={p2r_edges_key: new_edges},
      nodes={
        'pnodes': new_pnodes,
        'rnodes': new_rnodes
      })

    # Run the GNN
    p2r_out = self.gnn(input_graph, condition=tau)
    latent_rnodes = p2r_out.nodes['rnodes'].features
    latent_pnodes = p2r_out.nodes['pnodes'].features

    return latent_rnodes, latent_pnodes

class Processor(nn.Module):
  steps: int
  node_latent_size: int
  edge_latent_size: int
  mlp_hidden_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True
  p_edge_masking: float = .0

  def setup(self):
    self.gnn = DeepTypedGraphNet(
      embed_nodes=False,  # Node features already embdded by previous layers
      embed_edges=True,  # Embed raw features of the edges
      edge_latent_size=dict(r2r=self.edge_latent_size),
      node_latent_size=dict(rnodes=self.node_latent_size),
      mlp_hidden_size=self.mlp_hidden_size,
      mlp_num_hidden_layers=self.mlp_hidden_layers,
      num_message_passing_steps=self.steps,
      use_layer_norm=True,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      # NOTE: segment_mean because number of edges is not balanced
      aggregate_edges_for_nodes_fn='segment_mean',
    )

  def __call__(self,
    graph: TypedGraph,
    rnode_features: Array,
    tau: Union[None, float],
    key: Union[flax.typing.PRNGKey, None] = None,
  ) -> Array:
    """Runs the r2r GNN, extracting updated latent regional nodes."""

    # Get batch size
    batch_size = rnode_features.shape[1]

    # Replace the node features
    # NOTE: We don't need to add the structural node features, because these are
    # already part of  the latent state, via the original p2r gnn.
    rnodes = graph.nodes['rnodes']
    new_rnodes = rnodes._replace(features=rnode_features)

    # Get edges
    r2r_edges_key = graph.edge_key_by_name('r2r')
    # NOTE: We are assuming here that the r2r gnn uses a single set of edge keys
    # named 'r2r' for the edges and that it uses a single set of nodes named 'rnodes'
    msg = ('The setup currently requires to only have one kind of edge in the mesh GNN.')
    assert len(graph.edges) == 1, msg
    edges = graph.edges[r2r_edges_key]
    # Drop out edges randomly with the given probability
    # NOTE: We need the structural edge features, because it is the first
    # time we are seeing this particular set of edges.
    if key is not None:
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[0])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        key=key, arrays=[edges.features, edges.indices.senders, edges.indices.receivers])
      new_edge_features = new_edge_features[:n_edges_after]
      new_edge_senders = new_edge_senders[:n_edges_after]
      new_edge_receivers = new_edge_receivers[:n_edges_after]
    else:
      n_edges_after = edges.features.shape[0]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    # Change edge feature dtype
    new_edge_features = new_edge_features.astype(rnode_features.dtype)
    # Broadcast edge structural features to the required batch size
    new_edge_features = _add_batch_second_axis(new_edge_features, batch_size)
    # Build new edge set
    new_edges = EdgeSet(
      n_edge=jnp.array([n_edges_after]),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    # Build the graph
    input_graph = graph._replace(
      edges={r2r_edges_key: new_edges},
      nodes={'rnodes': new_rnodes},
    )

    # Run the GNN
    output_graph = self.gnn(input_graph, condition=tau)
    output_rnodes = output_graph.nodes['rnodes'].features

    return output_rnodes

class Decoder(nn.Module):
  variable_mesh: bool
  num_outputs: int
  node_latent_size: int
  edge_latent_size: int
  mlp_hidden_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True
  p_edge_masking: float = .0

  def setup(self):
    self.gnn = DeepTypedGraphNet(
    # NOTE: with variable mesh, the output pnode features must be embedded
    embed_nodes=(dict(pnodes=True) if self.variable_mesh else False),
    embed_edges=True,  # Embed raw features of the edges
    # Require a specific node dimensionaly for the physical node outputs
    # NOTE: This triggers the independent mapping for pnodes
    node_output_size=dict(pnodes=self.num_outputs),
    edge_latent_size=dict(r2p=self.edge_latent_size),
    node_latent_size=dict(rnodes=self.node_latent_size, pnodes=self.node_latent_size),
    mlp_hidden_size=self.mlp_hidden_size,
    mlp_num_hidden_layers=self.mlp_hidden_layers,
    num_message_passing_steps=1,
    use_layer_norm=True,
    conditioned_normalization=self.conditioned_normalization,
    cond_norm_hidden_size=self.cond_norm_hidden_size,
    include_sent_messages_in_node_update=False,
    activation='swish',
    f32_aggregation=False,
    # NOTE: segment_mean because number of edges is not balanced
    aggregate_edges_for_nodes_fn='segment_mean',
  )

  def __call__(self,
    graph: TypedGraph,
    rnode_features: Array,
    pnode_features: Array,
    tau: Union[None, float],
    key: Union[flax.typing.PRNGKey, None] = None,
  ) -> Array:
    """Runs the r2p GNN, extracting the output physical nodes."""

    # Get batch size
    batch_size = rnode_features.shape[1]

    # NOTE: We don't need to add the structural node features, because these are
    # already part of the latent state, via the original p2r gnn.
    rnodes = graph.nodes['rnodes']
    pnodes = graph.nodes['pnodes']
    new_rnodes = rnodes._replace(features=rnode_features)
    if self.variable_mesh:
      # NOTE: We can't use latent pnodes of the input mesh for the output mesh
      # TRY: Make sure that this does not harm the performance with fixed mesh
      # If it works, change the architecture, flowcharts, etc.
      new_pnodes = pnodes._replace(features=_add_batch_second_axis(pnodes.features, batch_size))
    else:
      new_pnodes = pnodes._replace(features=pnode_features)

    # Get edges
    r2p_edges_key = graph.edge_key_by_name('r2p')
    edges = graph.edges[r2p_edges_key]
    # Drop out edges randomly with the given probability
    if key is not None:
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[0])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        key=key, arrays=[edges.features, edges.indices.senders, edges.indices.receivers])
      new_edge_features = new_edge_features[:n_edges_after]
      new_edge_senders = new_edge_senders[:n_edges_after]
      new_edge_receivers = new_edge_receivers[:n_edges_after]
    else:
      n_edges_after = edges.features.shape[0]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    # Change edge feature dtype
    new_edge_features = new_edge_features.astype(pnode_features.dtype)
    # Broadcast edge structural features to the required batch size
    new_edge_features = _add_batch_second_axis(new_edge_features, batch_size)
    # Build new edge set
    new_edges = EdgeSet(
      n_edge=jnp.array([n_edges_after]),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    # Build the new graph
    input_graph = graph._replace(
      edges={r2p_edges_key: new_edges},
      nodes={
        'rnodes': new_rnodes,
        'pnodes': new_pnodes
      })

    # Run the GNN
    output_graph = self.gnn(input_graph, condition=tau)
    output_pnodes = output_graph.nodes['pnodes'].features

    return output_pnodes

class RIGNO(AbstractOperator):
  """TODO: Add docstrings"""

  num_outputs: int
  processor_steps: int = 18
  node_latent_size: int = 128
  edge_latent_size: int = 128
  mlp_hidden_layers: int = 1
  mlp_hidden_size: int = 128
  concatenate_t: bool = True
  concatenate_tau: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: int = 16
  p_edge_masking: int = 0.5

  def _check_coordinates(self, x: Array) -> None:
    assert x is not None
    assert x.ndim == 2
    assert x.shape[1] <= 3
    assert x.min() >= -1
    assert x.max() <= +1

  def _check_function(self, u: Array, x: Array) -> None:
    assert u is not None
    assert u.ndim == 4
    assert u.shape[1] == 1
    assert u.shape[2] == x.shape[2], f'u: {u.shape}, x: {x.shape}'

  def setup(self):
    # NOTE: There are a few architectural considerations for variable mesh
    # NOTE: variable_mesh=True means that the input and the output mesh can be different
    # NOTE: Check usages of this attribute
    self.variable_mesh = False

    self.encoder = Encoder(
      edge_latent_size=self.edge_latent_size,
      node_latent_size=self.node_latent_size,
      mlp_hidden_size=self.mlp_hidden_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      p_edge_masking=self.p_edge_masking,
      name='encoder',
    )

    self.processor = Processor(
      steps=self.processor_steps,
      edge_latent_size=self.edge_latent_size,
      node_latent_size=self.node_latent_size,
      mlp_hidden_size=self.mlp_hidden_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      p_edge_masking=self.p_edge_masking,
      name='processor',
    )

    self.decoder = Decoder(
      variable_mesh=self.variable_mesh,
      num_outputs=self.num_outputs,
      edge_latent_size=self.edge_latent_size,
      node_latent_size=self.node_latent_size,
      mlp_hidden_size=self.mlp_hidden_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      p_edge_masking=self.p_edge_masking,
      name='decoder',
    )

  @staticmethod
  def _reorder_features(feats: Array, num_nodes: int) -> Array:
    batch_size = feats.shape[1]
    num_feats = feats.shape[-1]
    feats = feats.reshape(num_nodes, batch_size, 1, num_feats)
    output = jnp.moveaxis(feats, source=(0, 1, 2), destination=(2, 0, 1))
    return output

  def _encode_process_decode(self,
    graphs: RegionInteractionGraphs,
    pnode_features: Array,
    tau: Union[None, float],
    key: flax.typing.PRNGKey = None,
  ) -> Array:

    # Transfer data for the physical mesh to the regional mesh
    # -> [num_nodes, batch_size, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    (latent_rnodes, latent_pnodes) = self.encoder(graphs.p2r, pnode_features, tau, key=subkey)
    self.sow(
      col='intermediates', name='pnodes_encoded',
      value=self._reorder_features(latent_pnodes, graphs.p2r.nodes['pnodes'].features.shape[0])
    )
    self.sow(
      col='intermediates', name='rnodes_encoded',
      value=self._reorder_features(latent_rnodes, graphs.p2r.nodes['rnodes'].features.shape[0])
    )

    # Run message-passing in the regional mesh
    # -> [num_rnodes, batch_size, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    updated_latent_rnodes = self.processor(graphs.r2r, latent_rnodes, tau, key=subkey)
    self.sow(
      col='intermediates', name='rnodes_processed',
      value=self._reorder_features(updated_latent_rnodes, graphs.r2r.nodes['rnodes'].features.shape[0])
    )

    # Transfer data from the regional mesh to the physical mesh
    # -> [num_pnodes_out, batch_size, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    output_pnodes = self.decoder(graphs.r2p, updated_latent_rnodes, latent_pnodes, tau, key=subkey)
    self.sow(
      col='intermediates', name='pnodes_decoded',
      value=self._reorder_features(output_pnodes, graphs.r2p.nodes['pnodes'].features.shape[0])
    )

    return output_pnodes

  def call(self,
    inputs: Inputs,
    graphs: RegionInteractionGraphs,
    key: flax.typing.PRNGKey = None,
  ) -> Array:
    """Inputs must be of shape [batch_size, 1, num_physical_nodes, num_inputs]"""

    # Check input functions
    self._check_function(inputs.u, x=inputs.x_inp)
    if inputs.c is not None:
      self._check_function(inputs.c, x=inputs.x_inp)
    assert inputs.u.shape[3] == self.num_outputs

    # Read dimensions
    batch_size = inputs.u.shape[0]
    num_pnodes_inp = inputs.x_inp.shape[2]
    num_pnodes_out = inputs.x_out.shape[2]

    # Prepare the time channel
    if self.concatenate_t:
      assert inputs.t is not None
      t_inp = jnp.array(inputs.t, dtype=jnp.float32)
      if t_inp.ndim == 4:
        t_inp = t_inp[:, :, 0, 0]
      if t_inp.size == 1:
        t_inp = jnp.tile(t_inp.reshape(1, 1), reps=(batch_size, 1))
    # Prepare the time difference channel
    if self.concatenate_tau:
      assert inputs.tau is not None
      tau = jnp.array(inputs.tau, dtype=jnp.float32)
      if tau.ndim == 4:
        tau = tau[:, :, 0, 0]
      if tau.size == 1:
        tau = jnp.tile(tau.reshape(1, 1), reps=(batch_size, 1))
    else:
      tau = None

    # Concatenate the known coefficients to the channels of the input function
    if inputs.c is None:
      u_inp = inputs.u
    else:
      u_inp = jnp.concatenate([inputs.u, inputs.c], axis=-1)

    # Prepare the physical node features
    # u -> [num_pnodes_inp, batch_size, num_inputs]
    pnode_features = jnp.moveaxis(u_inp,
      source=(0, 1, 2, 3), destination=(1, 3, 0, 2)
    ).squeeze(axis=3)

    # Concatente with forced features
    pnode_features_forced = []
    if self.concatenate_tau:
      pnode_features_forced.append(jnp.tile(tau, reps=(num_pnodes_inp, 1, 1)))
    if self.concatenate_t:
      pnode_features_forced.append(jnp.tile(t_inp, reps=(num_pnodes_inp, 1, 1)))
    pnode_features = jnp.concatenate([pnode_features, *pnode_features_forced], axis=-1)

    # Run the GNNs
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    output_pnodes = self._encode_process_decode(
      graphs=graphs, pnode_features=pnode_features, tau=tau, key=subkey)

    # Reshape the output to [batch_size, 1, num_pnodes_out, num_outputs]
    # [num_pnodes_out, batch_size, num_outputs] -> u
    output = self._reorder_features(output_pnodes, num_pnodes_out)
    self._check_function(output, x=inputs.x_out)

    return output

def _subsample_pointset(key, x: Array, factor: float) -> Array:
  x = jnp.array(x)
  x_shuffled, = shuffle_arrays(key, [x])
  return x_shuffled[:int(x.shape[0] / factor)]

def _get_edges_from_triangulation(tri: Delaunay, bidirectional: bool = True):
  indptr, indices = tri.vertex_neighbor_vertices
  edges = [(k, l) for k in range(tri.points.shape[0]) for l in indices[indptr[k]:indptr[k+1]]]
  if bidirectional:
    edges += [(l, k) for (k, l) in edges]
  return edges

def _compute_triangulation_medians(tri: Delaunay) -> Array:
  # Only in 2D

  edges = np.zeros(shape=tri.simplices.shape) # [N, 3]
  medians = np.zeros(shape=tri.simplices.shape)
  for i in range(tri.simplices.shape[1]):
    points = tri.points[np.delete(tri.simplices, i, axis=1)]
    points = [p.squeeze(1) for p in np.split(points, axis=1, indices_or_sections=2)]
    edges[:, i] = np.linalg.norm(np.subtract(*points), axis=1)
  for i in range(tri.simplices.shape[1]):
    medians[:, i] = .67 * np.sqrt((2 * np.sum(np.power(np.delete(edges, i, axis=1), 2), axis=1) - np.power(edges[:, i], 2)) / 4)

  return medians

def _add_batch_second_axis(data: Array, batch_size: int) -> Array:
  """
  Adds a batch axis by repeating the input

  input: [leading_dim, trailing_dim]
  output: [leading_dim, batch, trailing_dim]
  """

  assert data.ndim == 2
  ones = jnp.ones([batch_size, 1], dtype=data.dtype)
  return data[:, None] * ones
