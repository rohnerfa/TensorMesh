"""Interpolation-node generators and facet-basis index helpers.

Builds the reference-element basis-node coordinates (``*_basis``) and the
flat connectivity arrays that map element basis indices to facet basis
indices (``*_facet_basis_index``, ``mix_facet_basis_index``, ``edge_index``).
All routines are wired into the corresponding
:meth:`~tensormesh.Element.get_basis` / :meth:`~tensormesh.Element.get_facet`
overrides on the element subclasses; they are **not** part of the public
API and may change between releases.
"""
from typing import Tuple, Union, Sequence, Optional
import math

import torch
import numpy as np

from .types import Tensorx2


#######################################
# Linear Space Interpolation Functions
#######################################

def lin_linspace(edge:torch.Tensor, 
                 order:int = 1, 
                 include_boundary:bool=True,
                 )->torch.Tensor:
    r"""

    Generate evenly spaced points along a line segment.

    Creates a linear interpolation between two endpoints, generating points that are evenly
    spaced along the line segment. The number of points is determined by the order parameter.

    For order=1, returns just the endpoints if include_boundary=True, or an empty tensor if
    include_boundary=False.

    For order>1, returns order+1 evenly spaced points (including endpoints) if include_boundary=True,
    or order-1 interior points if include_boundary=False.

    The function handles both single edges and batches of edges.

    Examples
    --------
    >>> edge = torch.tensor([[0., 0.], [1., 1.]])  # Single edge from (0,0) to (1,1)
    >>> points = lin_linspace(edge, order=3, include_boundary=True)
    >>> print(points)
    tensor([[0.0000, 0.0000],  # Start point
            [0.3333, 0.3333],  # Interior point 1
            [0.6667, 0.6667],  # Interior point 2
            [1.0000, 1.0000]]) # End point

    >>> points = lin_linspace(edge, order=3, include_boundary=False)
    >>> print(points)
    tensor([[0.3333, 0.3333],  # Interior point 1
            [0.6667, 0.6667]]) # Interior point 2

    Parameters
    ----------
    edge: torch.Tensor [n_batch, 2, n_dim] or [2, n_dim]
    order: int 
            the order of the linspace, should be at least 1, default is 1
    include_boundary: bool
            whether to include the boundary

    Returns
    -------
    if include_boundary
        torch.Tensor [n_batch, order+1, n_dim] or [order+1, n_dim]
    else
        torch.Tensor [n_batch, order-1, n_dim] or [order-1, n_dim]
    """
    assert order >= 1, f"order should be at least 1, but got {order}"
    assert edge.shape[-2] == 2, f"edge should have 2 points, but got {edge.shape[-2]}"
    
    dtype  = edge.dtype
    device = edge.device

    if edge.dim() == 2: # edge [2, n_dim]

        src, dst = edge  # [n_dim]
        delta = (dst - src) 
        x = torch.linspace(0, 1, order+1, dtype=dtype, device=device)[:,None].repeat(1,edge.shape[1]) # [order+1, n_dim]
        x = x * delta[None, :] + src[None, :]
        x = x if include_boundary else x[1:-1]

    elif edge.dim() == 3: # edge [n_batch, 2, n_dim]

        src, dst = edge[:, 0], edge[:, 1] # [n_batch, n_dim]
        delta = (dst - src) 
        x = torch.linspace(0, 1, order+1, dtype=dtype, device=device)[None, :, None].repeat(edge.shape[0], 1, edge.shape[2]) # [n_batch, order+1, n_dim]
        x = x * delta[:, None, :] + src[:, None, :]
        x = x if include_boundary else x[:, 1:-1]
        
    else:
        raise NotImplementedError(f"edge should have 2 or 3 dimensions, but got {edge.dim()}")
    
    return x 

def quad_linspace(quad:torch.Tensor, 
                  order:int = 1, 
                  include_boundary:bool=True
                  )->torch.Tensor:
    r"""
    Generate evenly spaced points in a quadrilateral element.

    This function takes a quadrilateral element defined by 4 vertices and generates a grid of evenly spaced points.
    The points can optionally include or exclude the boundary points.

    Examples
    --------
    >>> quad = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    >>> points = quad_linspace(quad, order=2)
    >>> print(points)
    tensor([[0.0000, 0.0000],  # Bottom left
            [0.5000, 0.0000],  # Bottom middle 
            [1.0000, 0.0000],  # Bottom right
            [0.0000, 0.5000],  # Middle left
            [0.5000, 0.5000],  # Center
            [1.0000, 0.5000],  # Middle right 
            [0.0000, 1.0000],  # Top left
            [0.5000, 1.0000],  # Top middle
            [1.0000, 1.0000]]) # Top right

    >>> points = quad_linspace(quad, order=2, include_boundary=False) 
    >>> print(points)
    tensor([[0.5000, 0.5000]]) # Only interior point

    Parameters
    ----------
    quad: torch.Tensor [n_batch, 4, n_dim] or [4, n_dim]
    order: int 
    include_boundary: bool
        whether to include the boundary

    Returns
    -------
    if include_boundary
        torch.Tensor [n_batch, (order+1)^2, n_dim] or [(order+1)^2, n_dim]
    else
        torch.Tensor [n_batch, (order-1)^2, n_dim] or [(order-1)^2, n_dim]
    """

    if quad.dim() == 3: # quad [n_batch, 4, n_dim]
        dx = quad[:, [0,1]] - quad[:, 0:1] # [n_batch, 2, n_dim]
        dy = quad[:, [0,2]] - quad[:, 0:1]
    else: # quad [4, n_dim]
        dx = quad[[0,1]] - quad[0:1]
        dy = quad[[0,2]] - quad[0:1]
    dx = lin_linspace(dx, order, include_boundary) # [n_batch, order+1, n_dim] or [order+1, n_dim]
    dy = lin_linspace(dy, order, include_boundary) # [n_batch, order+1, n_dim] or [order+1, n_dim]
    
    n_points = dx.shape[-2]
    if quad.dim() == 3: # quad [n_batch, 4, n_dim]
        n_batch, _, n_dim = quad.shape
        dx = dx[:, None, :, :].repeat(1, n_points, 1, 1).reshape(n_batch, -1, n_dim)
        dy = dy[:, :, None, :].repeat(1, 1, n_points, 1).reshape(n_batch, -1, n_dim)
    else: # quad [4, n_dim]
        _, n_dim = quad.shape
        dx = dx[None, :, :].repeat(n_points, 1, 1).reshape(-1, n_dim)
        dy = dy[:, None, :].repeat(1, n_points, 1).reshape(-1, n_dim)
    
    if quad.dim() == 3:
        x = dx + dy + quad[:, 0:1] # [n_batch, (order+1)**2, n_dim]
        assert x.shape[1] == n_points**2
    else:
        x = dx + dy + quad[0:1]
        assert x.shape[0] == n_points**2
    return x 
         
def tri_linspace(tri:torch.Tensor, 
                 order:int = 1, 
                 include_boundary:bool = True
                 )->torch.Tensor:
    r"""
    Return evenly spaced points in a triangle.

    The points are evenly spaced in barycentric coordinates. If include_boundary is True,
    points on the boundary are included. Otherwise, only interior points are returned.

    Examples
    --------
    >>> tri = torch.tensor([[0., 0.], [1., 0.], [0., 1.]]) # Unit triangle
    >>> points = tri_linspace(tri, order=2)
    >>> print(points)
    tensor([[0.0000, 0.0000],  # Corner points
            [0.5000, 0.0000],
            [1.0000, 0.0000],
            [0.0000, 0.5000],
            [0.5000, 0.5000],
            [0.0000, 1.0000]])

    >>> points = tri_linspace(tri, order=2, include_boundary=False)
    >>> print(points)
    tensor([[0.3333, 0.3333]]) # Only interior point
    Parameters
    ----------
    tri: torch.Tensor [n_batch, 3, n_dim] or [3, n_dim]
    order: int 
    include_boundary: bool
        whether to include the boundary

    Returns
    -------
    torch.Tensor [n_batch, (order+1)*(order+2)/2, n_dim] or [(order+1)*(order+2)/2, n_dim]
    """
    dtype = tri.dtype
    device = tri.device
    assert order >= 1, f"the order should be at least 1 but got {order}"
    if not include_boundary:
        assert order >= 3, f"if include_boundary is False, the order should be at least 3, but got {order}"
    assert tri.shape[-2] == 3, f"tri should have 3 points, but got {tri.shape[-2]}"

    x = quad_linspace(tri, order, include_boundary) # [n_batch, (order+1)**2 or (order-1)**2, n_dim]
    n_points = math.sqrt(x.shape[-2])
    i, j = torch.meshgrid(torch.arange(n_points), torch.arange(n_points), indexing='ij') # [(order+1)**2]
    i, j = i.flatten().to(device), j.flatten().to(device)
    
    if not include_boundary:
        n_points -= 1

    if tri.dim() == 3: # tri [n_batch, 3, n_dim]
    
        x = x[:, i + j < n_points]
       
    elif tri.dim() == 2: # tri [3, n_dim]
        
        x = x[i + j < n_points]
        
    else:
        raise Exception(f"tri should have 2 or 3 dimensions, but got {tri.dim()}")

    return x

def hex_linspace(hex:Optional[torch.Tensor] = None, 
                 order:int = 1, 
                 include_boundary:bool = True
                 )->torch.Tensor:
    r"""
    Generate evenly spaced points inside a hexahedron.

    For order k, points are placed at:

    * Vertices (order 1)
    * Edge points at :math:`\{1/k, 2/k, ..., (k-1)/k\}` along each edge (order >= 2)
    * Face points in a rectangular pattern on each face (order >= 3)
    * Interior points (order >= 4)

    When include_boundary=False, only interior points are returned.

    The number of points generated is:

    .. math::

        N_{points} = \begin{cases}
            (n+1)^3 & \text{if include\_boundary=True} \\
            (n-1)^3 & \text{if include\_boundary=False}
        \end{cases}

    where n is the order.

    The points are arranged in a regular 3D grid pattern.

    Examples
    --------
    >>> hex = torch.tensor([
            [0.0, 0.0, 0.0],  # vertex 0 
            [1.0, 0.0, 0.0],  # vertex 1
            [0.0, 1.0, 0.0],  # vertex 2
            [0.0, 0.0, 1.0],  # vertex 3
            [1.0, 1.0, 0.0],  # vertex 4
            [1.0, 0.0, 1.0],  # vertex 5
            [0.0, 1.0, 1.0],  # vertex 6
            [1.0, 1.0, 1.0]]) # vertex 7

    >>> points = hex_linspace(hex, order=2, include_boundary=True)
    >>> print(points)
    tensor([[0.0000, 0.0000, 0.0000],  # vertex 0
            [0.5000, 0.0000, 0.0000],  # edge 01 midpoint
            [1.0000, 0.0000, 0.0000],  # vertex 1
            [0.0000, 0.5000, 0.0000],  # edge 02 midpoint
            [0.5000, 0.5000, 0.0000],  # face center
            [1.0000, 0.5000, 0.0000],  # edge 14 midpoint
            [0.0000, 1.0000, 0.0000],  # vertex 2
            [0.5000, 1.0000, 0.0000],  # edge 24 midpoint
            [1.0000, 1.0000, 0.0000],  # vertex 4
            [0.0000, 0.0000, 0.5000],  # edge 03 midpoint
            [0.5000, 0.0000, 0.5000],  # face center
            [1.0000, 0.0000, 0.5000],  # edge 15 midpoint
            [0.0000, 0.5000, 0.5000],  # face center
            [0.5000, 0.5000, 0.5000],  # cell center
            [1.0000, 0.5000, 0.5000],  # face center
            [0.0000, 1.0000, 0.5000],  # edge 26 midpoint
            [0.5000, 1.0000, 0.5000],  # face center
            [1.0000, 1.0000, 0.5000],  # edge 46 midpoint
            [0.0000, 0.0000, 1.0000],  # vertex 3
            [0.5000, 0.0000, 1.0000],  # edge 35 midpoint
            [1.0000, 0.0000, 1.0000],  # vertex 5
            [0.0000, 0.5000, 1.0000],  # edge 36 midpoint
            [0.5000, 0.5000, 1.0000],  # face center
            [1.0000, 0.5000, 1.0000],  # edge 56 midpoint
            [0.0000, 1.0000, 1.0000],  # vertex 6
            [0.5000, 1.0000, 1.0000],  # edge 67 midpoint
            [1.0000, 1.0000, 1.0000]]) # vertex 7

    >>> points = hex_linspace(hex, order=2, include_boundary=False)
    >>> print(points)
    tensor([[0.5000, 0.5000, 0.5000]]) # Only interior point
    Parameters
    ----------
    hex: Optional[torch.Tensor] 
        [n_batch, 8 or 6 or 5 or 4, n_dim] or [8 or 6 or 5 or 4, n_dim]
    
    order: int 
    include_boundary: bool
        whether to include the boundary

    Returns
    -------
    torch.Tensor [n_batch, (order+1)^3 or (order-1)^3, n_dim] or [(order+1)^3 or (order-1)^3, n_dim]
    """
    if hex is None:
        hex = torch.tensor([[0.0, 0.0, 0.0],  # vertex 0 
                           [1.0, 0.0, 0.0],  # vertex 1
                           [0.0, 1.0, 0.0],  # vertex 2 
                           [1.0, 1.0, 0.0],  # vertex 3
                           [0.0, 0.0, 1.0],  # vertex 4
                           [1.0, 0.0, 1.0],  # vertex 5
                           [0.0, 1.0, 1.0],  # vertex 6
                           [1.0, 1.0, 1.0]]) # vertex 7

    # if hex.dim() == 3:
    #     o  = hex[:, 0:1]
    #     dx = hex[:, [0,1]] - o
    #     dy = hex[:, [0,2]] - o
    #     if hex.shape[1] > 4: # for pyr and pri
    #         dz = hex[:, [0,4]] - o
    #     else:
    #         dz = hex[:, [0,3]] - o
    # else:
    #     o  = hex[0:1]
    #     dx = hex[[0,1]] - o
    #     dy = hex[[0,2]] - o
    #     if hex.shape[0] > 4:
    #         dz = hex[[0,4]] - o
    #     else:
    #         dz = hex[[0,3]] - o

    o  = hex[..., 0:1, :]
    dx = hex[..., [0, 1], :] - o
    dy = hex[..., [0, 2], :] - o
    if hex.shape[-2] == 4: # tet
        dz = hex[..., [0,3], :] - o
    elif hex.shape[-2] == 5: # pyr 
        dz = hex[..., [0,4], :] - o
    elif hex.shape[-2] == 6: # pri
        dz = hex[..., [0,3], :] - o 
    elif hex.shape[-2] == 8: # hex 
        dz = hex[..., [0,4], :] - o
    else:
        raise ValueError(f"Invalid number of vertices {hex.shape[-2]}. Must be 4 (tet), 5 (pyr), 6 (pri), or 8 (hex).")

    dx = lin_linspace(dx, order, include_boundary) # [n_batch, order+1 or order-1, n_dim] or [order+1 or order-1, n_dim]
    dy = lin_linspace(dy, order, include_boundary) # [n_batch, order+1 or order-1, n_dim] or [order+1 or order-1, n_dim]
    dz = lin_linspace(dz, order, include_boundary) # [n_batch, order+1 or order-1, n_dim] or [order+1 or order-1, n_dim]
    n_points = dx.shape[-2]

    if hex.dim() == 3:

        n_batch, _, n_dim = hex.shape
        dx = dx[:, None, None, :, :].repeat(1, n_points, n_points, 1, 1).reshape(n_batch, -1, n_dim)
        dy = dy[:, None, :, None, :].repeat(1, n_points, 1, n_points, 1).reshape(n_batch, -1, n_dim)
        dz = dz[:, :, None, None, :].repeat(1, 1, n_points, n_points, 1).reshape(n_batch, -1, n_dim)

        x = dx + dy + dz + o # [n_batch, (order+1)**3, n_dim]

    else:
        n_dim = hex.shape[-1]
        dx = dx[None, None, :, :].repeat(n_points, n_points, 1, 1).reshape(-1, n_dim)
        dy = dy[None, :, None, :].repeat(n_points, 1, n_points, 1).reshape(-1, n_dim)
        dz = dz[:, None, None, :].repeat(1, n_points, n_points, 1).reshape(-1, n_dim)

        x = dx + dy + dz + o # [(order+1)**3, n_dim]

    return x 

def tet_linspace(tet:torch.Tensor, 
                 order:int = 1, 
                 include_boundary:bool = True
                 )->torch.Tensor:
    r"""
    Generate evenly spaced points inside a tetrahedron.

    The points are generated by first creating a regular grid in a cube, then filtering points 
    to keep only those inside the tetrahedron. For order k, points are placed at:

    * Vertices (order 1)
    * Edge points at :math:`\{1/k, 2/k, ..., (k-1)/k\}` along each edge (order >= 2)
    * Face points in a triangular pattern (order >= 3) 
    * Interior points (order >= 4)

    When include_boundary=False, only interior points are returned.

    Examples
    --------
    >>> import torch
    >>> tet = torch.tensor([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
    >>> points = tet_linspace(tet, order=2)
    >>> print(points.shape)
    torch.Size([10, 3])  # 4 vertices + 6 edge midpoints

    >>> points = tet_linspace(tet, order=5, include_boundary=False)
    >>> print(points.shape) 
    torch.Size([4, 3])  # Four interior points

    The points are generated by first creating a regular grid in a cube, then filtering
    points to keep only those inside the tetrahedron. For order k, points are placed at:
    - Vertices (order 1)
    - Edge points at 1/k, 2/k, ..., (k-1)/k along each edge (order >= 2) 
    - Face points in a triangular pattern (order >= 3)
    - Interior points (order >= 4)

    When include_boundary=False, only interior points are returned.
    
    Parameters
    ----------
    tet: torch.Tensor [n_batch, 4, n_dim] or [4, n_dim]
    order: int 
    include_boundary: bool
        whether to include the boundary

    Returns
    -------
    torch.Tensor [n_batch, (order+1)*(order+2)*(order+3)/6, n_dim] or [ (order+1)*(order+2)*(order+3)/6, n_dim]
    """
    device = tet.device
    assert order >= 1, f"the order should be at least 1 but got {order}"
    if not include_boundary:
        assert order >= 3, f"if include_boundary is False, the order should be at least 3, but got {order}"
    
    x = hex_linspace(tet, order, include_boundary) # [n_batch, (order+1)**3 or (order-1)**3, n_dim]
    n_points = order + 1 if include_boundary else order - 1
    k,j,i = torch.meshgrid(torch.arange(n_points), 
                           torch.arange(n_points), 
                           torch.arange(n_points), 
                           indexing='ij')
    i,j,k = i.flatten().to(device), j.flatten().to(device), k.flatten().to(device)
    

    if include_boundary:
        mask = (i + j + k) < n_points 
    else:
        mask = (i + j + k < n_points - 2)

    return x[..., mask, :]


def pyr_linspace(pyr:torch.Tensor, 
                 order:int = 1, 
                 include_boundary:bool=True
                 )->torch.Tensor:
    r"""Generate linearly spaced points inside a pyramid element.

    The points are arranged in a structured pattern with:

    - Vertex points (order >= 1)
    - Edge points in a linear pattern (order >= 2) 
    - Face points in a triangular/quadrilateral pattern (order >= 3)
    - Interior points (order >= 4)

    When include_boundary=False, only interior points are returned.

    The pyramid element has a square base and triangular faces meeting at an apex.
    The points are distributed in a tensor-product pattern that gets narrower 
    towards the apex.

    The number of points follows the pattern:

    .. math::

        N_{points} = \begin{cases}
            \frac{(n+1)(n+2)(2n+3)}{6} & \text{if include\_boundary=True} \\
            \frac{(n-1)(n)(2n-3)}{6} & \text{if include\_boundary=False}
        \end{cases}

    where n is the order.

    The points are arranged in layers from bottom to top, with each layer having
    a square pattern that reduces in size towards the apex.
    
    Parameters
    ----------
    pyr: torch.Tensor [n_batch, 5, n_dim] or [5, n_dim]
    order: int 
    include_boundary: bool
        whether to include the boundary

    Returns
    -------
    torch.Tensor [n_batch, (order+1)*(order+2)*(2*order2)/6, n_dim] or [(order+1)*(order+2)*(2*order2)/6, n_dim]
    """
    device = pyr.device
    x = hex_linspace(pyr, order, include_boundary) # [n_batch, (order+1)**3 or (order-1)**3, n_dim]
    n_points = int(np.round(x.shape[-2]**(1/3)))
    k,j,i = torch.meshgrid(torch.arange(n_points), 
                           torch.arange(n_points), 
                           torch.arange(n_points), 
                           indexing='ij')
    i,j,k = i.flatten().to(device), j.flatten().to(device), k.flatten().to(device)

    if include_boundary:
        mask = (i + k < n_points) & (j + k < n_points)
    else:
        mask = (i + k < n_points - 1) & (j + k < n_points - 1)

    return x[..., mask, :]


def pri_linspace(pri:torch.Tensor, 
                 order:int = 1, 
                 include_boundary:bool=True
                 )->torch.Tensor:
    r"""
    Generate evenly spaced points inside a triangular prism.

    The points are generated by first creating a regular grid in a hexahedron, then filtering points
    to keep only those inside the prism. For order k, points are placed at:

    * Vertices (order 1)
    * Edge points at :math:`\{1/k, 2/k, ..., (k-1)/k\}` along each edge (order >= 2)
    * Face points in a triangular pattern on triangular faces and rectangular pattern on quad faces (order >= 3)
    * Interior points (order >= 4)

    When include_boundary=False, only interior points are returned.

    The number of points generated is:

    .. math::

        N_{points} = \begin{cases}
            \frac{(n+1)^2(n+2)}{2} & \text{if include\_boundary=True} \\
            \frac{(n-1)^2(n)}{2} & \text{if include\_boundary=False}
        \end{cases}

    where n is the order.

    The points are arranged in triangular layers from bottom to top, with each layer having
    the same triangular pattern.

    Examples
    --------
    >>> import torch
    >>> pri = torch.tensor([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,0,1], [0,1,1]])
    >>> points = pri_linspace(pri, order=2)
    >>> print(points.shape)
    torch.Size([18, 3])  # 6 vertices + 9 edge midpoints + 3 face centers

    >>> points = pri_linspace(pri, order=3, include_boundary=False)
    >>> print(points.shape) 
    torch.Size([4, 3])  # Only interior points
    
    Parameters
    ----------
    pri: torch.Tensor [n_batch, 6, n_dim] or [6, n_dim]
    order: int 
    include_boundary: bool
        whether to include the boundary
    Returns
    -------
    torch.Tensor [n_batch, (order+1)^2(order+2)/2, n_dim] or [(order+1)^2(order+2)/2, n_dim]
    """
    device = pri.device
    x = hex_linspace(pri, order, include_boundary) # [n_batch, (order+1)**3, n_dim]
    n_points = int(np.round(x.shape[-2]**(1/3)))
    k,j,i = torch.meshgrid(torch.arange(n_points), 
                           torch.arange(n_points), 
                           torch.arange(n_points), 
                           indexing='ij')
    i,j,k = i.flatten().to(device), j.flatten().to(device), k.flatten().to(device)

    if include_boundary:
        mask = i + j < n_points
    else:
        mask = i + j < n_points - 1

    return x[..., mask, :]
    



########################
# High Order Basis 
########################
# the code depends on the order illustrated in element.py

def lin_basis(points:torch.Tensor, 
                vertex:torch.Tensor,
                edge:torch.Tensor, 
                order:int=1)->torch.Tensor:
    r"""
    Examples
    --------
    .. code-block:: text

        order = 3

            0 -- 2 -- 3 -- 1

    Parameters
    ----------
        points: torch.Tensor 
            2D Tensor of shape [n_vertex, n_dim]
        vertex: torch.Tensor
            2D Tensor of shape [n_vertex, 1]
        edge:  torch.Tensor
            2D Tensor of shape [n_edge, 2]
            the coordinate of the edge
        order: int
            the order of the basis, should be at least 1
        default is 1
    Returns
    -------
        basis: torch.Tensor
            2D Tensor of shape [n_basis, n_dim]    
    """
    n_dim       = points.shape[-1]
    edge_coords = points[edge] # [n_batch, 2, n_dim] 
    coords_0d   = points[vertex] # [n_batch, 1, n_dim]
    coords_0d   = coords_0d.reshape(-1, n_dim)
    if order == 1:
        return coords_0d
    elif order > 1:
        coords_1d = lin_linspace(edge_coords, order, include_boundary=False) # [n_batch, order-1, n_dim] or [order-1, n_dim]
        coords_1d = coords_1d.reshape(-1, n_dim)
        return torch.cat([coords_0d, coords_1d], dim=0)
    else:
        raise ValueError(f"order should be at least 1, but got {order}")
    
def tri_basis(points:torch.Tensor,
                 vertex:torch.Tensor,
                 edge:torch.Tensor,
                 face:torch.Tensor,
                 order:int=1)->torch.Tensor:
    r"""
    Examples
    --------
    .. code-block:: text

        triangle order = 3

            2
            |\
            | \
            4--6
            |  |\
            |  | \
            5--9--3
            |  |  |\
            |  |  | \
            0--7--8--1

    Parameters
    ----------
        points: torch.Tensor
            2D Tensor of shape [n_vertex, n_dim]
        vertex:torch.Tensor
            2D Tensor of shape [n_vertex, 1]
        edge: torch.Tensor
            2D Tensor of shape [n_edge, 2]
        face: torch.Tensor
            2D Tensor of shape [1, 3]
        order: int
            the order of the basis, should be at least 1
            default is 1
    Returns
    -------
        basis: torch.Tensor
            2D Tensor     
    """
    assert order >= 1, f"order should be at least 1, but got {order}"
    n_dim     = points.shape[-1]
    coords_0d = points[vertex] # [n_batch, 1, n_dim]
    coords_0d = coords_0d.reshape(-1, n_dim)
    if order == 1:
        return coords_0d
    
    coords_1d = lin_linspace(points[edge], order, include_boundary=False)
    coords_1d = coords_1d.reshape(-1, n_dim)
    
    if order == 2:
        return torch.cat([coords_0d, coords_1d], 0)
    
    coords_2d = tri_linspace(points[face], order, include_boundary=False)
    coords_2d = coords_2d.reshape(-1, n_dim)
    return torch.cat([coords_0d, coords_1d, coords_2d], 0)

def quad_basis(points:torch.Tensor,
                 vertex:torch.Tensor,
                 edge:torch.Tensor,
                 face:torch.Tensor,
                 order:int=1)->torch.Tensor:
    r"""
    Examples
    --------

    .. code-block:: text

        Quadrilateral element with order=3

        2 - 10 - 11 - 3
        |   |    |    |
        7 - 14 - 15 - 9       
        |   |    |    |
        6 - 12 - 13 - 8
        |   |    |    |
        0 - 4  - 5  - 1

    Parameters
    ----------
        points: torch.Tensor
            2D Tensor of shape [n_vertex, n_dim]
        vertex:torch.Tensor
            2D Tensor of shape [n_vertex, 1]
        edge: torch.Tensor
            2D Tensor of shape [n_edge, 2]
        face: torch.Tensor
            2D Tensor of shape [1, 4]
        order: int
            the order of the basis, should be at least 1
            default is 1
    Returns
    -------
        basis: torch.Tensor
            2D Tensor     
    """
    assert order >= 1, f"order should be at least 1, but got {order}"
    n_dim     = points.shape[-1]
    coords_0d = points[vertex] # [n_batch, 1, n_dim]
    coords_0d = coords_0d.reshape(-1, n_dim)
    if order == 1:
        return coords_0d
    
    coords_1d = lin_linspace(points[edge], order, include_boundary=False)
    coords_1d = coords_1d.reshape(-1, n_dim)
    coords_2d = quad_linspace(points[face], order, include_boundary=False)
    coords_2d = coords_2d.reshape(-1, n_dim)

    return torch.cat([coords_0d, coords_1d, coords_2d], 0)

def tet_basis(points:torch.Tensor,
              vertex:torch.Tensor,
              edge:torch.Tensor,  
              face:torch.Tensor,
              cell:torch.Tensor,
              order:int=1)->torch.Tensor:
    r"""
    Generate basis points for a tetrahedral element.

    For order k, points are placed at:

    * Vertices (order 1)
    * Edge points at :math:`\{1/k, 2/k, ..., (k-1)/k\}` along each edge (order >= 2) 
    * Face points in a triangular pattern on each face (order >= 3)
    * Interior points (order >= 4)

    When order=3, the basis points are arranged as (showing one face):

    .. code-block:: text

                3
               /|\
              / | \
             /  |  \
            9   8   7
           /    |    \
          /     |     \
         /      |      \
        /   14  13  12  \
       /        |        \
      /         |         \
     0-----4----5----6-----1
              Face 0

        Vertices: 0,1,2,3
        Edge points: 4,5,6,7,8,9,10,11 
        Face points: 12,13,14,15,16,17,18,19
        Interior point: 20 (not shown)

    The number of basis points is:

    .. math::

        N_{points} = \begin{cases}
            4 & \text{if order = 1} \\
            10 & \text{if order = 2} \\
            20 & \text{if order = 3} \\
            35 & \text{if order = 4}
        \end{cases}

    Examples
    --------
    >>> points = torch.tensor([
            [0,0,0], # vertex 0
            [1,0,0], # vertex 1  
            [0,1,0], # vertex 2
            [0,0,1]  # vertex 3
        ])
    >>> vertex = torch.tensor([[0],[1],[2],[3]])
    >>> edge = torch.tensor([[0,1],[1,2],[2,0],[0,3],[1,3],[2,3]])
    >>> face = torch.tensor([[0,1,2],[0,1,3],[1,2,3],[2,0,3]])
    >>> cell = torch.tensor([[0,1,2,3]])
    >>> basis = tet_basis(points, vertex, edge, face, cell, order=2)
    >>> print(basis.shape)
    torch.Size([10, 3])  # 4 vertices + 6 edge midpoints
    
    Parameters
    ----------
        points: torch.Tensor
            2D Tensor of shape [n_vertex, n_dim]
        vertex:torch.Tensor
            2D Tensor of shape [n_vertex, 1]
        edge:torch.Tensor
            2D Tensor of shape [n_edge, 2]
        face: torch.Tensor
            2D Tensor of shape [n_face, 3]
        cell: torch.Tensor
            2D Tensor of shape [1, 4]
        order: int
            the order of the basis, should be at least 1,
            default is 1
    Returns
    -------
        basis: torch.Tensor
            2D Tensor     
    """
    assert order  >= 1, f"order should be at least 1, but got {order}"
    assert len(face) == 4, f"tetrahedron should have 4 faces, but got {len(face)}"
    n_dim = points.shape[-1]
    coords_0d = points[vertex] # [n_batch, 1, n_dim]
    coords_0d = coords_0d.reshape(-1, n_dim)
    if order == 1:
        return coords_0d
    
    coords_1d = lin_linspace(points[edge], order, include_boundary=False) # [n_batch, order-1, n_dim]
    coords_1d = coords_1d.reshape(-1, n_dim)

    if order == 2:
        return torch.cat([coords_0d, coords_1d], 0)
    
    coords_2d = tri_linspace(points[face], order, include_boundary=False) # [n_batch, n_basis_per_face, 3]
    coords_2d = coords_2d.reshape(-1, n_dim)

    if order == 3:
        return torch.cat([coords_0d, coords_1d, coords_2d], 0)
    
    coords_3d = tet_linspace(points[cell], order, include_boundary=False) # [n_batch, n_basis_per_cell, 3]
    coords_3d = coords_3d.reshape(-1, n_dim)

    return torch.cat([coords_0d, coords_1d, coords_2d, coords_3d], 0)

def hex_basis(points:torch.Tensor,
                vertex:torch.Tensor,
                edge:torch.Tensor,
                face:torch.Tensor,
                cell:torch.Tensor,
                order:int=1)->torch.Tensor:
    r"""
    Parameters
    ----------
        points: torch.Tensor
            2D Tensor of shape [n_vertex, n_dim]
        vertex: torch.Tensor
            2D Tensor of shape [n_vertex, 1]
        edge: torch.Tensor
            2D Tensor of shape [n_edge, 2]
        face: torch.Tensor
            2D Tensor of shape [n_face, 4]
        cell: torch.Tensor
            2D Tensor of shape [1, 6]
        order: int
            the order of the basis, should be at least 1,
            default is 1
    Returns
    -------
        basis: torch.Tensor
            2D Tensor     
    """
    assert order  >= 1, f"order should be at least 1, but got {order}"
    assert len(face) == 6, f"hex should have 6 faces, but got {len(face)}"
    n_dim = points.shape[-1]
    coords_0d = points[vertex] # [n_batch, 1, n_dim]
    coords_0d = coords_0d.reshape(-1, n_dim)
    if order == 1:
        return coords_0d
    
    coords_1d = lin_linspace(points[edge], order, include_boundary=False) # [n_batch, order-1, n_dim]
    coords_1d = coords_1d.reshape(-1, n_dim)

    coords_2d = quad_linspace(points[face], order, include_boundary=False) # [n_batch, n_basis_per_face, 3]
    coords_2d = coords_2d.reshape(-1, n_dim)
    
    coords_3d = hex_linspace(points[cell], order, include_boundary=False) # [n_batch, n_basis_per_cell, 3]
    coords_3d = coords_3d.reshape(-1, n_dim)

    return torch.cat([coords_0d, coords_1d, coords_2d, coords_3d], 0)

def pyr_basis(points:torch.Tensor,
                vertex:torch.Tensor,
                edge:torch.Tensor,
                face:Tuple[Tuple[int,...],...],
                cell:torch.Tensor,
                order:int=1)->torch.Tensor:
    r"""
    Parameters
    ----------
        points: torch.Tensor
            2D Tensor of shape [n_vertex, n_dim]
        vertex:torch.Tensor
            2D Tensor of shape [n_vertex, 1]
        edge:torch.Tensor
            2D Tensor of shape [n_edge, 2]
        face: Tuple[Tuple[int,...],...]
            Tuple of Tuple of int, each tuple is a face
        cell: torch.Tensor
            2D Tensor of shape [1, 5]
        order: int
            the order of the basis, should be at least 1,
            default is 1
    Returns
    -------
        basis: torch.Tensor
            2D Tensor     
    """
    assert order  >= 1, f"order should be at least 1, but got {order}"
    assert len(face) == 5, f"pyramid should have 5 faces, but got {len(face)}"
    n_dim = points.shape[-1]
    assert len(face[0]) == 4, f"the first face of the pyramid should have 4 vertices, but got {len(face[0])}"
    assert all([len(f)==3 for f in face[1:]]), f"the other faces of the pyramid should have 3 vertices, but got {[len(f) for f in face[1:]]}"
    quad_face = torch.tensor(face[0])
    tri_face  = torch.tensor(face[1:])
    coords_0d = points[vertex] # [n_batch, 1, n_dim]
    coords_0d = coords_0d.reshape(-1, n_dim)
    if order == 1:
        return coords_0d
    
    coords_1d = lin_linspace(points[edge], order, include_boundary=False) # [n_batch, order-1, n_dim]
    coords_1d = coords_1d.reshape(-1, n_dim)

    if order == 2:
        coords_2d = quad_linspace(points[quad_face], order, include_boundary=False) # [n_basis_per_face, 3]
        coords_2d = coords_2d.reshape(-1, n_dim)
        return torch.cat([coords_0d, coords_1d, coords_2d], 0)
    
    coords_2d = torch.cat([
        quad_linspace(points[quad_face], order, include_boundary=False).reshape(-1, n_dim), 
        tri_linspace(points[tri_face], order, include_boundary=False).reshape(-1, n_dim)
    ], 0)
    
    coords_3d = pyr_linspace(points[cell], order, include_boundary=False) # [n_batch, n_basis_per_cell, 3]
    coords_3d = coords_3d.reshape(-1, n_dim)

    return torch.cat([coords_0d, coords_1d, coords_2d, coords_3d], 0)

def pri_basis(points:torch.Tensor,
                vertex:torch.Tensor,
                edge:torch.Tensor,
                face:Tuple[Tuple[int,...],...],
                cell:torch.Tensor,
                order:int=1)->torch.Tensor:
    r"""
    Parameters
    ----------
        points: torch.Tensor
            2D Tensor of shape [n_vertex, n_dim]
        vertex:torch.Tensor
            2D Tensor of shape [n_vertex, 1]
        edge:torch.Tensor
            2D Tensor of shape [n_edge, 2]
        face: Tuple[Tuple[int,...],...]
            Tuple of Tuple of int, each tuple is a face
        cell: torch.Tensor
            2D Tensor of shape [1, 6]
        order: int
            the order of the basis, should be at least 1,
            default is 1
    Returns
    -------
        basis: torch.Tensor
            2D Tensor     
    """
    assert order  >= 1, f"order should be at least 1, but got {order}"
    n_dim = points.shape[-1]
    assert len(face) == 5, f"prism should have 5 faces, but got {len(face)}"
    assert len(face[0]) == 3 and len(face[-1]) == 3, \
            f"the first and last face of the prism should have 4 vertices, but got {len(face[0])} and {len(face[-1])}"
    assert all([len(f)== 4 for f in face[1:-1]]), \
            f"the other faces of the prism should have 3 vertices, but got {[len(f) for f in face[1:-1]]}"
    quad_face = torch.tensor(face[1:4])
    tri_face  = torch.tensor([face[0],face[-1]])
    coords_0d = points[vertex] # [n_batch, 1, n_dim]
    coords_0d = coords_0d.reshape(-1, n_dim)
    if order == 1:
        return coords_0d
    
    coords_1d = lin_linspace(points[edge], order, include_boundary=False) # [n_batch, order-1, n_dim]
    coords_1d = coords_1d.reshape(-1, n_dim)

    if order == 2:
        coords_2d = quad_linspace(points[quad_face], order, include_boundary=False) # [n_basis_per_face, 3]
        coords_2d = coords_2d.reshape(-1, n_dim)
        return torch.cat([coords_0d, coords_1d, coords_2d], 0)
    
    coords_2d = torch.cat([
        tri_linspace(points[tri_face[0]], order, include_boundary=False).reshape(-1, n_dim),
        quad_linspace(points[quad_face], order, include_boundary=False).reshape(-1, n_dim), 
        tri_linspace(points[tri_face[-1]], order, include_boundary=False).reshape(-1, n_dim)
    ], 0)
    
    coords_3d = pri_linspace(points[cell], order, include_boundary=False) # [n_batch, n_basis_per_cell, 3]
    coords_3d = coords_3d.reshape(-1, n_dim)
    
    return torch.cat([coords_0d, coords_1d, coords_2d, coords_3d], 0)

##################################
# High Order Facet Basis Index
##################################

def accumulate_range(start:int, 
                     batch:Optional[int]=None, 
                     num:Optional[int]=None, 
                     nums:Optional[Sequence[int]]=None
                     )->Union[torch.Tensor, Tuple[Tuple[int,...],...]]:
        """
        Examples
        --------
        >>> accumulate_range(0, 3, 2)
        tensor([[0, 1],
                [2, 3],
                [4, 5]])

        >>> accumlate_range(0, (2, 3))
        ((0, 1), (2, 3, 4))

        Parameters
        ----------
        start:int 
            the start index
        batch:int 
            the number of batch
        num:int
            the number of elements

        Returns
        -------
        index: torch.Tensor
            2D Tensor of shape [batch, num]
        """
        if nums is None:
            assert batch is not None and num is not None, "batch and num should be provided"
            return torch.arange(start, start+num*batch).reshape(batch, num)
        else:
            result = [tuple() for _ in range(len(nums))]
            for i,num in enumerate(nums):
                result[i] = tuple(range(start, start+num))
                start     += num
            return tuple(result)

def index_reverse_mapping(index:Union[torch.Tensor,Sequence[torch.Tensor]], 
                         table:torch.Tensor
                         )->Union[torch.Tensor, Tuple[torch.Tensor,...]]:
    """The last dimension of the index is a unique combination, 
    which could be found in the table. The combination will be substituted 
    by the index (row number) of the table.
    
    Examples
    --------
    >>> # Single tensor index
    >>> index = torch.tensor([[0,1], [1,2], [2,0]])
    >>> table = torch.tensor([[0,1], [1,2], [2,0], [1,0], [2,1], [0,2]])
    >>> index_reverse_mapping(index, table)
    tensor([0, 1, 2])

    >>> # Multiple tensor indices
    >>> index1 = torch.tensor([[0,1], [1,2]])
    >>> index2 = torch.tensor([[2,0], [1,0]])
    >>> table = torch.tensor([[0,1], [1,2], [2,0], [1,0]])
    >>> idx1, idx2 = index_reverse_mapping([index1, index2], table)
    >>> idx1
    tensor([0, 1])
    >>> idx2
    tensor([2, 3])


    Parameters
    ----------
    index: Union[torch.Tensor,Sequence[torch.Tensor]]
        [..., n_dim] or sequence of [..., n_dim]
    table: torch.Tensor
        [n_combination, n_dim]

    Returns
    -------
    index: torch.Tensor
        [...] or sequence of [...]
    """
    # TODO : increase efficiency
    if isinstance(index, torch.Tensor):
        assert index.shape[-1] == table.shape[-1]
        *index_shape, dim = index.shape
        table_dict = {tuple(t.tolist()):i for i, t in enumerate(table)}
        index      = index.reshape(-1, dim)
        index      = torch.tensor([table_dict[tuple(t.tolist())] for t in index])
        index      = index.reshape(index_shape)
    else:
        table_dict = {tuple(t.tolist()):i for i, t in enumerate(table)}
        new_index  = [torch.Tensor() for _ in range(len(index))]
        for i,idx in enumerate(index):
            *idx_shape, dim = idx.shape
            assert dim == table.shape[-1]
            idx = idx.reshape(-1, dim)
            idx = torch.tensor([table_dict[tuple(t.tolist())] for t in idx])
            idx = idx.reshape(idx_shape)
            new_index[i] = idx
        index = tuple(new_index)
    return index

def facet_basis_index_2d( vertex:torch.Tensor,
                          edge:torch.Tensor, 
                          order:int = 1
                          )->torch.Tensor:
    """Get the basis function indices for facets (edges) of a 2D element.

    For a 2D element like a triangle or quadrilateral, this function returns the indices
    of basis functions along each edge/facet. The indices include both vertex nodes and
    any higher-order nodes along the edges.

    Examples
    --------
    >>> vertex = torch.tensor([[0], [1], [2]])  # Triangle vertices
    >>> edge = torch.tensor([[1, 2], [0, 2], [0, 1]])  # Triangle edges
    >>> indices = facet_basis_index_2d(vertex, edge, order=1)
    >>> print(indices)
    tensor([[1, 2],  # First edge connects vertices 1-2
            [0, 2],  # Second edge connects vertices 0-2 
            [0, 1]]) # Third edge connects vertices 0-1

    >>> indices = facet_basis_index_2d(vertex, edge, order=2)
    >>> print(indices)
    tensor([[1, 2, 3],  # First edge: vertices 1-2 plus midpoint node 3
            [0, 2, 4],  # Second edge: vertices 0-2 plus midpoint node 4
            [0, 1, 5]]) # Third edge: vertices 0-1 plus midpoint node 5

    The returned indices can be used to extract basis functions or nodes along element edges,
    which is useful for enforcing continuity between elements or applying boundary conditions.
    
    Parameters
    ----------
    vertex: torch.Tensor    
        2D Tensor of shape [n_vertex, 1]
    edge: torch.Tensor
        2D Tensor of shape [n_edge, 2]
    order: int
        the order of the basis, should be at least 1
        default is 1

    Returns
    -------
    index: torch.Tensor
        2D Tensor of shape [n_edge, order+1]
    """
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    index_0d = edge # [n_edge, 2]
    index_1d = accumulate_range(
                start = n_vertex, 
                batch = n_edge, 
                num   = order-1) # [n_edge, order-1]

    return  torch.cat([index_0d,  index_1d], -1) # type: ignore

def tet_facet_basis_index(vertex:torch.Tensor,
                          edge:torch.Tensor,  
                          face:torch.Tensor,
                          order:int = 1)->torch.Tensor:
    r"""
    Parameters
    ----------
        vertex: torch.Tensor    
            2D Tensor of shape [n_vertex, 1]
        edge: torch.Tensor
            2D Tensor of shape [n_edge, 2]
        face: torch.Tensor
            2D Tensor of shape [n_face, 3]
        order: int
            the order of the basis, should be at least 1
            default is 1

    Returns
    -------
        index: torch.Tensor
            2D Tensor of shape [n_face, (order+1)(order+2)//2]
    """
    facet_edge= torch.tensor([[1, 2], [0, 2], [0, 1]])  # [3x2]
    n_vertex  = vertex.shape[0]
    n_edge    = edge.shape[0]
    n_face    = face.shape[0]

    index_0d  = face  # [n_face, 3]

    if order == 1:
        return index_0d 

    basis_per_edge = order - 1
    edge_index    = accumulate_range(
                        start = n_vertex, 
                        batch = n_edge, 
                        num   = basis_per_edge) # [n_edge, order-1] 

    edge2face     = index_reverse_mapping(face[:, facet_edge], edge) # [n_face, n_edge_per_face] 
    index_1d      = edge_index[edge2face, :] # type: ignore # [n_face, 3, order-1]
    index_1d      = index_1d.reshape(n_face, -1)

    if order == 2:
        return torch.cat([index_0d, index_1d], -1)
    
    basis_per_face = (order-2)*(order-1)//2
    face_index    = accumulate_range(
                        start = n_vertex + basis_per_edge * n_edge, 
                        batch = n_face,
                        num   = basis_per_face ) # [n_face, (order-1)(order-2)//2]
    
    index_2d      = face_index # [n_face, (order-1)(order-2)//2]

    return torch.cat([index_0d, index_1d, index_2d], -1) # type: ignore

def hex_facet_basis_index(vertex:torch.Tensor,
                          edge:torch.Tensor,  
                          face:torch.Tensor,
                          order:int = 1)->torch.Tensor:
    r"""
    Parameters
    ----------
        vertex: torch.Tensor    
            2D Tensor of shape [n_vertex, 1]
        edge: torch.Tensor
            2D Tensor of shape [n_edge, 2]
        face: torch.Tensor
            2D Tensor of shape [n_face, 4]
        order: int
            the order of the basis, should be at least 1
            default is 1

    Returns
    -------
        index: torch.Tensor
            2D Tensor of shape [n_face, (order+1)(order+2)//2]
    """
    facet_edge = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]]) # [4x2]
    n_vertex  = vertex.shape[0]
    n_edge    = edge.shape[0]
    n_face    = face.shape[0]

    index_0d  = face  # [n_face, 3]

    if order == 1:
        return index_0d 

    basis_per_edge = order - 1
    edge_index    = accumulate_range(
                        start = n_vertex, 
                        batch = n_edge, 
                        num   = basis_per_edge) # [n_edge, order-1]

    edge2face     = index_reverse_mapping(face[:, facet_edge], edge) # [n_face, n_edge_per_face]
    index_1d      = edge_index[edge2face, :] # [n_face, 3, order-1] type: ignore
    index_1d      = index_1d.reshape(n_face, -1)

    basis_per_face = (order-1)*(order-1)
    face_index    = accumulate_range(
                        start = n_vertex + basis_per_edge * n_edge, 
                        batch = n_face,
                        num   = basis_per_face ) # [n_face, (order-1)*(order-1)]
    
    index_2d      = face_index # [n_face,  (order-1)*(order-1)]

    return torch.cat([index_0d, index_1d, index_2d], -1) # type: ignore
    
def mix_facet_basis_index(vertex:torch.Tensor,
                          edge:torch.Tensor,  
                          face:Tuple[Tuple[int,...],...],
                          order:int = 1)->Tensorx2:
    r"""
    Parameters
    ----------
        vertex: torch.Tensor    
            2D Tensor of shape [n_vertex, 1]
        edge: torch.Tensor
            2D Tensor of shape [n_edge, 2]
        face: Tuple[Tuple[int,...],...]
            Tuple of Tuple of int, each tuple is a face
        order: int
            the order of the basis, should be at least 1
            default is 1

    Returns
    -------
        tri_index: torch.Tensor
            2D Tensor of shape [n_tri_facet, n_basis_per_tri_facet]
        quad_index: torch.Tensor
            2D Tensor of shape [n_quad_facet, n_basis_per_quad_facet]
    """
    tri_edge  = torch.tensor([[1, 2], [0, 2], [0, 1]])  # [3x2]
    quad_edge = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]]) # [4x2]
    n_vertex  = vertex.shape[0]
    n_edge    = edge.shape[0]
    n_face    = len(face)

    tri_mask  = torch.tensor([len(f) == 3 for f in face]) # [n_face]
    quad_mask = ~tri_mask # [n_face]

    index_0d  = face  # [n_face, 3]

    if order == 1:
        tri_facet = torch.tensor([index_0d[i] for i in range(len(face)) if tri_mask[i]])
        quad_facet= torch.tensor([index_0d[i] for i in range(len(face)) if quad_mask[i]])   
        return tri_facet, quad_facet

    basis_per_edge = order - 1
    edge_index     = accumulate_range(
                        start = n_vertex, 
                        batch = n_edge, 
                        num   = basis_per_edge) # [n_edge, order-1]

    edge2face     = index_reverse_mapping(
        [torch.tensor(f)[tri_edge] if len(f) == 3 else torch.tensor(f)[quad_edge] for f in face],
        edge) # Tuple[torch.Tensor[n_edge_per_face],...]
    index_1d      = tuple([tuple(edge_index[e2f].flatten().tolist()) for e2f in edge2face]) # type: ignore
    
    basis_per_tri_face  = (order-2)*(order-1)//2
    basis_per_quad_face = (order-1)*(order-1)
    face_index    = accumulate_range(
                        start = n_vertex + basis_per_edge * n_edge, 
                        nums  = [basis_per_tri_face if len(f) == 3 else basis_per_quad_face for f in face]  
                      ) # Tuple[Tuple[int, ...],...]
    
    index_2d      = face_index 

    index = tuple([index_0d[i]+index_1d[i]+index_2d[i] for i in range(n_face)])

    tri_facet = torch.tensor([index[i] for i in range(len(face)) if tri_mask[i]])
    quad_facet= torch.tensor([index[i] for i in range(len(face)) if quad_mask[i]])
    return tri_facet, quad_facet

def edge_index(vertex:torch.Tensor, 
               edge:torch.Tensor,
               order:int,
               )->torch.Tensor:
    """
    Get the edge basis indices for a given order.

    For a given order n, returns the connectivity array for the edges of the element. Each row represents 
    an edge segment between two consecutive basis nodes along an edge.

    Examples
    --------
    >>> vertex = torch.tensor([[0], [1], [2]])
    >>> edge = torch.tensor([[0, 1], [1, 2]])
    >>> edges = edge_index(vertex, edge, order=1)
    >>> edges
    tensor([[0, 1],  # First edge segment
            [1, 2]]) # Second edge segment

    >>> # Higher order elements have more segments per edge
    >>> edges = edge_index(vertex, edge, order=2)
    >>> edges
    tensor([[0, 2],  # First edge divided into two segments
            [2, 1],
            [1, 3],  # Second edge divided into two segments 
            [3, 2]])

    Parameters
    ----------
    vertex: torch.Tensor    
        2D Tensor of shape [n_vertex, 1]
    edge: torch.Tensor
        2D Tensor of shape [n_edge, 2]
    order: int
        The order of the basis

    Returns
    -------
    edges: torch.Tensor
        2D Tensor of shape [n_ordered_edge, 2] containing edge segment connectivity
    
    """
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    middle = accumulate_range(
                start = n_vertex, 
                batch = n_edge, 
                num   = order-1) # [n_edge, order-1]

    line =  torch.cat([edge[:,0:1],  middle, edge[:,1:2]], -1) # [n_edge, order+1]

    edges = torch.stack([line[:,:-1], line[:,1:]], -1).reshape(-1, 2)

    return edges


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.patches as patches
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    import numpy as np
    from scipy.spatial import ConvexHull


    def plot_1d_points(x, p, ax):
        if x.dim() == 3: # [n_batch, 2, n_dim]
            # p [n_batch, order+1/order-1, n_dim]
            cmap = cm.get_cmap('Spectral')
            norm = lambda a: a / p.shape[0]
            for i,_p in enumerate(p): # _p [order+1/order-1, n_dim]
                color = cmap(norm(i))
                # breakpoint()
                ax.plot(x[i, :, 0], x[i, :, 1], color=color, alpha=0.5)
                ax.scatter(_p[:, 0], _p[:, 1], color=color)
                for j in range(_p.shape[0]):
                    ax.text(_p[j, 0], _p[j, 1], f"{j}", fontsize=8)
        else:
            # p [order+1/order-1, n_dim]
            ax.plot(x[:, 0], x[:, 1], color='blue', alpha=0.5)
            ax.scatter(p[:, 0], p[:, 1], color='orange')     
            for i in range(p.shape[0]):
                ax.text(p[i, 0], p[i, 1], f"{i}", fontsize=8)

    def plot_2d_points(x, p, ax):
        """
        Parameters
        ----------
            x: torch.Tensor [n_batch, 3 or 4, n_dim] or [3 or 4, n_dim]
            p: torch.Tensor [n_batch, n_points, n_dim] or [n_points, n_dim]
        """
        if x.dim() == 3: # [n_batch, 3 or 4, n_dim]
            # p [n_batch, (order+1)*(order+2)/2, n_dim]
            # cmap = cm.get_cmap('Spectral')
            cmap   = cm.get_cmap("plasma")
            norm = lambda a: a / p.shape[0]
            for i,_p in enumerate(p): # _p [(order+1)*(order+2)/2, n_dim]
                color = cmap(norm(i))
                hull  = ConvexHull(x[i].detach().numpy())
                polygon = patches.Polygon(x[i][hull.vertices], closed=True, fill=False, edgecolor=color, alpha=0.5)
                ax.add_patch(polygon)
                ax.scatter(_p[:, 0], _p[:, 1], color=color)
                for j in range(_p.shape[0]):
                    ax.text(_p[j, 0], _p[j, 1], f"{j}", fontsize=8)
        else: # [3 or 4, n_dim]
            # p [(order+1)*(order+2)/2, n_dim]
            hull    = ConvexHull(x.detach().numpy())
            polygon = patches.Polygon(x[hull.vertices], closed=True, fill=False, edgecolor='blue', alpha=0.5)
            ax.add_patch(polygon)
            
            ax.scatter(p[:, 0], p[:, 1], color='orange')
            for i in range(p.shape[0]):
                ax.text(p[i, 0], p[i, 1], f"{i}", fontsize=8)

    def plot_3d_points(x, p, ax):
        """
        Parameters
        ----------
            x: torch.Tensor [n_batch, 4 or 5 or 6, n_dim] or [4 or 5 or 6, n_dim]
            p: torch.Tensor [n_batch, n_points, n_dim] or [n_points, n_dim]
        """
        x = x.detach().numpy()
        p = p.detach().numpy()

        if len(x.shape) == 3: # [n_batch, 4 or 5 or 6, n_dim]
            cmap = cm.get_cmap('plasma')
            norm = lambda a: a / p.shape[0]
            for i,_p in enumerate(p): # _p [n_points, n_dim]
                color = cmap(norm(i))
                hull  = ConvexHull(x[i])
                for s in hull.simplices:
                    s = np.append(s, s[0])
                    ax.plot(x[i][s, 0], x[i][s, 1], x[i][s, 2], color=color)
                ax.scatter(_p[:, 0], _p[:, 1], _p[:, 2], color=color)
                for j in range(_p.shape[0]):
                    ax.text(_p[j, 0], _p[j, 1], _p[j, 2], f"{j}", fontsize=8)

        else: # [4 or 5 or 6, n_dim]
            hull = ConvexHull(x)
            simplices = [x[s] for s in hull.simplices]
            poly = Poly3DCollection(simplices, linewidths=1, edgecolors='k', alpha=.25)
            ax.add_collection(poly)
            # for s in hull.simplices:
            #     s = np.append(s, s[0])
            #     ax.plot(x[s, 0], x[s, 1], x[s, 2], 'b-')

            ax.scatter(p[:,0], p[:, 1], p[:, 2], color="orange")
            for i in range(p.shape[0]):
                ax.text(p[i, 0], p[i, 1], p[i, 2], f"{i}", fontsize=8)

    def plot_2d_facet_points(x, p, t, ax):
        """
        Parameters
        ----------
            x: torch.Tensor [3 or 4, n_dim]
            p: torch.Tensor [n_points, n_dim]
            t: torch.Tensor [n_points]
        """
        # p [(order+1)*(order+2)/2, n_dim]
        hull    = ConvexHull(x.detach().numpy())
        polygon = patches.Polygon(x[hull.vertices], closed=True, fill=False, edgecolor='blue', alpha=0.5)
        ax.add_patch(polygon)
        
        ax.scatter(p[:2, 0], p[:2, 1], color='orange')
        if len(p) > 2:
            ax.scatter(p[2:, 0], p[2:, 1], color='blue')
        for i in range(p.shape[0]):
            ax.text(p[i, 0], p[i, 1], f"{t[i]}", fontsize=8)

    def plot_3d_facet_points(x, p, t, is_tri_facet, order, ax):
        """
        Parameters
        ----------
            x: torch.Tensor [4 or 5 or 6, n_dim]
            p: torch.Tensor [n_points, n_dim]
            t: torch.Tensor [n_points]
            is_tri_facet: bool
            order: int
        """
        hull = ConvexHull(x)
        simplices = [x[s] for s in hull.simplices]
        poly = Poly3DCollection(simplices, linewidths=1, edgecolors='k', alpha=.25)
        ax.add_collection(poly)

        n_vertex = 3 if is_tri_facet else 4
        n_edge   = 3 if is_tri_facet else 4
        n_basis_per_edge = order - 1 

        basis_0d = p[:n_vertex]
        basis_1d = p[n_vertex:n_vertex+n_edge*n_basis_per_edge]
        basis_2d = p[n_vertex+n_edge*n_basis_per_edge:]
        ax.scatter(basis_0d[:, 0], basis_0d[:, 1], basis_0d[:, 2], color='orange')
        ax.scatter(basis_1d[:, 0], basis_1d[:, 1], basis_1d[:, 2], color='blue')
        ax.scatter(basis_2d[:, 0], basis_2d[:, 1], basis_2d[:, 2], color='green')

        for i in range(p.shape[0]):
            ax.text(p[i, 0], p[i, 1], p[i, 2], f"{t[i]}", fontsize=8)

    # test linspace
    
    def plot_test_lin_linspace():
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 12))
        for i, order in enumerate([2, 5]):
            for j, use_batch in enumerate([False, True]):
                x = torch.rand(3, 2, 2) if use_batch else torch.rand(2, 2)
                p1 = lin_linspace(x, order, True)
                p2 = lin_linspace(x, order, False)
                plot_1d_points(x, p1,  ax[j, i*2 + 0])
                plot_1d_points(x, p2, ax[j, i*2 + 1])
                ax[0, i*2 + 0].set_title(f"order={order}\n boundary=True")
                ax[0, i*2 + 1].set_title(f"order={order}\n boundary=False")
        plt.show()

    def plot_test_tri_linspace():
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 12))
        triangle  = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.float32)
        triangles = torch.tensor([[[0, 0], [1, 0], [0, 1]], [[1, 1], [1, 2], [1.5, 1]]], dtype=torch.float32)
        for i, order in enumerate([3, 5]):
            for j, use_batch in enumerate([False, True]):
                for k, include_boundary in enumerate([True, False]):
                    x = triangles if use_batch else triangle
                    p = tri_linspace(x, order, include_boundary)
                    plot_2d_points(x, p, ax[j, i*2 + k])
                    ax[0, i*2 + k].set_title(f"order={order}\n boundary={include_boundary}")
        plt.show()
    
    def plot_test_quad_linspace():
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 12))
        quad    = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        quads   = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]], [[1, 0], [1, 1], [2, 1], [2, 2]]], dtype=torch.float32)
        for i, order in enumerate([2, 5]):
            for j, use_batch in enumerate([False, True]):
                for k, include_boundary in enumerate([True, False]):
                    print(f"order:{order} use_batch:{use_batch}, include_boundary:{include_boundary}")
                    x = quads if use_batch else quad
                    p = quad_linspace(x, order, include_boundary)
                    plot_2d_points(x, p, ax[j, i*2 + k])
                    ax[0, i*2 + k].set_title(f"order={order}\n boundary={include_boundary}")
        plt.show()

    def plot_test_tet_linspace():
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), subplot_kw={'projection': '3d'})
        tet = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])
        for i, order in enumerate([3, 5]):
            for j, include_boundary in enumerate([True, False]):
                p = tet_linspace(tet, order, include_boundary)
                plot_3d_points(tet, p, ax[i][j])
                ax[i][j].set_title(f"order={order}\n boundary={include_boundary}")
        plt.show()

    def plot_test_hex_linspace():
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), subplot_kw={'projection': '3d'})
        hex = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0],[1.0, 1.0, 1.0]]) # 8x3
        for i, order in enumerate([2, 4]):
            for j, include_boundary in enumerate([True, False]):
                p = hex_linspace(hex, order, include_boundary)
                plot_3d_points(hex, p, ax[i][j])
                ax[i][j].set_title(f"order={order}\n boundary={include_boundary}")
        plt.show()

    def plot_test_pyr_linspace():
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), subplot_kw={'projection': '3d'})
        pyr = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) # 5x3
        for i, order in enumerate([3, 5]):
            for j, include_boundary in enumerate([True, False]):
                p = pyr_linspace(pyr, order, include_boundary)
                plot_3d_points(pyr, p, ax[i][j])
                ax[i][j].set_title(f"order={order}\n boundary={include_boundary}")
        plt.show()

    def plot_test_pri_linspace():
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), subplot_kw={'projection': '3d'})
        pri = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0]]) # 5x3
        for i, order in enumerate([3, 5]):
            for j, include_boundary in enumerate([True, False]):
                p = pri_linspace(pri, order, include_boundary)
                plot_3d_points(pri, p, ax[i][j])
                ax[i][j].set_title(f"order={order}\n boundary={include_boundary}")
        plt.show()

    # test basis 

    def plot_test_lin_basis():
        orders = [1, 2, 3, 4]
        fig, ax = plt.subplots(nrows=1, ncols=len(orders), figsize=(4*len(orders), 4))
        points = torch.tensor([[0.0],[1.0]]) # 2x1
        vertex = torch.tensor([[0], [1]]) # 2x1
        edge   = torch.tensor([[0, 1]]) # 1x2
        for i, order in enumerate(orders):
            p = lin_basis(points.repeat(1, 2), vertex, edge, order) # [n_basis, n_dim]
            plot_1d_points(points.repeat(1, 2), p,  ax[i])
            ax[i].set_title(f"order={order}")
        plt.show()

    def plot_test_tri_basis():
        orders = [1, 2, 3, 4]
        fig, ax = plt.subplots(nrows=1, ncols=len(orders), figsize=(4*len(orders), 4))
        points = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0]]) # 3x2
        vertex = torch.tensor([[0], [1], [2]]) # 3x1
        edge   = torch.tensor([[1, 2], [0, 2], [0, 1]]) # 3x2
        face   = torch.tensor([[0, 1, 2]]) # 1x3
        for i, order in enumerate(orders):
            p = tri_basis(points, vertex, edge, face, order) # [n_basis, n_dim]
            plot_2d_points(points, p,  ax[i])
            ax[i].set_title(f"order={order}")
        plt.show()

    def plot_test_quad_basis():
        orders = [1, 2, 3, 4]
        fig, ax = plt.subplots(nrows=1, ncols=len(orders), figsize=(4*len(orders), 4))
        points = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]]) # 4x2
        vertex = torch.tensor([[0], [1], [2], [3]]) # 4x1
        edge   = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]]) # 4x2
        face   = torch.tensor([[0, 1, 2, 3]]) # 1x4
        for i, order in enumerate(orders):
            p = quad_basis(points, vertex, edge, face, order) # [n_basis, n_dim]
            plot_2d_points(points, p,  ax[i])
            ax[i].set_title(f"order={order}")
        plt.show()

    def plot_test_tet_basis():
        orders = [1, 2, 3, 4]
        fig, ax = plt.subplots(nrows=1, ncols=len(orders), figsize=(4*len(orders), 4), subplot_kw={'projection': '3d'})
        points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) # 4x3
        vertex = torch.tensor([[0], [1], [2], [3]]) # 4x1
        edge   = torch.tensor([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]) # 6x2
        face   = torch.tensor(((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))) # 4x3
        cell   = torch.tensor([[0, 1, 2, 3]]) # 1x4
        for i, order in enumerate(orders):
            p = tet_basis(points, vertex, edge, face, cell, order) # [n_basis, n_dim]
            plot_3d_points(points, p,  ax[i])
            ax[i].set_title(f"order={order}")
        plt.show()

    def plot_test_hex_basis():
        orders = [1, 2, 3, 4]
        fig, ax = plt.subplots(nrows=1, ncols=len(orders), figsize=(4*len(orders), 4), subplot_kw={'projection': '3d'})
        points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0],[1.0, 1.0, 1.0]]) # 8x3
        vertex = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7]]) # 8x1
        edge   = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]) # 12x2
        face   = torch.tensor(((0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6), (1, 3, 5, 7), (2, 3, 6, 7), (4, 5, 6, 7))) # 6x4
        cell   = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]) # 1x8
        for i, order in enumerate(orders):
            p = hex_basis(points, vertex, edge, face, cell, order) # [n_basis, n_dim]
            plot_3d_points(points, p,  ax[i])
            ax[i].set_title(f"order={order}")
        plt.show()

    def plot_test_pyr_basis():
        orders = [1, 2, 3, 4]
        fig, ax = plt.subplots(nrows=1, ncols=len(orders), figsize=(4*len(orders), 4), subplot_kw={'projection': '3d'})
        points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) # 5x3
        vertex = torch.tensor([[0], [1], [2], [3], [4]]) # 5x1
        edge   = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]) # 8x2
        face   = ((0, 1, 2, 3), (0, 1, 4), (0, 2, 4), (1, 3, 4), (2, 3, 4)) # 5x4
        cell   = torch.tensor([[0, 1, 2, 3, 4]]) # 1x5
        for i, order in enumerate(orders):
            p = pyr_basis(points, vertex, edge, face, cell, order) # [n_basis, n_dim]
            plot_3d_points(points, p,  ax[i])
            ax[i].set_title(f"order={order}")
        plt.show()

    def plot_test_pri_basis():
        orders = [1, 2, 3, 4]
        fig, ax = plt.subplots(nrows=1, ncols=len(orders), figsize=(4*len(orders), 4), subplot_kw={'projection': '3d'})
        points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0]]) # 5x3
        vertex = torch.tensor([[0], [1], [2], [3], [4], [5]]) # 6x1
        edge   = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]) # 9x2
        face   = ((0, 1, 2), (0, 1, 3, 4), (0, 2, 3, 5), (1, 2, 4, 5),(3,4,5)) # 4x4
        cell   = torch.tensor([[0, 1, 2, 3, 4, 5]]) # 1x5
        for i, order in enumerate(orders):
            p = pri_basis(points, vertex, edge, face, cell, order) # [n_basis, n_dim]
            plot_3d_points(points, p,  ax[i])
            ax[i].set_title(f"order={order}")
        plt.show()

    # test facet basis index
    def plot_test_tri_facet_basis_index():
        order = 3
        
        points = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0]]) # 3x2
        vertex = torch.tensor([[0], [1], [2]]) # 3x1
        edge   = torch.tensor([[1, 2], [0, 2], [0, 1]]) # 3x2
        face   = torch.tensor([[0, 1, 2]]) # 1x3
        
        n_facet = len(edge)
        fig, ax = plt.subplots(nrows=1, ncols=1+n_facet, figsize=((1+n_facet)*4, 4))

        p      = tri_basis(points, vertex, edge, face, order) # [n_basis, n_dim]
        index  = facet_basis_index_2d(vertex, edge, order)    # [n_face, (order+1)*(order+2)//2]
        plot_2d_points(points, p, ax[0])
        facet_p= p[index] # [n_face, (order+1)*(order+2)//2, n_dim]
        for i in range(index.shape[0]):
            plot_2d_facet_points(points, facet_p[i], index[i], ax[i+1])
        plt.show()

    def plot_test_quad_facet_basis_index():
        order = 3
        points = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]]) # 4x2
        vertex = torch.tensor([[0], [1], [2], [3]]) # 4x1
        edge   = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]]) # 4x2
        face   = torch.tensor([[0, 1, 2, 3]]) # 1x4

        n_facet = len(edge)
        fig, ax = plt.subplots(nrows=1, ncols=1+n_facet, figsize=((1+n_facet)*4, 4))

        p      = quad_basis(points, vertex, edge, face, order) # [n_basis, n_dim]
        index  = facet_basis_index_2d(vertex, edge, order)    # [n_face, (order+1)*(order+2)//2]
        plot_2d_points(points, p, ax[0])
        facet_p= p[index] # [n_face, (order+1)*(order+2)//2, n_dim]
        for i in range(index.shape[0]):
            plot_2d_facet_points(points, facet_p[i], index[i], ax[i+1])
        plt.show()

    def plot_test_tet_facet_basis_index():
        order = 3
        points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) # 4x3
        vertex = torch.tensor([[0], [1], [2], [3]]) # 4x1
        edge   = torch.tensor([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]) # 6x2
        face   = torch.tensor(((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))) # 4x3
        cell   = torch.tensor([[0, 1, 2, 3]]) # 1x4

        n_facet = len(face)
        fig, ax = plt.subplots(nrows=1, ncols=1+n_facet, figsize=((1+n_facet)*4, 4), subplot_kw={'projection': '3d'})

        p      = tet_basis(points, vertex, edge, face, cell, order) # [n_basis, n_dim]
        index  = tet_facet_basis_index(vertex, edge, face, order)    # [n_face, (order+1)*(order+2)//2]
        plot_3d_points(points, p, ax[0])
        facet_p= p[index] # [n_face, (order+1)*(order+2)//2, n_dim]
        for i in range(index.shape[0]):
            plot_3d_facet_points(points, facet_p[i], index[i], True, order, ax[i+1])
        plt.show()

    def plot_test_hex_facet_basis_index():
        order = 2
        points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0],[1.0, 1.0, 1.0]]) # 8x3
        vertex = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7]]) # 8x1
        edge   = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]) # 12x2
        face   = torch.tensor(((0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6), (1, 3, 5, 7), (2, 3, 6, 7), (4, 5, 6, 7))) # 6x4
        cell   = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]) # 1x8

        n_facet = len(face)
        fig, ax = plt.subplots(nrows=1, ncols=1+n_facet, figsize=((1+n_facet)*4, 4), subplot_kw={'projection': '3d'})

        p      = hex_basis(points, vertex, edge, face, cell, order) # [n_basis, n_dim]
        index  = hex_facet_basis_index(vertex, edge, face, order)    # [n_face, (order+1)*(order+2)//2]
        plot_3d_points(points, p, ax[0])
        facet_p= p[index] # [n_face, (order+1)*(order+2)//2, n_dim]
        for i in range(index.shape[0]):
            plot_3d_facet_points(points, facet_p[i], index[i], False, order, ax[i+1])
        plt.show()

    def plot_test_pyr_facet_basis_index():
        order  = 2
        points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) # 5x3
        vertex = torch.tensor([[0], [1], [2], [3], [4]]) # 5x1
        edge   = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]) # 8x2
        face   = ((0, 1, 2, 3), (0, 1, 4), (0, 2, 4), (1, 3, 4), (2, 3, 4)) # 5x4
        cell   = torch.tensor([[0, 1, 2, 3, 4]]) # 1x5

        n_facet = len(face)
        fig, ax = plt.subplots(nrows=1, ncols=1+n_facet, figsize=((1+n_facet)*4, 4), subplot_kw={'projection': '3d'})

        p      = pyr_basis(points, vertex, edge, face, cell, order) # [n_basis, n_dim]
        tri_facet, quad_facet  = mix_facet_basis_index(vertex, edge, face, order)    # [n_face, (order+1)*(order+2)//2]
        plot_3d_points(points, p, ax[0])
        tri_counter, quad_counter = 0, 0
        for i in range(len(face)):
            is_tri = len(face[i]) == 3
            index  = tri_facet[tri_counter] if is_tri else quad_facet[quad_counter]
            plot_3d_facet_points(points, p[index], index, is_tri, order, ax[i+1])
            if is_tri:
                tri_counter += 1
            else:
                quad_counter += 1
        plt.show()

    def plot_test_pri_facet_basis_index():
        order = 2
        points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0]]) # 5x3
        vertex = torch.tensor([[0], [1], [2], [3], [4], [5]]) # 6x1
        edge   = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]) # 9x2
        face   = ((0, 1, 2), (0, 1, 3, 4), (0, 2, 3, 5), (1, 2, 4, 5),(3,4,5)) # 4x4
        cell   = torch.tensor([[0, 1, 2, 3, 4, 5]]) # 1x5

        n_facet = len(face)
        fig, ax = plt.subplots(nrows=1, ncols=1+n_facet, figsize=((1+n_facet)*4, 4), subplot_kw={'projection': '3d'})

        p      = pri_basis(points, vertex, edge, face, cell, order) # [n_basis, n_dim]
        tri_facet, quad_facet  = mix_facet_basis_index(vertex, edge, face, order)    # [n_face, (order+1)*(order+2)//2]
        plot_3d_points(points, p, ax[0])
        tri_counter, quad_counter = 0, 0
        for i in range(len(face)):
            is_tri = len(face[i]) == 3
            index  = tri_facet[tri_counter] if is_tri else quad_facet[quad_counter]
            plot_3d_facet_points(points, p[index], index, is_tri, order, ax[i+1])
            if is_tri:
                tri_counter += 1
            else:
                quad_counter += 1
        plt.show()


    plot_test_pri_facet_basis_index()
    # plot_test_quad_linspace()
    


 


