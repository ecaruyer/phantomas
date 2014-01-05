"""
This module implements icosahedron tessellation to provide a (relatively)
regular tessellation of the unit sphere.
"""

import numpy as np
from scipy.sparse import lil_matrix


_golden =  (1 + np.sqrt(5)) / 2

_tessellations = {
    0 : {
        'vertices' :  np.array(
            [[0, 1, _golden],  
            [0, -1, _golden], 
            [0, 1, -_golden], 
            [0, -1, -_golden],
            [1, _golden, 0],
            [-1, _golden, 0],
            [1, -_golden, 0],
            [_golden, 0, 1],
            [_golden, 0, -1]]) / np.sqrt(1 + _golden ** 2),
        'faces' : [
            [0,  7,  1],
            [0,  4,  7],
            [0,  5,  4],
            [2,  4,  5],
            [2,  8,  4],
            [2,  3,  8],
            [1,  7,  6],
            [3,  6,  8],
            [4,  8,  7],
            [6,  7,  8]],
        'nb_edges' : 18
    }
}


def tessellation(order, face_centers=False, face_areas=False):
    """
    Constructs a tessellation from a subdivision of an icosahedron.  
   
    Parameters
    ----------
    order : int
        The order of the recursive tessellation.
    compute_face_centers : bool, optional
    compute_face_areas : bool, optional

    Returns
    -------
    vertices : array-lie, (N, 3)
        The array of vertices
    faces : array-like
        The array of triangles. Each triangle is a tuple with the indices of
        the 3 corresponding vertices.
    nb_edges : int

    Notes
    -----
    We have only kept 10 faces out of 20, so as to generate a sphere 
    tessellation adding the antipodal symmetric. Results are cached for 
    future use in the dictionary _tessellations.
    """
    if order in _tessellations:
        result = [_tessellations[order]['vertices'], \
                  _tessellations[order]['faces'], \
                  _tessellations[order]['nb_edges']]
        if face_centers:
            result.append(_tessellations[order]['face_centers'])
        if face_areas:
            result.append(_tessellations[order]['face_areas'])
        return result

    # recursive call
    (vertices, faces, nb_edges) = tessellation(order - 1)
    
    nb_vertices = vertices.shape[0]
    nb_faces = len(faces)
    
    new_nb_vertices = nb_vertices + nb_edges
    new_nb_faces = 4*nb_faces
    new_nb_edges = 2*nb_edges + 3*nb_faces
    
    new_vertices = np.zeros((new_nb_vertices, 3))
    new_vertices[:nb_vertices] = vertices
    
    new_faces = []
    
    # in the array indices_middle, we store the index of the middle point of a
    # segment... s.t. new_vertices[indices_middle[i, j]] is the position of 
    # the middle of positions[i],positions[j], after reprojecting onto the
    # sphere. Note that since the indices are sorted, we only store the upper 
    # triangular part of this matrix.
    indices_middle = lil_matrix((nb_vertices, nb_vertices), dtype=np.int)
    
    next = nb_vertices
    
    for face in faces:
    	face.sort()
    	indexA = face[0]
    	indexB = face[1]
    	indexC = face[2]
    
    	indexAB = indices_middle[indexA, indexB]
    	if indexAB == 0:
    	    # if indexAB equals -1, the middle of AB has not been computed yet
    	    new_vertices[next] = (vertices[indexA] + vertices[indexB]) / 2
    	    new_vertices[next] /= np.sqrt((new_vertices[next] ** 2).sum())
    	    indexAB = next
    	    indices_middle[indexA, indexB] = indexAB
    	    next += 1
    
    	indexAC = indices_middle[indexA, indexC]
    	if indexAC == 0:
    	    # if indexAB equals -1, the middle of AB has not been computed yet
    	    new_vertices[next] = (vertices[indexA] + vertices[indexC]) / 2
    	    new_vertices[next] /= np.sqrt((new_vertices[next] ** 2).sum())
    	    indexAC = next
    	    indices_middle[indexA, indexC] = indexAC
    	    next += 1
    
    	indexBC = indices_middle[indexB, indexC]
    	if indexBC == 0:
    	    # if indexAB equals -1, the middle of AB has not been computed yet
    	    new_vertices[next] = (vertices[indexB] + vertices[indexC]) / 2
    	    new_vertices[next] /= np.sqrt((new_vertices[next] ** 2).sum())
    	    indexBC = next
    	    indices_middle[indexB, indexC] = indexBC
    	    next += 1
    
    	new_faces.append([indexA, indexAB, indexAC])
    	new_faces.append([indexB, indexAB, indexBC])
    	new_faces.append([indexC, indexAC, indexBC])
    	new_faces.append([indexAB, indexBC, indexAC])
    _tessellations[order] = {
        'vertices' : new_vertices,
        'faces' : new_faces,
        'nb_edges' : new_nb_edges
    }
    result = [new_vertices, new_faces, new_nb_edges]
    if face_centers:
        _tessellations[order]['face_centers'] = \
            compute_face_centers(new_vertices, new_faces)
        result.append(_tessellations[order]['face_centers'])
    if face_areas:
        _tessellations[order]['face_areas'] = \
            compute_face_areas(new_vertices, new_faces)
        result.append(_tessellations[order]['face_areas'])
    return result


def compute_face_centers(vertices, faces):
    """
    Computes the (triangular) faces center of mass.

    Parameters
    ----------
    vertices : array-lie, (N, 3)
        The array of vertices
    faces : array-like
        The array of triangles. Each triangle is a tuple with the indices of the 3
        corresponding vertices.
    """    
    centers = np.zeros((len(faces), 3))
    for i in range(len(faces)):
        centers[i] = vertices[faces[i]].mean(0)
    return centers


def compute_face_areas(vertices, faces):
    """
    Computes the (triangular) face areas.

    Parameters
    ----------
    vertices : array-lie, (N, 3)
        The array of vertices
    faces : array-like
        The array of triangles. Each triangle is a tuple with the indices of the 3
        corresponding vertices.
    """
    areas = np.zeros(len(faces))
    for i in range(len(faces)):
        ab = vertices[faces[i][1]] - vertices[faces[i][0]]
        ac = vertices[faces[i][2]] - vertices[faces[i][0]]
        areas[i] = np.linalg.norm(np.cross(ab, ac)) / 2
    return areas


if __name__=='__main__':
    tessellation_order = 3
    print "Computing recursive tessellation of order %d" % tessellation_order
    vertices, faces, nb_edges, centers, areas = \
        tessellation(tessellation_order, face_centers=True, face_areas=True)
    centers *= 1.01

    print "Order = %d, nb_vertices=%d, nb_faces=%d, nb_edges=%d" \
              % (tessellation_order, vertices.shape[0], len(faces), nb_edges)

    face_surface = 2 * np.pi / len(faces)
    dist_face_centers = np.sqrt((2 * face_surface) / (3 * np.sqrt(3)))
    angle_face_centers = 2 * np.arcsin((dist_face_centers / 2)) * 180 / np.pi
    print "Angle between two cell centers: %.1f (approx)" % angle_face_centers
    
    verts = np.zeros((len(faces), 3, 3))
    for i in range(len(faces)):
        verts[i] = vertices[faces[i]]

    fig = plt.figure()
    ax = fig.add_subplot('111', projection='3d', aspect='equal')

    ax.add_collection(art3d.Poly3DCollection(verts, linewidth=1, color='grey', facecolor='w'))
    # We demonstrate here that adding the antipodal symmetric of the faces, we get a full
    # sphere tessellation.
    ax.add_collection(art3d.Poly3DCollection(-verts, linewidth=1, color='grey', facecolor='w'))
    point_cloud = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=areas)
    fig.colorbar(point_cloud)
    max_val = .7
    ax.set_xlim3d(-max_val, max_val)
    ax.set_ylim3d(-max_val, max_val)
    ax.set_zlim3d(-max_val, max_val)

    plt.show()
