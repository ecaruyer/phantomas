#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>


#define X 0
#define Y 1
#define Z 2


static double _dot(double* v1, double* v2, int dim) {
    /*
    Computes the dot product of two 1d arrays (assume same dimension).
    */
    double res = 0;
    int i;
    for (i=0; i<dim; i++) {
        res += v1[i] * v2[i];
    }
    return res;
}


static void _vect(double* v1, double* v2, double* res) {
    /*
    Computes the vectorial product of v1 and v2 (assume dimension=3) and place
    results in array res (already instantiated and memory-allocated). 
    */
    res[X] = v1[Y]*v2[Z] - v1[Z]*v2[Y];
    res[Y] = v1[Z]*v2[X] - v1[X]*v2[Z];
    res[Z] = v1[X]*v2[Y] - v1[Y]*v2[X];
    return;
}


static double c_distance_to_line(double* point,
                                 double* a, double* b) {
    /*
    Returns the distance from the given point to the line (ab).
    */
    double ab[3];
    double ap[3];
    double vect_prod[3];
    int i;
    for (i=0; i<3; i++) {
        ab[i] = b[i] - a[i];
        ap[i] = point[i] - a[i];
    }
    _vect(ab, ap, vect_prod);
    return sqrt(_dot(vect_prod, vect_prod, 3) / _dot(ab, ab, 3));
}


static double c_in_truncated_cylinder(double* point, 
                           double* center_1, double* center_2,
                           double* normal_cap_1, double* normal_cap_2,
                           double radius) {
    /*
    Checks whether a given point is within a cylinder. The cylinder is defined by
    the two endpoints of the centerline segment, the normals to the caps and its
    radius. The normals to the cap is assumed oriented towards the inside of the
    cylinder. If positive, then the distance to the centerline is returned.
    If negative, -1 is returned.
    */
    double c_1p[3];
    double c_2p[3];
    int i;
    double dst_to_centerline;
    for (i=0; i<3; i++) {
        c_1p[i] = point[i] - center_1[i];
    }
    if (_dot(normal_cap_1, c_1p, 3) < 0) {
        return -1;
    }
    for (i=0; i<3; i++) {
        c_2p[i] = point[i] - center_2[i];
    }
    if (_dot(normal_cap_2, c_2p, 3) < 0) {
        return -1;
    }
    dst_to_centerline = c_distance_to_line(point, center_1, center_2);
    if (dst_to_centerline > radius) {
        return -1;
    }
    return dst_to_centerline;
}


void c_in_fiber(double* points, int nb_points,
                double* fiber_points, double* fiber_tangents, int fiber_nb_points,
                double fiber_radius, signed int* segment_indices) {
    /*
    Checks whether a collection of points lie within a cylindrical fiber bundle.
    The result is returned as an array indices of fiber segments, assumed
    initialized at -1.
    */
    int i, j, k;
    double n1[3], n2[3];
    double min_dst, dst;
    for (j=0; j<nb_points; j++) {
        min_dst = fiber_radius;
        for (i=0; i<fiber_nb_points-1; i++) {
            for (k=0; k<3; k++) {
                n1[k] = fiber_tangents[i*3 + k];
                n2[k] = -fiber_tangents[(i + 1)*3 + k];
            }
            dst = c_in_truncated_cylinder(&points[j*3], 
                                          &fiber_points[i*3],  &fiber_points[(i + 1)*3],
                                          n1, n2, fiber_radius);
            if (dst > -1 && dst <= min_dst) {
                segment_indices[j] = i;
                min_dst = dst;
//                printf("dst=%f\t", j, min_dst);
            }
        }
    }
//    printf("\n");
    return;
}

