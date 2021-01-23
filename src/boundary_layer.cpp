#include <iostream>
#include <vector>

#include <ultimaille/all.h>

using namespace UM;

#include "untangle2d.cpp"

int main(int argc, char** argv) {
    Triangles m;
    read_by_extension("../potichat.obj", m);

    int nlock = m.nverts();
    int nbdtr = m.nfacets();

    for (int l : range(3)) {
        SurfaceConnectivity fec(m);
        PointAttribute<int> bid(m.points);
        int nbord = 0;
        for (int v : vert_iter(m)) {
            bid[v] = -1;
            if (!fec.is_boundary_vert(v)) continue;
            bid[v] = nbord++;
        }
        int offv = m.points.create_points(nbord);
        for (int v : range(offv)) {
            if (!fec.is_boundary_vert(v)) continue;
            bid[offv+bid[v]] = -1;
            m.points[offv+bid[v]] = m.points[v];
        }


        for (int c : corner_iter(m)) {
            int opp = fec.opposite(c);
            if (opp>=0) continue;
            vec3 e = fec.geom(c);
            vec3 n = {e.y, -e.x, 0};
            n = n*.01;
            m.points[offv+bid[fec.from(c)]] = m.points[offv+bid[fec.from(c)]] + n;
            m.points[offv+bid[fec.to(c)]]     = m.points[offv+bid[fec.to(c)]] + n;
            int offf = m.create_facets(2);

            m.vert(offf+0, 0) = fec.to(c);
            m.vert(offf+0, 1) = fec.from(c);
            m.vert(offf+0, 2) = offv+bid[fec.from(c)];

            m.vert(offf+1, 0) = fec.to(c);
            m.vert(offf+1, 1) = offv+bid[fec.from(c)];
            m.vert(offf+1, 2) = offv+bid[fec.to(c)];
        }
    }

    Untangle2D opt(m);

    for (int v : vert_iter(m))
        opt.lock[v] = v<nlock;

    double avg_area = 0;
    for (int t : facet_iter(m))
        avg_area += facet_area_2d(m, t)/m.nfacets();

    for (int t=nbdtr; t<m.nfacets(); t++)
        opt.ref_tri[t] = mat<3,2>{{ {0,-1}, {std::sqrt(3.)/2.,.5}, {-std::sqrt(3.)/2.,.5} }}*std::sqrt(4*avg_area/std::sqrt(3.)) / (-2*avg_area);

    opt.maxiter = 100;
    opt.go();

    for (int v : vert_iter(m))
        for (int d : range(2))
            m.points[v][d] = opt.X[v*2+d];

    FacetAttribute<bool> blayer(m);
    for (int t : facet_iter(m)) {
        blayer[t] = t>=nbdtr;
    }
    write_geogram("blopt.geogram", m, { {}, {{"boundary_layer", blayer.ptr}}, {}  });

    return 0;
}

