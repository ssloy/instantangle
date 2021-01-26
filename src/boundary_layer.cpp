#include <iostream>
#include <vector>

#include <ultimaille/all.h>

using namespace UM;

#include "untangle2d.cpp"

int main(int argc, char** argv) {
    Triangles m;
    read_by_extension("../potitepieuvre.obj", m);
    PointAttribute<int> resp(m.points);

    for (int v : vert_iter(m))
        resp[v] = -1;

    int nlock = m.nverts();
    int nbdtr = m.nfacets();

    CornerAttribute<bool> brd(m);

    for (int l : range(1)) {
        SurfaceConnectivity fec(m);
        int nbord = 0;
        for (int v : vert_iter(m)) {
            if (!fec.is_boundary_vert(v)) continue;
            resp[v] = m.nverts() + nbord++;
        }
        int offv = m.points.create_points(nbord);
        for (int v : range(offv)) {
            if (!fec.is_boundary_vert(v)) continue;
            resp[resp[v]] = -1;
            m.points[resp[v]] = m.points[v];
        }


        for (int c : corner_iter(m)) {
            int opp = fec.opposite(c);
            if (opp>=0) continue;
            brd[c] = true;
            vec3 e = fec.geom(c);
            vec3 n = {e.y, -e.x, 0};
            n = n*.01;
            m.points[resp[fec.from(c)]] = m.points[resp[fec.from(c)]] + n;
            m.points[resp[fec.to(c)]]     = m.points[resp[fec.to(c)]] + n;
            int offf = m.create_facets(2);

            m.vert(offf+0, 0) = fec.to(c);
            m.vert(offf+0, 1) = fec.from(c);
            m.vert(offf+0, 2) = resp[fec.from(c)];

            m.vert(offf+1, 0) = fec.to(c);
            m.vert(offf+1, 1) = resp[fec.from(c)];
            m.vert(offf+1, 2) = resp[fec.to(c)];
            if (resp[fec.from(c)]<0 || resp[fec.to(c)]<0) std::cerr << "ERROR\n";
        }
    }

    Untangle2D opt(m);

    for (int v : vert_iter(m))
        opt.lock[v] = v<nlock;

    double avg_area = 0;
    for (int t : facet_iter(m))
        avg_area += facet_area_2d(m, t)/m.nfacets();

    for (int t=nbdtr; t<m.nfacets(); t++)
        opt.ref_tri[t] = mat<3,2>{{ {0,-1}, {std::sqrt(3.)/2.,.5}, {-std::sqrt(3.)/2.,.5} }}*std::sqrt(4*avg_area/std::sqrt(3.)) / (-2*avg_area)*2;

    opt.maxiter = 100;
  opt.go();

    for (int v : vert_iter(m))
        for (int d : range(2))
            m.points[v][d] = opt.X[v*2+d];

    FacetAttribute<bool> blayer(m);
    for (int t : facet_iter(m)) {
        blayer[t] = t>=nbdtr;
    }

    write_geogram("blopt.geogram", m, { {{"resp", resp.ptr}}, {{"boundary_layer", blayer.ptr}}, {{"brd", brd.ptr}}  });

    return 0;
}

