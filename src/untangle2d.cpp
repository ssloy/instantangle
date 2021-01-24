#include <iostream>
#include <limits>
#include <cassert>
#include <algorithm>
#include <array>
#include <chrono>
#include <sstream>
#include <iomanip>

#include <ultimaille/all.h>
#include <OpenNL_psm/OpenNL_psm.h>
#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
        std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
        initializer(omp_priv = std::vector<double>(omp_orig.size(), 0))

template <typename T> auto square(T &number) { return number * number; }

#define EPS_FROM_THE_THEOREM 0

double get_average_edge_size(Surface &m) {
    double sum = 0;
    int nb = 0;
    for (int f : facet_iter(m))
        for (int i : facet_vert_iter(m, f)) {
            int j = m.vert(f, (i+1)%m.facet_size(f));
            sum += (m.points[i] - m.points[j]).norm();
            nb++;
        }
    assert(nb > 0);
    return sum / double(nb);
}

double triangle_area_3d(Triangles &m, int f) {
    vec3 eij = m.points[m.vert(f, 1)] - m.points[m.vert(f, 0)];
    vec3 eik = m.points[m.vert(f, 0)] - m.points[m.vert(f, 2)];
    return 0.5*cross(eij, eik).norm();
}

double facet_area_2d(Surface &m, int f) {
    double area = 0;
    for (int v=0; v<m.facet_size(f); v++) {
        vec3 a = m.points[m.vert(f, v)];
        vec3 b = m.points[m.vert(f, (v+1)%m.facet_size(f))];
        area += (b.y-a.y)*(b.x+a.x)/2;
    }
    return area;
}

double triangle_area_2d(vec2 a, vec2 b, vec2 c) {
    return .5*((b.y-a.y)*(b.x+a.x) + (c.y-b.y)*(c.x+b.x) + (a.y-c.y)*(a.x+c.x));
}

double project_triangle(const vec3& p0, const vec3& p1, const vec3& p2, vec2& z0, vec2& z1, vec2& z2) {
    vec3 X = (p1 - p0).normalize(); // construct an orthonormal 3d basis
    vec3 Z = cross(X, p2 - p0).normalize();
    vec3 Y = cross(Z, X);

    z0 = vec2(0,0); // project the triangle to the 2d basis (X,Y)
    z1 = vec2((p1 - p0).norm(), 0);
    z2 = vec2((p2 - p0)*X, (p2 - p0)*Y);
    return triangle_area_2d(z0, z1, z2);
}

inline double chi(double eps, double det) {
    if (det>0)
        return (det + std::sqrt(eps*eps + det*det))*.5;
    return .5*eps*eps / (std::sqrt(eps*eps + det*det) - det);
}

inline double chi_deriv(double eps, double det) {
    return .5+det/(2.*std::sqrt(eps*eps + det*det));
}

struct Untangle2D {
    Untangle2D(Triangles &mesh) : m(mesh), X(m.nverts()*2), lock(m.points), ref_tri(m), J(m), K(m), det(m) {
        calls = 0;
        avglen = get_average_edge_size(m);

        for (int v : vert_iter(mesh))
            for (int d : range(2))
                X[2*v+d] = m.points[v][d];

        rebuild_reference_geometry();
    }

    void rebuild_reference_geometry() {
        for (int t : facet_iter(m)) {
            vec3 pi = m.points[m.vert(t, 0)];
            vec3 pj = m.points[m.vert(t, 1)];
            vec3 pk = m.points[m.vert(t, 2)];
            double area = facet_area_2d(m, t);
            if (area<=0) {
                std::cerr << "Error: the reference area must be positive" << std::endl;
                //                  return;
            }
            //              double a = std::sqrt(4*area/std::sqrt(3.));
            //              if (t>=1792)
            //              ref_tri[t] = mat<3,2>{{ {0,-1}, {std::sqrt(3.)/2.,.5}, {-std::sqrt(3.)/2.,.5} }}*a / (-2*area) ; // equilateral triangle with unit side length (sqrt(3)/4 area): three non-unit normal vectors
            //              else
            ref_tri[t] = mat<3,2>{{ {(pk-pj).y, -(pk-pj).x}, {(pi-pk).y, -(pi-pk).x}, {(pj-pi).y, -(pj-pi).x} }}/(-2.*area);
        }
    }

    void no_two_coverings() {
        SurfaceConnectivity fec(m);
        for (int v : vert_iter(m)) {
            if (fec.is_boundary_vert(v)) continue;
            std::vector<int> ring;
            {
                int cir = fec.v2c[v];
                do {
                    ring.push_back(fec.to(cir));
                    cir = fec.next_around_vertex(cir);
                } while (cir != fec.v2c[v]);
            }

/*
            int off = m.create_facets(ring.size());
            for (int lv=0; lv<ring.size(); lv++) {
                m.vert(off+lv, 0) = ring[lv];
                m.vert(off+lv, 1) = ring[(lv+1)%ring.size()];
                m.vert(off+lv, 2) = ring[(lv+1+ring.size()/2)%ring.size()];
            }
*/
                {
                int off = m.create_facets(2);
                if (ring.size()>=5) {
                    m.vert(off, 0) = ring[0];
                    m.vert(off, 1) = ring[2];
                    m.vert(off, 2) = ring[4];

                    m.vert(off+1, 0) = ring[1];
                    m.vert(off+1, 1) = ring[3];
                    m.vert(off+1, 2) = ring[5%ring.size()];
                }
            }

            {
                int off = m.create_facets(ring.size()-2);
                int v0 = ring[0];
                for (int lv=1; lv+1<ring.size(); lv++) {
                    m.vert(off+lv-1, 0) = v0;
                    m.vert(off+lv-1, 1) = ring[lv];
                    m.vert(off+lv-1, 2) = ring[lv+1];
                }
            }
            {
                int off = m.create_facets(ring.size()-2);
                int v0 = ring[2];
                for (int lv=1; lv+1<ring.size(); lv++) {
                    m.vert(off+lv-1, 0) = v0;
                    m.vert(off+lv-1, 1) = ring[(lv+2)%ring.size()];
                    m.vert(off+lv-1, 2) = ring[(lv+1+2)%ring.size()];
                }
            }

        }
        rebuild_reference_geometry();

        write_geogram("tri.geogram", m);
    }

    void compute_hessian_pattern() {
        hessian_pattern = std::vector<int>(m.nverts());

        // enumerate all non-zero entries of the hessian matrix
        std::vector<std::tuple<int, int> > nonzero;
        for (int t : facet_iter(m)) {
            for (int i : range(3)) {
                int vi = m.vert(t,i);
                if (lock[vi]) continue;
                for (int j : range(3)) {
                    int vj = m.vert(t,j);
                    if (lock[vj]) continue;
                    nonzero.emplace_back(vi, vj);
                }
            }
        }
        for (int i=0; i<m.nverts(); i++)
            if (lock[i])
                nonzero.emplace_back(i, i);

        // well those are not triplets, because we have stored indices only, but you get the idea
        // sort the nonzero array, and then determine the number of nonzero entries per row (the pattern)
        int ntriplets = nonzero.size();
        std::sort(nonzero.begin(), nonzero.end());
        int a=0, b=0;
        int nnz = 0;
        for (int v : vert_iter(m)) {
            a = b;
            while (b<ntriplets && std::get<0>(nonzero[++b])<v+1);
            int cnt = 1;
            for (int i=a; i<b-1; i++)
                cnt += (std::get<1>(nonzero[i]) != std::get<1>(nonzero[i+1]));
            hessian_pattern[v] = cnt;
            nnz += cnt;
        }

        if (debug>0) {
            std::cerr << "hessian matrix #non-zero entries: " << nnz << std::endl;
            std::cerr << "hessian matrix avg #nnz per row: " << double(nnz)/double(m.nverts()) << std::endl;
        }
    }

    void evaluate_jacobian() {
        if (debug>3) std::cerr << "evaluate the jacobian...";
        detmin = std::numeric_limits<double>::max();
        ninverted = 0;
#pragma omp parallel for reduction(min:detmin) reduction(+:ninverted)
        for (int t=0; t<m.nfacets(); t++) {
            mat<2,2> &J = this->J[t];
            J = {};
            for (int i=0; i<3; i++)
                for (int d : range(2))
                    J[d] += ref_tri[t][i]*X[2*m.vert(t,i) + d];
            det[t] = J.det();
            detmin = std::min(detmin, det[t]);
            ninverted += (det[t]<=0);

            mat<2,2> &K = this->K[t];
            K = { // dual basis
                {{
                     +J[1].y,
                     -J[1].x
                 },
                {
                    -J[0].y,
                    +J[0].x
                }}
            };
        }
        if (debug>3) std::cerr << "ok" << std::endl;
    }

    double evaluate_energy() {
        evaluate_jacobian();
        double E = 0;
#pragma omp parallel for reduction(+:E)
        for (int t=0; t<m.nfacets(); t++) {
            double chi_ = chi(eps, det[t]);
            double f = (square(J[t][0]) + square(J[t][1]))/chi_;
            double g = (1+square(det[t]))/chi_;
            E += (1.-theta)*f + theta*g;
        }
        return E;
    }

    bool go() {
        calls++;
        compute_hessian_pattern();
        evaluate_jacobian();

        double e0 = 1e-8;
        for (int iter=0; iter<maxiter; iter++) {
            if (debug>0) std::cerr << "iteration #" << iter << std::endl;
            double det_prev = detmin;

#if !EPS_FROM_THE_THEOREM
            eps = detmin>0 ? e0 : std::sqrt(square(e0) + 0.04*square(detmin));
#endif

            if (debug>0) std::cerr << "detmin: " << detmin << " ninv: " << ninverted << std::endl;
            double E_prev = evaluate_energy();

if (detmin<0) {
//#if 1
            const hlbfgs_optimizer::simplified_func_grad_eval func = [&](const std::vector<double>& X, double& F, std::vector<double>& G) {
                std::fill(G.begin(), G.end(), 0);
                F = evaluate_energy();

#pragma omp parallel for reduction(vec_double_plus:G)
                for (int t=0; t<m.nfacets(); t++) {
                    double c1 = chi(eps, det[t]);
                    double c2 = chi_deriv(eps, det[t]);

                    double f = (square(J[t][0]) + square(J[t][1]))/c1;
                    double g = (1+square(det[t]))/c1;

                    for (int dim : range(2)) {
                        vec2 a = J[t][dim]; // tangent basis
                        vec2 b = K[t][dim]; // dual basis
                        vec2 dfda = (a*2. - b*f*c2)/c1;
                        vec2 dgda = b*(2*det[t]-g*c2)/c1;
                        for (int i=0; i<3; i++) {
                            int v = m.vert(t,i);
                            if (!lock[v])
                                G[v*2+dim] += (dfda*(1.-theta) + dgda*theta)*ref_tri[t][i];
                        }
                    }
                }
            };


            hlbfgs_optimizer opt(func);
            opt.set_epsg(bfgs_threshold);
            opt.set_max_iter(bfgs_maxiter);
            opt.set_verbose(true);
            opt.optimize(X);
//#else
} else {
            std::vector<double> deltaX, deltaY;
            newton(0, deltaX);
            newton(1, deltaY);

            line_search(deltaX, deltaY);
            }
//#endif

            double E = evaluate_energy();
#if EPS_FROM_THE_THEOREM
            double sigma = std::max(1.-E/E_prev, 1e-1);
            if (detmin>=0)
                eps *= (1-sigma);
            else
                eps *= 1 - (sigma*std::sqrt(square(detmin) + square(eps)))/(std::abs(detmin) + std::sqrt(square(detmin) + square(eps)));
#endif
                if  (detmin>0 && std::abs(E_prev - E)/E<1e-7) break;
        }

        if (detmin<0 && calls%30==0) {
            for (int f : facet_iter(m)) {
                if (det[f]>0) continue;
                for (int i : range(3)) {
                    int v = m.vert(f,i);
                    if (lock[v]) continue;
                    for (int d : range(2))
                        X[v*2+d] += (rand()/(double)RAND_MAX*2-1)*avglen/10.;
                }
            }
        }

        if (debug>0) std::cerr << "E: " << evaluate_energy() << " detmin: " << detmin << " ninv: " << ninverted << std::endl;
        return !ninverted;
    }

    void newton(const int dim, std::vector<double> &sln) {
        int nvar = m.nverts();
        sln = std::vector<double>(nvar, 0);

        nlNewContext();
        nlEnable(NL_NO_VARIABLES_INDIRECTION);
        nlSolverParameteri(NL_NB_VARIABLES, nvar);
        nlSolverParameteri(NL_SOLVER, NL_CG);
        nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI);
        nlSolverParameteri(NL_MAX_ITERATIONS, NLint(nlmaxiter));
        nlSolverParameterd(NL_THRESHOLD, nlthreshold);
        nlEnable(NL_VARIABLES_BUFFER);
        nlBegin(NL_SYSTEM);

        nlBindBuffer(NL_VARIABLES_BUFFER, 0, sln.data(), NLuint(sizeof(double)));

        nlBegin(NL_MATRIX_PATTERN);
        for (auto [row, size] : enumerate(hessian_pattern))
            nlSetRowLength(row, size);
        nlEnd(NL_MATRIX_PATTERN);

        nlBegin(NL_MATRIX);

        if (debug>3) std::cerr << "preparing the matrix...";
        for (int t : facet_iter(m)) {
            double c1 = chi(eps, det[t]);
            double c2 = chi_deriv(eps, det[t]);

            double f = (J[t][0]*J[t][0] + J[t][1]*J[t][1])/c1;
            double g = (1+square(det[t]))/c1;

            vec2 a = J[t][dim]; // tangent basis
            vec2 b = K[t][dim]; // dual basis
            vec2 dfda = (a*2. - b*f*c2)/c1;
            vec2 dgda = b*(2*det[t]-g*c2)/c1;

            for (int i=0; i<3; i++) {
                int v = m.vert(t,i);
                if (!lock[v])
                    nlAddIRightHandSide(v, (dfda*(1.-theta) + dgda*theta)*ref_tri[t][i]);
            }

            mat<2,1> A = {{{a.x}, {a.y}}};
            mat<2,1> B = {{{b.x}, {b.y}}};

            mat<2,2> Fii = mat<2,2>::identity()*(2./c1) - (A*B.transpose() + B*A.transpose())*((2.*c2)/square(c1)) + (B*B.transpose())*((2.*f*square(c2))/square(c1));
            mat<2,2> Gii = (B*B.transpose())*( 2./c1 - (2.*c2*(2.*det[t] - g*c2))/square(c1) );
            mat<2,2> Pii = Fii*(1.-theta) + Gii*theta;

            for (int i=0; i<3; i++) {
                int vi = m.vert(t,i);
                if (lock[vi]) continue;

                for (int j=0; j<3; j++) {
                    int vj = m.vert(t,j);
                    if (lock[vj]) continue;
                    nlAddIJCoefficient(vi, vj, ref_tri[t][i]*(Pii*ref_tri[t][j]));
                }
            }
        }
        for (int v : vert_iter(m))
            if (lock[v])
                nlAddIJCoefficient(v, v, 1);
        if (debug>3) std::cerr << "ok" << std::endl;

        nlEnd(NL_MATRIX);
        nlEnd(NL_SYSTEM);
        if (debug>1) std::cerr << "solving the linear system...";
        nlSolve();
        if (debug>1) std::cerr << "ok" << std::endl;

        if (debug>1) {
            int used_iters=0;
            double elapsed_time=0.0;
            double gflops=0.0;
            double error=0.0;
            int nnz = 0;
            nlGetIntegerv(NL_USED_ITERATIONS, &used_iters);
            nlGetDoublev(NL_ELAPSED_TIME, &elapsed_time);
            nlGetDoublev(NL_GFLOPS, &gflops);
            nlGetDoublev(NL_ERROR, &error);
            nlGetIntegerv(NL_NNZ, &nnz);
            std::cerr << ("Linear solve") << "   " << used_iters << " iters in " << elapsed_time << " seconds " << gflops << " GFlop/s" << "  ||Ax-b||/||b||=" << error << std::endl;
        }
        nlDeleteContext(nlGetCurrent());
    }

    double line_search(std::vector<double> &deltaX, std::vector<double> &deltaY) {
        if (debug>2) std::cerr << "line search...";
        double Tau[] = {4,3,2,1,.5,.25};
        double E_min = std::numeric_limits<double>::max();
        double tau_min = Tau[0];
        std::vector<double> pts = X;

        for (double &tau : Tau) {
            for (int v : vert_iter(m)) {
                X[v*2+0] = pts[v*2+0] - deltaX[v]*tau;
                X[v*2+1] = pts[v*2+1] - deltaY[v]*tau;
            }

            double E = evaluate_energy();
            if (E<E_min) {
                E_min = E;
                tau_min = tau;
            }
        }
        for (int v : vert_iter(m)) {
            X[v*2+0] = pts[v*2+0] - deltaX[v]*tau_min;
            X[v*2+1] = pts[v*2+1] - deltaY[v]*tau_min;
        }
        if (debug>2) std::cerr << "ok, tau_min: " << tau_min << std::endl;;
        return tau_min;
    }


    ////////////////////////////////
    // Untangle2D state variables //
    ////////////////////////////////

    // optimization input parameters
    Triangles &m;           // the mesh to optimize
    double theta = 1./4.;   // the energy is (1-theta)*(shape energy) + theta*(area energy)
    int maxiter = 3;    // max number of outer iterations
    double bfgs_threshold = 1e-1;
    int bfgs_maxiter = 50; // max number of inner iterations
    int nlmaxiter = 15000;
    double nlthreshold = 1e-20;

    int debug = 1;          // verbose level

    // optimization state variables

    std::vector<double> X;     // current geometry
    PointAttribute<bool> lock; // currently lock = boundary vertices
    FacetAttribute<mat<3,2>> ref_tri;
    FacetAttribute<mat<2,2>> J; // per-tet Jacobian matrix = [[JX.x JX.y, JX.z], [JY.x, JY.y, JY.z], [JZ.x, JZ.y, JZ.z]]
    FacetAttribute<mat<2,2>> K; // per-tet dual basis: det J = dot J[i] * K[i]
    FacetAttribute<double> det; // per-tet determinant of the Jacobian matrix
    double eps;       // regularization parameter, depends on min(jacobian)
    double avglen;

    double detmin;    // min(jacobian) over all tetrahedra
    int ninverted; // number of inverted tetrahedra

    std::vector<int> hessian_pattern; // number of non zero entries per row of the hessian matrix
    int calls;
};

