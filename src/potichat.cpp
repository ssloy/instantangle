#undef NDEBUG
#include <cassert>
#include <iostream>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <ultimaille/all.h>

using namespace UM;
#include "untangle2d.cpp"

const GLuint width = 800, height = 800;
Triangles mesh;
std::vector<bool> to_kill_coverings;
std::vector<bool> to_kill_blayer;

Untangle2D *opt = nullptr;

int vertgrab = -1;

void get_bbox(const PointSet &pts, vec3 &min, vec3 &max) {
    min = max = pts[0];
    for (auto const &p : pts) {
        for (int d : range(3)) {
            min[d] = std::min(min[d], p[d]);
            max[d] = std::max(max[d], p[d]);
        }
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    std::cerr << "key_callback(" << window << ", " << key << ", " << scancode << ", " << action << ", " << mode << ");" << std::endl;
    if (GLFW_RELEASE == action) {
        return;
    }
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        for (int v : vert_iter(mesh)) {
            for (int d=0; d<2; d++) {
                mesh.points[v][d] = opt->X[v*2+d];
            }
        }
        {
            Triangles m = mesh;
            m.delete_facets(to_kill_coverings);
            write_geogram("optfull.geogram", m);
        }

            Triangles m = mesh;
            m.delete_facets(to_kill_blayer);
            return;

        PointAttribute<double> A(m.points);
        SurfaceConnectivity fec(m);
        for (int v : vert_iter(m)) {
            A[v] = 2*M_PI;
            int c = fec.v2c[v];
            if (c<0) continue;
            A[v] = 0;
            do {
                vec3 e1 =  fec.geom(c);
                vec3 e2 = -fec.geom(fec.prev(c));
                double a = atan2(vec3{0,0,1} * cross(e1, e2), e1*e2);//atan2(e1.x*e2.y-e1.y*e2.x,e1.x*e2.x+e1.y*e2.y);
                if (a<0) std::cerr << "ERROR: NEGATIVE ANGLE!" << std::endl;
                A[v] += a;
                c = fec.c2c[c];
            } while (c!=fec.v2c[v]);
        }
        write_geogram("optint.geogram", m, {{{"angle", A.ptr}}, {}, {}});
    }
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (vertgrab<0) return;
    xpos = (xpos*2)/width  - 1;
    ypos = -(ypos*2)/height + 1;
    opt->X[vertgrab*2+0] = xpos;
    opt->X[vertgrab*2+1] = ypos;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (action==GLFW_RELEASE) vertgrab = -1;
    if (action==GLFW_PRESS && button == 0) {
        vec2 pos;
        glfwGetCursorPos (window, &pos.x, &pos.y);
        pos.x = (pos.x*2)/width  - 1;
        pos.y = -(pos.y*2)/height + 1;

        double bestd = 1e10;
        int besti = -1;
        for (int v : vert_iter(mesh)) {
            vec2 p = {opt->X[v*2+0], opt->X[v*2+1]};
            double len2 = (pos - vec2(p.x, p.y)).norm2();
            if (bestd>len2) {
                bestd = len2;
                besti = v;
            }
        }
        if (besti>=0 && bestd<.001) {
            opt->lock[besti] = !opt->lock[besti];
            if (opt->lock[besti])
            vertgrab = besti;
        }
    }
}


void read_n_compile_shader(const std::string filename, GLuint &hdlr, GLenum shaderType) {
    std::cerr << "Loading shader file " << filename << "... ";
    std::ifstream is(filename, std::ios::in|std::ios::binary|std::ios::ate);
    if (!is.is_open()) {
        std::cerr << "failed" << std::endl;
        return;
    }
    std::cerr << "ok" << std::endl;

    long size = is.tellg();
    char *buffer = new char[size+1];
    is.seekg(0, std::ios::beg);
    is.read(buffer, size);
    is.close();
    buffer[size] = 0;

    std::cerr << "Compiling the shader " << filename << "... ";
    hdlr = glCreateShader(shaderType);
    glShaderSource(hdlr, 1, (const GLchar**)&buffer, NULL);
    glCompileShader(hdlr);
    GLint success;
    glGetShaderiv(hdlr, GL_COMPILE_STATUS, &success);
    std::cerr << (success ? "ok" : "failed") << std::endl;

    GLint log_length;
    glGetShaderiv(hdlr, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length>0) {
        std::vector<char> v(log_length, 0);
        glGetShaderInfoLog(hdlr, log_length, NULL, v.data());
        if (strlen(v.data())>0) {
            std::cerr << v.data() << std::endl;
        }
    }
    delete [] buffer;
}

GLuint set_shaders(const std::string vsfile, const std::string fsfile) {
    GLuint vert_hdlr, frag_hdlr;
    read_n_compile_shader(vsfile, vert_hdlr, GL_VERTEX_SHADER);
    read_n_compile_shader(fsfile, frag_hdlr, GL_FRAGMENT_SHADER);

    std::cerr << "Linking shaders... ";
    GLuint prog_hdlr = glCreateProgram();
    glAttachShader(prog_hdlr, vert_hdlr);
    glAttachShader(prog_hdlr, frag_hdlr);
    glDeleteShader(vert_hdlr);
    glDeleteShader(frag_hdlr);
    glLinkProgram(prog_hdlr);

    GLint success;
    glGetProgramiv(prog_hdlr, GL_LINK_STATUS, &success);
    std::cerr << (success ? "ok" : "failed") << std::endl;

    GLint log_length;
    glGetProgramiv(prog_hdlr, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length>0) {
        std::vector<char> v(log_length);
        glGetProgramInfoLog(prog_hdlr, log_length, NULL, v.data());
        if (strlen(v.data())>0) {
            std::cerr << v.data() << std::endl;
        }
    }
    return prog_hdlr;
}

int setup_window(GLFWwindow* &window, const GLuint width, const GLuint height) {
    std::cerr << "Starting GLFW context, OpenGL 3.3" << std::endl;
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    window = glfwCreateWindow(width, height, "OpenGL starter pack", NULL, NULL);
    glfwMakeContextCurrent(window);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        return -1;
    }

    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);

    glfwSetMouseButtonCallback(window, mouse_button_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize OpenGL context" << std::endl;
        return -1;
    }
    return 0;
}

void heat_map_color(float value, float &red, float &green, float &blue) {
    const int NUM_COLORS = 4;
    static float color[NUM_COLORS][3] = { {.3,.3,1}, {.3,1,.3}, {1,1,.3}, {1,.3,.3} };
    // A static array of 4 colors:  (blue,   green,  yellow,  red) using {r,g,b} for each.

    int idx1;        // |-- Our desired color will be between these two indexes in "color".
    int idx2;        // |
    float fractBetween = 0;  // Fraction between "idx1" and "idx2" where our value is.

    if (value <= 0)      {  idx1 = idx2 = 0;            }    // accounts for an input <=0
    else if (value >= 1) {  idx1 = idx2 = NUM_COLORS-1; }    // accounts for an input >=0
    else {
        value = value * (NUM_COLORS-1);        // Will multiply value by 3.
        idx1  = floor(value);                  // Our desired color will be after this index.
        idx2  = idx1+1;                        // ... and before this index (inclusive).
        fractBetween = value - float(idx1);    // Distance between the two indexes (0-1).
    }

    red   = (color[idx2][0] - color[idx1][0])*fractBetween + color[idx1][0];
    green = (color[idx2][1] - color[idx1][1])*fractBetween + color[idx1][1];
    blue  = (color[idx2][2] - color[idx1][2])*fractBetween + color[idx1][2];
}



int main(int argc, char** argv) {
    const double boxsize = 2.;
    const double shrink  = 1.3;
    SurfaceAttributes attr = read_by_extension("../potitepieuvre-boundary-layer.geogram", mesh);



    std::vector<int> dectri;

{

    Triangles mdec;
    read_by_extension("../potichatdec2.obj", mdec);
    std::vector<vec3> vmerge = *(mesh.points.data);
    vmerge.insert(vmerge.end(), mdec.points.data->begin(), mdec.points.data->end());
    std::vector<int> old2new;
    colocate(vmerge, old2new, 1e-2);
    int n = mesh.nverts();
    dectri = std::vector<int>(mdec.nfacets()*3);
    for (int t : facet_iter(mdec)) {
        for (int i : range(3)) {
            dectri[t*3+i] = old2new[n+mdec.vert(t, i)];
        }
//      assert(old2new[v+n]<n);
    }

}
    FacetAttribute<bool> tblayer("boundary_layer", attr, mesh);
    PointAttribute<bool> vblayer(mesh.points);

//    mesh.attr_facets.clear();
    to_kill_coverings = std::vector<bool>(mesh.nfacets(), false);
    to_kill_blayer = std::vector<bool>(mesh.nfacets(), false);

    int nintvrt = 0;
    int ninttri = 0;
    for (int t : facet_iter(mesh))
        ninttri += !tblayer[t];

    SurfaceConnectivity fec(mesh);
    for (int v : vert_iter(mesh)) {
        vblayer[v] = true;
        int c = fec.v2c[v];
        do {
            if (!tblayer[fec.c2f[c]]) {
                vblayer[v] = false;
                nintvrt++;
                break;
            }
            c = fec.next_around_vertex(c);
        } while (c!=fec.v2c[v]);
    }

    vec3 bbmin, bbmax;
    get_bbox(mesh.points, bbmin, bbmax);
    float maxside = std::max(bbmax.x-bbmin.x, bbmax.y-bbmin.y);
    for (vec3 &p : mesh.points) {
        p = (p - (bbmax+bbmin)/2.)*boxsize/(shrink*maxside);
        p.z = 0;
    }

    GLFWwindow* window;
    if (setup_window(window, width, height)) {
        glfwTerminate();
        return -1;
    }

    GLuint prog_hdlr = set_shaders("../src/vertex.glsl", "../src/fragment.glsl");
    std::vector<GLfloat> vertices(3*3*ninttri, 0); // TODO get rid of these, it is ridiculous to have arrays that are not used
    std::vector<GLfloat>   colors(4*3*ninttri, 1);

    std::vector<int> tri2draw(ninttri, 0);
    {
        int cnt = 0;
        for (int t : facet_iter(mesh)) {
            if (tblayer[t]) continue;
            tri2draw[cnt++] = t;
        }
    }


    int ntri = mesh.nfacets();

    /*
    if (0) {
        for (int v : vert_iter(mesh)) {
            if (!vblayer[v] || fec.is_boundary_vert(v)) continue;
            //              if (rand()%20!=0) continue;
            std::vector<int> ring;
            {
                int cir = fec.v2c[v];
                do {
                    ring.push_back(fec.to(cir));
                    cir = fec.next_around_vertex(cir);
                } while (cir != fec.v2c[v]);
            }

            int off = mesh.create_facets(ring.size()-2);

            int v0 = ring[0];
            for (int lv=1; lv+1<ring.size(); lv++) {
                mesh.vert(off+lv-1, 0) = v0;
                mesh.vert(off+lv-1, 1) = ring[lv];
                mesh.vert(off+lv-1, 2) = ring[lv+1];
            }
        }
    }
    */
    opt = new Untangle2D(mesh);
opt->no_two_coverings();

if (0) {
    int off = mesh.create_facets(dectri.size()/3);
    for (int t : range(dectri.size()/3)) {
        for (int i : range(3)) {
            mesh.vert(off + t, i) = dectri[t*3+i];
        }
    }
    write_geogram("comp.geogram", mesh);
}






    to_kill_coverings.resize(mesh.nfacets());
    to_kill_blayer.resize(mesh.nfacets());

    assert(mesh.nfacets()==to_kill_coverings.size());

    for (int t : facet_iter(mesh)) {
        to_kill_coverings[t] = t>=ntri;
        to_kill_blayer[t] = (t>=ntri || tblayer[t]);
    }

// opt->rebuild_reference_geometry();

    opt->lock[30] = true;
    opt->lock[900] = true;


    // create the VAO that we use when drawing
    GLuint vao = 0;
    glGenVertexArrays(1, &vao); // allocate and assign a Vertex Array Object to our handle
    glBindVertexArray(vao);     // bind our Vertex Array Object as the current used object

    glEnableVertexAttribArray(0);
    GLuint vertexbuffer = 0;
    glGenBuffers(1, &vertexbuffer);              // allocate and assign one Vertex Buffer Object to our handle
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer); // bind our VBO as being the active buffer and storing vertex attributes (coordinates)
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*vertices.size(), vertices.data(), GL_STREAM_DRAW); // copy the vertex data to our buffer. The buffer contains sizeof(GLfloat) * 3 * nverts bytes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glEnableVertexAttribArray(1);
    GLuint colorbuffer = 0;
    glGenBuffers(1, &colorbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
    glBufferData(GL_ARRAY_BUFFER, colors.size()*sizeof(GLfloat), colors.data(), GL_STREAM_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);


    glViewport(0, 0, width, height);
    glClearColor(0.1f, 0.2f, 0.2f, 1.0f);
    glDisable(GL_DEPTH_TEST);
    glClearDepth(0);
    glDepthFunc(GL_GREATER); // accept fragment if it is closer to the camera than the former one
    glUseProgram(prog_hdlr); // specify the shaders to use



    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

//  auto start = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window)) {
//      { // 20ms sleep to reach 50 fps, do something useful instead of sleeping
//          auto end = std::chrono::steady_clock::now();
//          if (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() < 20) { 
//              std::this_thread::sleep_for(std::chrono::milliseconds(3));
//              continue;
//          }
//          start = end;
//      }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // wipe the screen buffers

        {
            glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
            float *ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
            assert(ptr);
            int cnt = 0;
            for (int i : range(ninttri)) {
                int t = tri2draw[i];
                for (int lv : range(3))
                    for (int d : range(2))
                        ptr[cnt*9+lv*3+d] = opt->X[mesh.vert(t, lv)*2+d];
                cnt++;
            }
            glUnmapBuffer(GL_ARRAY_BUFFER);     // release pointer to mapping buffer
        }

        opt->go();




        {
            glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
            float *ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
            assert(ptr);


            int cnt = 0;
            for (int i : range(ninttri)) {
                int t = tri2draw[i];
                for (int lv : range(3)) {
                    float d = opt->det[t]/2-.2;
                    heat_map_color(d, ptr[cnt*12+lv*4+0], ptr[cnt*12+lv*4+1], ptr[cnt*12+lv*4+2]);
                    ptr[cnt*12+lv*4+3] = .7;
                    if (opt->lock[mesh.vert(t, lv)]) {
                        ptr[cnt*12+lv*4+0] = ptr[cnt*12+lv*4+1] = ptr[cnt*12+lv*4+2] = ptr[cnt*12+lv*4+3] = 1;
                    }
                }
                cnt++;
            }

            glUnmapBuffer(GL_ARRAY_BUFFER);     // release pointer to mapping buffer

        }



        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

        glDrawArrays(GL_TRIANGLES, 0, ninttri*3);





        {
            glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
            float *ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
            assert(ptr);


            int cnt = 0;
            for (int i : range(ninttri)) {
                int t = tri2draw[i];
                for (int lv : range(3)) {
                    if (opt->lock[mesh.vert(t, lv)]) {
                        ptr[cnt*12+lv*4+0] = ptr[cnt*12+lv*4+1] = ptr[cnt*12+lv*4+2] = ptr[cnt*12+lv*4+3] = 1;
                    } else {
                        ptr[cnt*12+lv*4+0] = ptr[cnt*12+lv*4+1] = ptr[cnt*12+lv*4+2] = .3;
                        ptr[cnt*12+lv*4+3] = .3;
                    }
                }
                cnt++;
            }

            glUnmapBuffer(GL_ARRAY_BUFFER);     // release pointer to mapping buffer

        }

        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
        glDrawArrays(GL_TRIANGLES, 0, ninttri*3);


        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // de-allocate all the resources once they have outlived their purpose
    glUseProgram(0);
    glDeleteProgram(prog_hdlr); // note that the shader objects are automatically detached and deleted, since they were flagged for deletion by a previous call to glDeleteShader
    glDisableVertexAttribArray(0);
    glDeleteBuffers(1, &colorbuffer);
    glDeleteBuffers(1, &vertexbuffer);
    glDeleteVertexArrays(1, &vao);
    delete opt;

    glfwTerminate();
    return 0;
}

