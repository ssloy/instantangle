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

Triangles meshorig;
Triangles meshbrd;
int nvorig = -1;
int nvbrd = -1;

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
        for (int v : range(nvorig)) {
            for (int d=0; d<2; d++) {
                meshorig.points[v][d] = opt->X[v*2+d];
            }
        }
        for (int v : range(nvbrd)) {
            for (int d=0; d<2; d++) {
                meshbrd.points[v][d] = opt->X[v*2+d];
            }
        }
        write_geogram("opt.geogram", meshorig);
        write_geogram("optbrd.geogram", meshbrd);
        write_geogram("optfull.geogram", mesh);
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

int main(int argc, char** argv) {
    const double boxsize = 2.;
    const double shrink  = 1.3;
    read_by_extension("../potichat.obj", mesh);
    meshorig = mesh;
    nvorig = meshorig.nverts();
    int orignfacets= mesh.nfacets();


    SurfaceConnectivity fec(mesh);
    PointAttribute<int> bid(mesh.points);
    int nbord = 0;
    for (int v : vert_iter(mesh)) {
        bid[v] = -1;
        if (!fec.is_boundary_vert(v)) continue;
        bid[v] = nbord++;
    }
    int offv = mesh.points.create_points(nbord);
    for (int v : range(offv)) {
        if (!fec.is_boundary_vert(v)) continue;
        bid[offv+bid[v]] = -1;
        mesh.points[offv+bid[v]] = mesh.points[v];
    }


    for (int c : corner_iter(mesh)) {
        int opp = fec.opposite(c);
        if (opp>=0) continue;
        vec3 e = fec.geom(c);
        vec3 n = {e.y, -e.x, 0};
        n = n*.2;
        mesh.points[offv+bid[fec.from(c)]] = mesh.points[offv+bid[fec.from(c)]] + n;
        mesh.points[offv+bid[fec.to(c)]]     = mesh.points[offv+bid[fec.to(c)]] + n;
        int offf = mesh.create_facets(2);

        mesh.vert(offf+0, 0) = fec.to(c);
        mesh.vert(offf+0, 1) = fec.from(c);
        mesh.vert(offf+0, 2) = offv+bid[fec.from(c)];

        mesh.vert(offf+1, 0) = fec.to(c);
        mesh.vert(offf+1, 1) = offv+bid[fec.from(c)];
        mesh.vert(offf+1, 2) = offv+bid[fec.to(c)];

    }
    write_geogram("bl.geogram", mesh, {{ {"bid", bid.ptr}}, {}, {}  });

    meshbrd = mesh;
    nvbrd = meshbrd.nverts();

orignfacets= mesh.nfacets();

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
    std::vector<GLuint> indices(3*orignfacets, 0);
    std::vector<GLfloat> vertices(3*mesh.nverts(), 0);
    std::vector<GLfloat> colors(3*mesh.nverts(), 1);
    double angle = 1;

    for (int i=0; i<orignfacets; i++) {
        for (int j=0; j<3; j++)
            indices[i*3+j] = mesh.vert(i, j);
    }

    for (int i=0; i<mesh.nverts(); i++) {
        for (int k=0; k<3; k++) vertices[i*3 + k] = mesh.points[i][k];
        //      if (0==rand()%10) {
        //          colors[i*3+0] = 1.;
        //          colors[i*3+1] = 0.;
        //          colors[i*3+2] = 0.;
        //      } else {
        //          colors[i*3+0] = colors[i*3+1] = colors[i*3+2] = 1.;
        //      }
    }


    opt = new Untangle2D(mesh);
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
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);


    GLuint elementbuffer = 0;
    glGenBuffers(1, &elementbuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);


    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );


    glViewport(0, 0, width, height);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glClearDepth(0);
    glDepthFunc(GL_GREATER); // accept fragment if it is closer to the camera than the former one
    glUseProgram(prog_hdlr); // specify the shaders to use

    auto start = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window)) {
        { // 20ms sleep to reach 50 fps, do something useful instead of sleeping
            auto end = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() < 20) { 
//              std::this_thread::sleep_for(std::chrono::milliseconds(3));
//              continue;
            }
            start = end;
        }
        std::cerr << "start" << std::endl;
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        float *ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
        if (ptr) {
            // wobble vertex in and out along normal
            for (int i=0; i<mesh.nverts(); i++) {
                for (int d=0; d<2; d++) {
                    ptr[i*3+d] = opt->X[i*2+d];
                }
            }

            //          updateVertices(ptr, srcVertices, teapotNormals, vertexCount, (float)timer.getElapsedTime());
            glUnmapBuffer(GL_ARRAY_BUFFER);     // release pointer to mapping buffer
        }

opt->go();

        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        float *ptr2 = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
        if(ptr2) {

            for (int i=0; i<mesh.nverts(); i++) {
                if (opt->lock[i]) {
                    ptr2[i*3+0] = 1.;
                    ptr2[i*3+1] = 0.;
                    ptr2[i*3+2] = 0.;
                } else {
                    ptr2[i*3+0] = ptr2[i*3+1] = ptr2[i*3+2] = 1.;
                }
            }
            glUnmapBuffer(GL_ARRAY_BUFFER);     // release pointer to mapping buffer
        }




        angle += .01;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // wipe the screen buffers

        glDrawElements(
                GL_TRIANGLES,      // mode
                indices.size(),    // count
                GL_UNSIGNED_INT,   // type
                (void*)0           // element array buffer offset
                );


        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // de-allocate all the resources once they have outlived their purpose
    glUseProgram(0);
    glDeleteProgram(prog_hdlr); // note that the shader objects are automatically detached and deleted, since they were flagged for deletion by a previous call to glDeleteShader
    glDisableVertexAttribArray(0);
    glDeleteBuffers(1, &elementbuffer);
    glDeleteBuffers(1, &colorbuffer);
    glDeleteBuffers(1, &vertexbuffer);
    glDeleteVertexArrays(1, &vao);
    delete opt;

    glfwTerminate();
    return 0;
}

