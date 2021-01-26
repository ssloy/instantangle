#version 330 core

// Input vertex data, different for all executions of this shader
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec4 vertexcolor;
layout(location = 2) in vec2 vertexUV;


out vec4 varying_color;
out vec2 UV;

void main() {
    gl_Position = vec4(vertexPosition_modelspace,1);
    varying_color = vertexcolor;
    UV = vertexUV;  // UV of the vertex. No special space for this one.
}

