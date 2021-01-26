#version 330 core

// Output data
out vec4 color;

in vec4 varying_color;
in vec2 UV;

uniform sampler2D diffuse;

void main() {
//    color =  varying_color;
    color = vec4(texture(diffuse, UV).xyz, .7) + varying_color;
}

