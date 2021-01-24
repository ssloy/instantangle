#version 330 core

// Output data
out vec4 color;

in vec4 varying_color;

void main() {
    color =  varying_color;
}

