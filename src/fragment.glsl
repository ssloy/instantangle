#version 330 core

// Output data
out vec3 color;

in vec3 varying_color;

void main() {
    color =  varying_color;//vec3(1,1,1);
}

