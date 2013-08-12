#version 130
in vec4 eye_position;
in vec2 g_texCoord;
in vec2 g_imposterTexCoord;

out vec4 g_fragColor;

uniform vec4 g_detail;
uniform mat4 g_projection;
uniform sampler2D g_sampler;

void main()
{
    //r^2 = x^2 + y^2 + z^2;
    float x = g_imposterTexCoord.x;
    float y = g_imposterTexCoord.y;
    float zz = 1.0 - x*x - y*y;

    if (zz <= 0.0)
       discard;

    float z = sqrt(zz);

    vec4 pos = eye_position;
    pos.xyz += z * g_detail.w;
    pos = g_projection * pos;

    gl_FragDepth = (pos.z / pos.w + 1.0) / 2.0;
    g_fragColor = vec4(g_detail.xyz * texture(g_sampler, g_texCoord).xyz, 1);
}