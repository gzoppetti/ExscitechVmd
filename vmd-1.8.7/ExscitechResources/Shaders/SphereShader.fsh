in vec4 eye_position;
in vec2 g_texCoord;

uniform vec4 g_detail;
uniform mat4 g_projection;

void main()
{
    //r^2 = x^2 + y^2 + z^2;
    float x = g_texCoord.x;
    float y = g_texCoord.y;
    float zz = 1.0 - x*x - y*y;

    if (zz <= 0.0)
       discard;

    float z = sqrt(zz);

    vec3 normal = vec3(x, y, z);

    // Lighting
    float diffuse_value = max(dot(normal, vec3(0, 0.3, 1)), 0.0);

    vec4 pos = eye_position;
    pos.xyz += z * g_detail.w;
    pos = g_projection * pos;

    gl_FragDepth = (pos.z / pos.w + 1.0) / 2.0;
    gl_FragColor = vec4(g_detail.xyz* diffuse_value, 1);
}