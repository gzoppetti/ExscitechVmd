in vec4 g_point;
in vec3 g_color;

in vec4 eye_position;
in vec2 g_texCoord;
in vec3 g_fragmentColor;
in float zDelta;
in float g_radius;

uniform mat4 g_projection;

void main()
{
    // Lighting
    float normalCoef = (g_texCoord.y + 1.0) / 2.0;
	float diffuse_value = sin(normalCoef * 3.14 / 2.0);
    vec4 pos = eye_position;
    
    pos.z += g_texCoord.x * zDelta;
    //pos.z += g_radius * sin(normalCoef * 3.14);
    pos = g_projection * pos;

    gl_FragDepth = (pos.z / pos.w + 1.0) / 2.0;
    gl_FragColor = vec4(g_fragmentColor, 1);//vec4(g_fragmentColor* diffuse_value, 1);
}