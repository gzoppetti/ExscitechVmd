#ifndef KeynoteDemo_FullQuad_hpp
#define KeynoteDemo_FullQuad_hpp


#include <vector>

#include "Exscitech/Graphics/Shaders/ShaderProgram.hpp"
#include "Exscitech/Math/Vector3.hpp"

namespace Exscitech
{
class FullQuad
{
public:
    
    FullQuad();
    
    virtual
    ~FullQuad();
    
    void
    setTexture(unsigned int texId);
    
    virtual void
    draw(int texUnit);
    
    
private:
    
    static const float VERTS[18];
    
private:
    
    unsigned int m_vboId;
    unsigned int m_texId;
    
    ShaderProgram m_program;
};
}
#endif
