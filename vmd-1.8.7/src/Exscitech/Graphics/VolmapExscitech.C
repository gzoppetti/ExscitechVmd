#include "VolmapExscitech.hpp"

namespace Exscitech
{
  std::vector<Vector3f>* VolmapExscitech::ms_cachedVertices = NULL;
  std::vector<Vector3f>* VolmapExscitech::ms_cachedNormals = NULL;
  std::vector<int>* VolmapExscitech::ms_cachedIndices = NULL;

  VolmapExscitech::VolmapExscitech (const std::string& moleculeFile) :
      m_program ("./vmd-1.8.7/ExscitechResources/Shaders/GeneralShader.vsh",
          "./vmd-1.8.7/ExscitechResources/Shaders/GeneralShader.fsh"), m_lightUniformManager (
          m_program.getId (), 3)
  {
    FileSpec spec;
    static GameController* instance = GameController::acquire();
    int molId = instance->m_vmdApp->molecule_load (-1,
        moleculeFile.c_str (), NULL, &spec);

    Molecule* molecule = instance->m_vmdApp->moleculeList->mol_from_id (
        molId);

    instance->m_vmdApp->molecule_make_top (molId);

    molecule->set_cent_trans (0, 0, 0);

    fprintf (stderr, "Loading Volmap");
    instance->m_vmdApp->uiText->read_from_file (
        "./vmd-1.8.7/ExscitechResources/exscitechVolMap.tcl");

    DrawMolItem* molItem = molecule->component (1);

    molecule->set_cent_trans (0, 0, 0);
    molecule->set_glob_trans (0, 0, 0);

    const VolumetricData* data = molecule->get_volume_data (0);

    ms_cachedVertices = &m_vertices;
    ms_cachedNormals = &m_normals;
    ms_cachedIndices = &m_indices;
    molItem->generateTriMeshCommand (data);

    ms_cachedVertices = NULL;
    ms_cachedNormals = NULL;
    ms_cachedIndices = NULL;
    fprintf (stderr, "Vector sizes: %lu %lu %lu", m_vertices.size (),
        m_normals.size (), m_indices.size ());

    instance->m_vmdApp->scene->root.remove_child (molecule);
    instance->m_vmdApp->molecule_delete (molId);
  }

  void
  VolmapExscitech::draw (Camera* camera)
  {

    m_program.enable ();
    m_lightUniformManager.setUniforms (camera, &m_material);
    m_program.setUniform (m_program.getUniformLocation ("g_world"),
        getTransform4x4 ());
    m_program.setUniform (m_program.getUniformLocation ("g_view"),
        camera->getView ());
    m_program.setUniform (m_program.getUniformLocation ("g_projection"),
        camera->getProjection ());

    m_program.setAttribPointer (m_program.getAttribLocation ("g_position"), 3,
        GL_FLOAT, false, 0, 0, &m_vertices[0]);
    m_program.setAttribPointer (m_program.getAttribLocation ("g_normal"), 3,
        GL_FLOAT, false, 0, 0, &m_normals[0]);

    glDrawElements (GL_TRIANGLES, m_indices.size (), GL_UNSIGNED_INT,
        &m_indices[0]);

    m_program.disableAttribute ("g_position");
    m_program.disableAttribute ("g_normal");
    m_program.disable ();

  }

  void
  VolmapExscitech::printArrays ()
  {
    fprintf (stderr, "Vertices: %lu\n", m_vertices.size ());
    fprintf (stderr, "Normals: %lu\n", m_normals.size ());
    fprintf (stderr, "Indices: %lu\n", m_indices.size ());
    for (unsigned int i = 0; i < m_vertices.size (); i++)
    {
      fprintf (stderr, "V: %f %f %f\n", m_vertices[i].x, m_vertices[i].y,
          m_vertices[i].z);
    }
    for (unsigned int i = 0; i < m_normals.size (); i++)
    {
      fprintf (stderr, "V: %f %f %f\n", m_normals[i].x, m_normals[i].y,
          m_normals[i].z);
    }
    for (unsigned int i = 0; i < m_indices.size (); i += 3)
    {
      fprintf (stderr, "P: %d %d %d\n", m_indices[i], m_indices[i + 1],
          m_indices[i + 2]);
    }
  }

  void
  VolmapExscitech::cache (float* cvn, int* indices, int numVerts,
      int numIndices)
  {
    // Here's the deal - Vertices are wound wrong, giving bullet the wrong normals.
    if (ms_cachedVertices == NULL)
    {
      fprintf (stderr, "Cached Vertices not set\n");
      return;
    }

    if (ms_cachedNormals == NULL)
    {
      fprintf (stderr, "Cached Normals not set\n");
      return;
    }

    if (ms_cachedIndices == NULL)
    {
      fprintf (stderr, "Cached Indices not set\n");
      return;
    }

    int stride = 10;
    for (int i = 0; i < numVerts; ++i)
    {
      ms_cachedNormals->push_back (Vector3f (&cvn[4 + i * stride]));
      ms_cachedNormals->back ().negate ();
      ms_cachedNormals->back ().normalize ();
      ms_cachedVertices->push_back (Vector3f (&cvn[7 + i * stride]));
    }

    for (int i = 0; i < numIndices; i += 3)
    {
      ms_cachedIndices->push_back (indices[i + 2]);
      ms_cachedIndices->push_back (indices[i + 1]);
      ms_cachedIndices->push_back (indices[i]);
    }
  }

  void
  VolmapExscitech::getData (float*& vertices, int*& indices, int& numVertices,
      int& numIndices)
  {
    vertices = &m_vertices[0].coords[0];
    indices = &m_indices[0];
    numVertices = m_vertices.size ();
    numIndices = m_indices.size ();
  }

  std::vector<VolmapPiece*>
  VolmapExscitech::split (unsigned int numPieces)
  {
    unsigned int numIndices = m_indices.size ();
    unsigned int verticesPerPiece = numIndices / numPieces;
    verticesPerPiece += 3 - verticesPerPiece % 3;

    std::vector<VolmapPiece*> pieces;
    for (unsigned int i = 0; i < numPieces; ++i)
    {
      std::vector<Vector3f> vertices;
      std::vector<Vector3f> normals;

      unsigned int start = i * verticesPerPiece;
      unsigned int end = start + verticesPerPiece;
      if (end > numIndices)
        end = numIndices;

      for (unsigned int j = start; j < end; ++j)
      {
        vertices.push_back (m_vertices[m_indices[j]]);
        normals.push_back (m_normals[m_indices[j]]);
      }

      VolmapPiece* piece = new VolmapPiece (vertices, normals);

      pieces.push_back (piece);
    }
    return pieces;
  }

}
