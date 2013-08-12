#include <vector>

#include "Exscitech/Utilities/LigandServerData.hpp"
#include "Exscitech/Utilities/ConformationServerData.hpp"

namespace Exscitech
{
  LigandServerData::LigandServerData ()
  {
  }

  LigandServerData::~LigandServerData ()
  {
    for (ConformationServerData* c : m_conformations)
    {
      delete c;
    }
  }

  LigandServerData*
  LigandServerData::setName (const std::string& name)
  {
    m_name = name;
    return this;
  }

  LigandServerData*
  LigandServerData::setId (const std::string& id)
  {
    m_id = id;
    return this;
  }

  LigandServerData*
  LigandServerData::setConformations (
      const std::vector<ConformationServerData*>& conformations)
  {
    m_conformations = conformations;
    return this;
  }

  std::string
  LigandServerData::getName () const
  {
    return m_name;
  }

  std::string
  LigandServerData::getId () const
  {
    return m_id;
  }

  std::vector<ConformationServerData*>&
  LigandServerData::getConformations ()
  {
    return m_conformations;
  }

}
