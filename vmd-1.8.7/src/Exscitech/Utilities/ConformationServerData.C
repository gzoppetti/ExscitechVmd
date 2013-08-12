#include "Exscitech/Utilities/ConformationServerData.hpp"

namespace Exscitech
{

  ConformationServerData::ConformationServerData ()
  {

  }

  ConformationServerData*
  ConformationServerData::setDownloadUrl (const std::string& url)
  {
    m_url = url;
    return this;
  }

  ConformationServerData*
  ConformationServerData::setPdbFilePath (const std::string& filePath)
  {
    m_pdbFilePath = filePath;
    return this;
  }

  ConformationServerData*
  ConformationServerData::setThumbnailFilePath (const std::string& filePath)
  {
    m_thumbnailFilePath = filePath;
    return this;
  }

  ConformationServerData*
  ConformationServerData::setConformationId (const std::string& id)
  {
    m_conformationId = id;
    return this;
  }

  ConformationServerData*
  ConformationServerData::setLigandId (const std::string& id)
  {
    m_ligandId = id;
    return this;
  }

  std::string
  ConformationServerData::getUrl () const
  {
    return m_url;
  }

  std::string
  ConformationServerData::getPdbFilePath () const
  {
    return m_pdbFilePath;
  }

  std::string
  ConformationServerData::getThumbnailFilePath () const
  {
    return m_thumbnailFilePath;
  }

  std::string
  ConformationServerData::getConformationId () const
  {
    return m_conformationId;
  }

  std::string
  ConformationServerData::getLigandId () const
  {
    return m_ligandId;
  }
}
