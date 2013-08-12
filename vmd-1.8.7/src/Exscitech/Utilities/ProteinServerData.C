#include "Exscitech/Utilities/ProteinServerData.hpp"

namespace Exscitech
{
  ProteinServerData::ProteinServerData ()
  {
  }

  ProteinServerData*
  ProteinServerData::setDownloadUrl (const std::string& url)
  {
    m_url = url;
    return this;
  }

  ProteinServerData*
  ProteinServerData::setName (const std::string& name)
  {
    m_name = name;
    return this;
  }

  ProteinServerData*
  ProteinServerData::setDisease(const std::string& disease)
  {
    m_disease = disease;
    return this;
  }

  ProteinServerData*
  ProteinServerData::setPdbFilePath (const std::string& filePath)
  {
    m_pdbFilePath = filePath;
    return this;
  }

  ProteinServerData*
  ProteinServerData::setThumbnailFilePath(const std::string& filePath)
  {
    m_thumbnailFilePath = filePath;
    return this;
  }
  ProteinServerData*
  ProteinServerData::setNotes (const std::string& notes)
  {
    m_notes = notes;
    return this;
  }

  ProteinServerData*
  ProteinServerData::setId (const std::string& id)
  {
    m_id = id;
    return this;
  }

  std::string
  ProteinServerData::getUrl () const
  {
    return m_url;
  }

  std::string
  ProteinServerData::getName () const
  {
    return m_name;
  }

  std::string
  ProteinServerData::getDisease () const
  {
    return m_disease;
  }

  std::string
  ProteinServerData::getPdbFilePath () const
  {
    return m_pdbFilePath;
  }

  std::string
  ProteinServerData::getThumbnailFilePath () const
  {
    return m_thumbnailFilePath;
  }

  std::string
  ProteinServerData::getNotes () const
  {
    return m_notes;
  }

  std::string
  ProteinServerData::getId () const
  {
    return m_id;
  }
}
