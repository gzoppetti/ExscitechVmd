#ifndef PROTEINSERVERDATA_HPP_
#define PROTEINSERVERDATA_HPP_

#include <string>

namespace Exscitech
{
  class ProteinServerData
  {
  public:

    ProteinServerData ();

    ProteinServerData*
    setDownloadUrl (const std::string& url);

    ProteinServerData*
    setName (const std::string& name);

    ProteinServerData*
    setDisease(const std::string& disease);

    ProteinServerData*
    setPdbFilePath (const std::string& filePath);

    ProteinServerData*
    setThumbnailFilePath(const std::string& filePath);

    ProteinServerData*
    setNotes (const std::string& notes);

    ProteinServerData*
    setId (const std::string& id);

    std::string
    getUrl () const;

    std::string
    getName () const;

    std::string
    getDisease() const;

    std::string
    getPdbFilePath () const;

    std::string
    getThumbnailFilePath() const;

    std::string
    getNotes () const;

    std::string
    getId () const;

  private:

    std::string m_url;
    std::string m_name;
    std::string m_disease;
    std::string m_pdbFilePath;
    std::string m_thumbnailFilePath;
    std::string m_notes;
    std::string m_id;

  };
}
#endif
