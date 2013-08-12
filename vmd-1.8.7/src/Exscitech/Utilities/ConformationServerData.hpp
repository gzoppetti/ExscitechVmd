#ifndef CONFORMATIONSERVERDATA_HPP_
#define CONFORMATIONSERVERDATA_HPP_

#include <string>

namespace Exscitech
{
  class ConformationServerData
  {
  public:

    ConformationServerData ();

    ConformationServerData*
    setDownloadUrl (const std::string& url);

    ConformationServerData*
    setPdbFilePath (const std::string& filePath);

    ConformationServerData*
    setThumbnailFilePath (const std::string& filePath);

    ConformationServerData*
    setConformationId (const std::string& id);

    ConformationServerData*
    setLigandId (const std::string& id);

    std::string
    getUrl () const;

    std::string
    getPdbFilePath () const;

    std::string
    getThumbnailFilePath () const;

    std::string
    getConformationId () const;

    std::string
    getLigandId () const;

  private:

    std::string m_conformationId;
    std::string m_ligandId;
    std::string m_url;
    std::string m_pdbFilePath;
    std::string m_thumbnailFilePath;
  };
}
#endif
