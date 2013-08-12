#include <QtXml/QDomDocument>

namespace Exscitech
{
  class HostIdManager
  {
    friend class ServerCommunicationManager;

  private:

    HostIdManager ();

    void
    saveHostId (QDomDocument* boincReply);

  private:

    static std::string
    hostSaveLocation ();

  private:

    std::string m_hostId;
    bool m_idInitialized;

  };
}
