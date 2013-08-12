#ifndef LIGANDSERVERDATA_HPP_
#define LIGANDSERVERDATA_HPP_

#include <string>

namespace Exscitech
{
  class ConformationServerData;
  class LigandServerData
  {
  public:

    LigandServerData ();

    ~LigandServerData();

    LigandServerData*
    setName (const std::string& name);

    LigandServerData*
    setConformations(const std::vector<ConformationServerData*>& conformations);

    LigandServerData*
    setId (const std::string& id);

    std::string
    getName () const;

    std::string
    getId () const;

    std::vector<ConformationServerData*>&
    getConformations();

  private:

    std::string m_name;
    std::string m_id;
    std::vector<ConformationServerData*> m_conformations;
  };
}
#endif
