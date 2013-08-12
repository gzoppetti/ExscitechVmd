#ifndef WORKUNIT_ID_HPP
#define WORKUNIT_ID_HPP

#include <string>

#include <QtXml/QDomDocument>

namespace Exscitech
{

  class WorkunitId
  {
  public:

    // typically a unique package identifier
    std::string primaryName;
    // used in quit request for package with above identifier
    std::string quitName;

  public:

    WorkunitId ();

    bool
    isValid ();

    void
    fillIntoQuitRequest (QDomDocument* quitRequestResultSection);

    static WorkunitId
    initFromBoincResponse (QDomDocument* boincResponse);

  private:

    // name of tag in Boinc response that contains work unit name data
    static const std::string ms_workunitParentTagInResponse;
    // this is tag inside work unit data tag that contains work unit name
    static const std::string ms_workunitPrimaryNameTag;
    // this is tag inside work unit data tag that contains work unit "quit" name
    // also tag in quit request result section that should contain "quit" name
    static const std::string ms_workunitQuitNameTag;

  };
}
#endif
