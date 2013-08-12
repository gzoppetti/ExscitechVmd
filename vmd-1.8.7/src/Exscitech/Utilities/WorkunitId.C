#include "WorkunitId.hpp"

namespace Exscitech
{
  const std::string WorkunitId::ms_workunitParentTagInResponse = "result";
  const std::string WorkunitId::ms_workunitPrimaryNameTag = "wu_name";
  const std::string WorkunitId::ms_workunitQuitNameTag = "name";

  WorkunitId::WorkunitId () :
      primaryName (""), quitName ("")
  {

  }

  bool
  WorkunitId::isValid ()
  {
    return (!primaryName.empty () && !quitName.empty ());
  }

  void
  WorkunitId::fillIntoQuitRequest (QDomDocument* quitRequestResultSection)
  {
    QDomElement packageIdentifierSection =
        quitRequestResultSection->elementsByTagName (
            QString (ms_workunitQuitNameTag.c_str ())).at (0).toElement ();
    packageIdentifierSection.firstChild ().setNodeValue (
        QString (quitName.c_str ()));
  }

  WorkunitId
  WorkunitId::initFromBoincResponse (QDomDocument* boincResponse)
  {
    WorkunitId workunitId;

    QDomElement workunitSection = boincResponse->elementsByTagName (
        QString (ms_workunitParentTagInResponse.c_str ())).at (0).toElement ();

    QDomElement workunitNameTag = workunitSection.elementsByTagName (
        QString (ms_workunitPrimaryNameTag.c_str ())).at (0).toElement ();
    workunitId.primaryName = workunitNameTag.text ().toStdString ();

    QDomElement workunitQuitNameTag = workunitSection.elementsByTagName (
        QString (ms_workunitQuitNameTag.c_str ())).at (0).toElement ();
    workunitId.quitName = workunitQuitNameTag.text ().toStdString ();

    return (workunitId);
  }
}
