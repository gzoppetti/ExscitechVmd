#ifndef SERVERCOMMUNICATIONMANAGER_HPP_
#define SERVERCOMMUNICATIONMANAGER_HPP_

#include <string>
#include <vector>
#include <QtXml/QDomDocument>

#include "Exscitech/Games/GameController.hpp"

namespace Exscitech
{
  class HostIdManager;

  struct UserData
  {
    UserData (const std::string& auth, const std::string& userId,
        const std::string& codeKey, const std::string& userCategory) :
        successful (true), authenticator (auth), userId (userId), userCategory (
            userCategory), codeSignKey (codeKey)
    {
    }

    UserData () :
        successful (false)
    {
    }

    // whether or not the login request was successful
    bool successful;
    // unique id for the user's account
    std::string authenticator;
    // unique id for the user (redundant??????)
    std::string userId;
    // identifies skill level of user, send in game requests so games can adjust data
    std::string userCategory;
    // for boinc requests
    std::string codeSignKey;
  };

  class ServerCommunicationManager
  {

  public:

    typedef std::pair<std::string, std::string> ReqData;

  public:

    static ServerCommunicationManager*
    acquire();

    void
    login (const std::string& username, const std::string& password);

    UserData
    getUserData ();

     void
    resetSequentialNumber ();

     QDomDocument*
    makeLearningGameRequest (const std::vector<ReqData>& requestData);

     QDomDocument*
    makeBoicRequest (const std::vector<ReqData>& fieldsToChange,
        QDomNode* additionalData = NULL);

     bool
    downloadFile (const std::string& url, const std::string& destinationName,
        const std::string& md5Checksum = "");

     bool
    requestJob (const std::string& fileToOutputTo);

     QDomDocument*
    makeStartJobSubmitRequest ();

     QDomDocument*
    submitJob (const std::string& sessionId, const std::string& proteinId,
        const std::string& ligandId, const std::string& conformationId,
        float rotationX, float rotationY, float rotationZ, float rotationPhi,
        int minTemp, int maxTemp, int totalTime, int heatPercent,
        int coolPercent);

  private:

    ServerCommunicationManager ();

     void
    fillBoincRequestEntries (QDomDocument& request);

     QDomDocument*
    createDefaultBoincRequest ();

     QDomDocument*
    createDefaultLearningGameRequest ();

     QDomDocument*
    createDocumentFromRequest (const std::string & title);

     QDomDocument*
    createStartJobSubmissionRequest ();

     void
    writeResponseAsXmlToTempFile (const std::string& sourceFilePath);

     void
    addEntryToElement (QDomDocument* doc, QDomElement* element,
        const std::string& tagName, const std::string& tagContents);

     std::string
    intToString(int value);

     std::string
    floatToString(float value);

  private:

    static HostIdManager ms_hostIdManager;
    static GameController* ms_gameControllerInstance;

    static UserData ms_currentUser;
    static int ms_sequentialNumber;

    static std::string ms_exscitechLoginUrl;
    static std::string ms_exscitechLearningGameUrl;
    static std::string ms_exscitechBoincRequestUrl;
    static std::string ms_exscitechRequestHandlerUrl;
    static std::string ms_tempReplyFile;

    static std::string ms_defaultBoincRequestFile;

    static std::string ms_authTagText;
    static std::string ms_userIdTagText;
    static std::string m_firstInnerTagText;
    static std::string ms_codeSignKeyTagText;
    static std::string ms_learnGameReqTagText;
    static std::string ms_startJobSubmitRequestTypeText;
    static std::string ms_submitJobRequestTypeText;

  public:

     const std::string m_boincHostIdTag;
     const std::string m_boincSeqNumTag;
     const std::string m_fileInfoTag;
     const std::string m_fileInfoNameTag;
     const std::string m_fileInfoUrlTag;
     const std::string m_fileInfoChecksumTag;
     const std::string m_fileInfoIgnoreTag1;
     const std::string m_fileInfoIgnoreTag2;
     const std::string m_gameEndTag;
     const std::string m_clientRequestTag;
     const std::string m_requestTypeTag;
     const std::string m_appIdTag;
     const std::string m_sessionIdTag;
     const std::string m_proteinIdTag;
     const std::string m_ligandIdTag;
     const std::string m_conformationIdTag;
     const std::string m_rotationXTag;
     const std::string m_rotationYTag;
     const std::string m_rotationZTag;
     const std::string m_rotationPhiTag;
     const std::string m_minTempTag;
     const std::string m_maxTempTag;
     const std::string m_totalTimeTag;
     const std::string m_heatPercentTag;
     const std::string m_coolPercentTag;

  };
}
#endif
