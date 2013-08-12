#include <fstream>
#include <curl/curl.h>
#include <sstream>

#include <QtCore/QFile>
#include <QtCore/QCryptographicHash>
#include <QtXml/QDomDocument>

#include "ServerCommunicationManager.hpp"
#include "Exscitech/Games/GameController.hpp"
#include "Exscitech/Utilities/HostIdManager.hpp"

namespace Exscitech {

std::string ServerCommunicationManager::ms_exscitechLoginUrl =
		"http://docktest.gcl.cis.udel.edu/docktest2_cgi/authVMDGame";
std::string ServerCommunicationManager::ms_exscitechLearningGameUrl =
		"http://docktest.gcl.cis.udel.edu/testingcgi";
std::string ServerCommunicationManager::ms_exscitechBoincRequestUrl =
		"http://docktest.gcl.cis.udel.edu/docktest2_cgi/cgi";
std::string ServerCommunicationManager::ms_exscitechRequestHandlerUrl =
		"http://docktest.gcl.cis.udel.edu/exscitech/request_handler.php";
std::string ServerCommunicationManager::ms_tempReplyFile =
		GameController::acquire()->getExscitechDirectory().append("/").append(
				GameController::acquire()->m_serverDataFolderName).append(
				"/temp.txt");

// set to DefaultBoincRequest2 to make work. This way keeps new work
// unit from being generated each time
std::string ServerCommunicationManager::ms_defaultBoincRequestFile =
		"./vmd-1.8.7/ExscitechResources/ServerCommunication/DefaultBoincRequest.xml";

std::string ServerCommunicationManager::ms_authTagText = "authenticator";
std::string ServerCommunicationManager::ms_userIdTagText = "userid";
std::string ServerCommunicationManager::m_firstInnerTagText =
		"scheduler_request";
std::string ServerCommunicationManager::ms_codeSignKeyTagText = "code_sign_key";
std::string ServerCommunicationManager::ms_learnGameReqTagText =
		"client_request";
std::string ServerCommunicationManager::ms_startJobSubmitRequestTypeText =
		"start_job_submit_game";
std::string ServerCommunicationManager::ms_submitJobRequestTypeText =
		"submit_job";

// TODO: add the platform_name to these?

UserData ServerCommunicationManager::ms_currentUser;
int ServerCommunicationManager::ms_sequentialNumber = 1;

HostIdManager ServerCommunicationManager::ms_hostIdManager;
GameController* ServerCommunicationManager::ms_gameControllerInstance =
		GameController::acquire();
ServerCommunicationManager*
ServerCommunicationManager::acquire() {
	static ServerCommunicationManager* instance =
			new ServerCommunicationManager();
	return instance;
}

ServerCommunicationManager::ServerCommunicationManager() :
		m_boincHostIdTag("hostid"), m_boincSeqNumTag("rpc_seqno"), m_fileInfoTag(
				"file_info"), m_fileInfoNameTag("name"), m_fileInfoUrlTag(
				"url"), m_fileInfoChecksumTag("md5_cksum"), m_fileInfoIgnoreTag1(
				"executable"), m_fileInfoIgnoreTag2("upload_when_present"), m_gameEndTag(
				"end_of_game"), m_clientRequestTag("client_request"), m_requestTypeTag(
				"request_type"), m_appIdTag("app_id"), m_sessionIdTag(
				"session_id"), m_proteinIdTag("protein_id"), m_ligandIdTag(
				"ligand_id"), m_conformationIdTag("conformation_id"), m_rotationXTag(
				"rotation_x"), m_rotationYTag("rotation_y"), m_rotationZTag(
				"rotation_z"), m_rotationPhiTag("rotation_phi"), m_minTempTag(
				"min_temp"), m_maxTempTag("max_temp"), m_totalTimeTag(
				"total_time"), m_heatPercentTag("heat_percent"), m_coolPercentTag(
				"cool_percent") {

}

void ServerCommunicationManager::login(const std::string& username,
		const std::string& password) {
	CURL *curl;
	CURLcode res;

	struct curl_httppost *formpost = NULL;
	struct curl_httppost *lastptr = NULL;
	struct curl_slist *headerlist = NULL;
	static const char buf[] = "Expect:";

	curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "email",
			CURLFORM_COPYCONTENTS, username.c_str(), CURLFORM_END);

	curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "password",
			CURLFORM_COPYCONTENTS, password.c_str(), CURLFORM_END);

	curl = curl_easy_init();
	headerlist = curl_slist_append(headerlist, buf);

	if (curl) {
		curl_easy_setopt(curl, CURLOPT_URL, ms_exscitechLoginUrl.c_str ());

		curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);

		FILE* file = std::fopen(ms_tempReplyFile.c_str(), "w");

		curl_easy_setopt(curl, CURLOPT_WRITEDATA, file);
		res = curl_easy_perform(curl);
		curl_easy_cleanup(curl);
		curl_formfree(formpost);
		curl_slist_free_all(headerlist);

		// "correct" non-xml issue
		std::fseek(file, 2, SEEK_SET);
		std::fputs("--", file);
		std::fseek(file, 47, SEEK_SET);
		std::fputs("--", file);

		std::fclose(file);

		QDomDocument* doc = createDocumentFromRequest("Login Response");

		if (doc != NULL && doc->hasChildNodes()) {
			QDomElement element = doc->documentElement();

			if (!element.firstChildElement("AUTHENTICATOR").hasChildNodes()) {
				ErrorLog::logMessage(
						"Server Communication Utility: Login Failed, Invalid info.");
				return;
			}
			// trim authenticator, userId & category to remove newlines before & after
			std::string authenticator =
					element.firstChildElement("AUTHENTICATOR").firstChild().nodeValue().trimmed().toStdString();
			std::string userId =
					element.firstChildElement("USERID").firstChild().nodeValue().trimmed().toStdString();
			std::string userCategory =
					element.firstChildElement("CATEGORY").firstChild().nodeValue().trimmed().toStdString();
			// DO NOT trim code sign key - newlines at beginning and end are significant
			std::string codeKey =
					element.firstChildElement("KEY").firstChild().nodeValue().toStdString();

			delete doc;
			ms_currentUser = UserData(authenticator, userId, codeKey,
					userCategory);
		} else {
			ErrorLog::logError(
					"Server Communication Utility: Login failed: Document is NULL or empty.");
			return;
		}
	} else {
		ErrorLog::logError(
				"Server Communication Utility: Login Failed: Curl failed to initialize.");
		return;
	}
}

UserData ServerCommunicationManager::getUserData() {
	return ms_currentUser;
}

// resets sequential number counter
// all games that want to use boinc requests should call this when they begin
void ServerCommunicationManager::resetSequentialNumber() {
	ms_sequentialNumber = 1;
}

// turns the temporarily stored server response into an xml doc for the game to use
QDomDocument*
ServerCommunicationManager::createDocumentFromRequest(
		const std::string & title) {
	QFile response(QString(ms_tempReplyFile.c_str()));
	QDomDocument* responseDoc = new QDomDocument(QString(title.c_str()));
	QString errorText;
	int errLine, errCol;
	responseDoc->setContent(&response, true, &errorText, &errLine, &errCol);
	if (!errorText.isEmpty()) {
		ErrorLog::logWarning(
				"Server Communication Utility: parsing %s, %s line %d col %d",
				title.c_str(), errorText.toLocal8Bit().data(), errLine, errCol);
	}

	fprintf(stderr, "\n%s:\n-->%s<--\n", title.c_str (),
	responseDoc->toString ().toLocal8Bit ().data ());
	return (responseDoc);
	return (NULL);
}

// this method *MUST* be called when making a boinc request. Do not fill in the
// values manually, you need this method to get hostId && increment request number
void ServerCommunicationManager::fillBoincRequestEntries(
		QDomDocument& request) {
	QDomElement docElement = request.documentElement();
	docElement.firstChildElement(QString(ms_authTagText.c_str())).firstChild().setNodeValue(
			QString(ms_currentUser.authenticator.c_str()));
	docElement.firstChildElement(QString(ms_codeSignKeyTagText.c_str())).firstChild().setNodeValue(
			QString(ms_currentUser.codeSignKey.c_str()));
	docElement.firstChildElement(QString(m_boincSeqNumTag.c_str())).firstChild().setNodeValue(
			QString::number(ms_sequentialNumber++));
	docElement.firstChildElement(QString(m_boincHostIdTag.c_str())).firstChild().setNodeValue(
			QString(ms_hostIdManager.m_hostId.c_str()));
	// TODO: maybe ip_addr & domain_name
}

QDomDocument*
ServerCommunicationManager::createDefaultBoincRequest() {
	QFile boincRequestFile(QString(ms_defaultBoincRequestFile.c_str()));
	QDomDocument* boincRequest = new QDomDocument("Request");
	boincRequest->setContent(&boincRequestFile);
	return boincRequest;
}

QDomDocument*
ServerCommunicationManager::createDefaultLearningGameRequest() {
	QDomDocument* learningGameRequest = new QDomDocument();
	QDomElement mainElement = learningGameRequest->createElement(
			QString(ms_learnGameReqTagText.c_str()));
	learningGameRequest->appendChild(mainElement);

	// authenticator & userId go at beginning of every game request
	addEntryToElement(learningGameRequest, &mainElement, ms_authTagText,
			ms_currentUser.authenticator);
	addEntryToElement(learningGameRequest, &mainElement, ms_userIdTagText,
			ms_currentUser.userId);

	return (learningGameRequest);
}

void ServerCommunicationManager::addEntryToElement(QDomDocument* doc,
		QDomElement* element, const std::string& tagName,
		const std::string& tagContents) {
	QDomElement newElement = doc->createElement(QString(tagName.c_str()));

	QDomText contents = doc->createTextNode(QString(tagContents.c_str()));
	newElement.appendChild(contents);

	element->appendChild(newElement);
}

QDomDocument*
ServerCommunicationManager::makeLearningGameRequest(
		const std::vector<ReqData>& requestData) {
	fprintf(stderr,
	"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
	fprintf (stderr,
			"----------------------------------------------------------------------------\n");
	CURL *curlHandle;
	curlHandle = curl_easy_init ();

	if (!curlHandle)
	{
		ErrorLog::logError ("Server Communication Manager: could not init curl");
		return (NULL);
	}

	curl_easy_setopt(curlHandle, CURLOPT_URL,
			ms_exscitechLearningGameUrl.c_str ());

	QDomDocument* learningGameRequest = createDefaultLearningGameRequest ();
//    struct curl_httppost *postHead = NULL;
//    struct curl_httppost *postTail = NULL;
//
//    // authenticator & userId go at beginning of every game request
//    curl_formadd (&postHead, &postTail, CURLFORM_COPYNAME,
//        ms_authTagText.c_str (), CURLFORM_COPYCONTENTS,
//        ms_currentUser.authenticator.c_str (), CURLFORM_END);
//    fprintf (stderr, "|%s| --> |%s|\n", ms_authTagText.c_str (),
//        ms_currentUser.authenticator.c_str ());
//    curl_formadd (&postHead, &postTail, CURLFORM_COPYNAME,
//        ms_userIdTagText.c_str (), CURLFORM_COPYCONTENTS,
//        ms_currentUser.userId.c_str (), CURLFORM_END);
//    fprintf (stderr, "|%s| --> |%s|\n", ms_userIdauthTagText.c_str (),
//        ms_currentUser.userId.c_str ());

//    std::string str2 = "<client_request>";
//
//    str2.append ("\n\t<").append (ms_authTagText).append (">").append (
//        ms_currentUser.authenticator).append ("</").append (ms_authTagText).append (
//        ">");
//    str2.append ("\n\t<").append (ms_userIdTagText).append (">").append (
//        ms_currentUser.userId).append ("</").append (ms_userIdTagText).append (
//        ">");

// add additional game-specific tags
	QDomElement mainElement = learningGameRequest->documentElement ();
	for (size_t i = 0; i < requestData.size (); ++i)
	{
		ReqData requestArg = requestData[i];
		addEntryToElement (learningGameRequest, &mainElement, requestArg.first,
				requestArg.second);
//      curl_formadd (&postHead, &postTail, CURLFORM_COPYNAME,
//          requestArg.first.c_str (), CURLFORM_COPYCONTENTS,
//          requestArg.second.c_str (), CURLFORM_END);
		fprintf (stderr, "|%s| --> |%s|\n", requestArg.first.c_str (),
				requestArg.second.c_str ());
//      str2.append ("\n\t<").append (requestArg.first).append (">").append (
//          requestArg.second).append ("</").append (requestArg.first).append (
//          ">");
	}
//    str2.append ("\n</client_request>");

	curl_easy_setopt(curlHandle, CURLOPT_POST, true);

	fprintf (stderr, "\n____|%s|____\n",
			learningGameRequest->toString (4).trimmed ().toLocal8Bit ().data ());

	curl_easy_setopt(curlHandle, CURLOPT_COPYPOSTFIELDS,
			learningGameRequest->toString (4).trimmed ().toLocal8Bit ().data ());
//curl_easy_setopt(curlHandle, CURLOPT_POSTFIELDS, str.c_str());
//curl_easy_setopt(curlHandle, CURLOPT_HTTPPOST, postHead);

	FILE* tempFile = std::fopen (ms_tempReplyFile.c_str (), "w");
	curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, tempFile);

	char* errBuffer;
	curl_easy_setopt(curlHandle, CURLOPT_ERRORBUFFER, &errBuffer);

	CURLcode result = curl_easy_perform (curlHandle);
	if (result != CURLE_OK)
	{
		std::string errorText (errBuffer);
		ErrorLog::logError ("Server Communication Manager: curl problem %s",
				errorText.c_str ());
		return (NULL);
	}

	//curl_formfree (postHead);
	curl_easy_cleanup (curlHandle);

	std::fclose (tempFile);

	// put the response in a QDomDocument for game to use
	// TODO: any modifications to these files first??
	QDomDocument* responseDoc = createDocumentFromRequest (
			"Learning Game Response");

	fprintf (stderr,
			"----------------------------------------------------------------------------\n");
	fprintf (stderr,
			"vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n");

	return (responseDoc);

}

// TODO: have a separate makeQuitRequest method that handles adding the additional data and inserting the work unit name??? Determine when have more games.
QDomDocument*
ServerCommunicationManager::makeBoicRequest(
		const std::vector<ReqData>& fieldsToChange, QDomNode* additionalData) {
	// get the default request structure
	QDomDocument* boicRequest = createDefaultBoincRequest();

	// fill in non-game specific entries, like authenticator & code sign key
	fillBoincRequestEntries(*boicRequest);

	// change game specific fields to the provided values
	for (size_t i = 0; i < fieldsToChange.size(); ++i) {
		ReqData requestData = fieldsToChange[i];

		QString tagName(requestData.first.c_str());
		QDomNodeList matchingNodes = boicRequest->elementsByTagName(tagName);

		if (matchingNodes.isEmpty()) {
			ErrorLog::logError(
					"Server Communication Manager: no nodes match tag %s",
					requestData.first.c_str());
		} else {
			QString newContent(requestData.second.c_str());
			matchingNodes.item(0).childNodes().item(0).setNodeValue(newContent);
		}
	}

	// add node with additional data, if needed
	if (additionalData != NULL) {
		boicRequest->documentElement().appendChild(*additionalData);
	}

	// send request using curl
	CURL* curlHandle = curl_easy_init();
	if (!curlHandle) {
		ErrorLog::logError("Server Communication Manager: could not init curl");
		return (NULL);
	}

	curl_easy_setopt(curlHandle, CURLOPT_URL,
			ms_exscitechBoincRequestUrl.c_str ());

	curl_easy_setopt(curlHandle, CURLOPT_POST, true);

	curl_easy_setopt(curlHandle, CURLOPT_COPYPOSTFIELDS,
			boicRequest->toString (4).trimmed ().toLocal8Bit ().data ());

	fprintf(stderr, ">>>|%s|<<<\n",
	boicRequest->toString (4).toLocal8Bit ().data ());

	// experimental stuff here - does not appear to be necessary
	// curl_easy_setopt(curlHandle, CURLOPT_POSTFIELDSIZE, textRequest.length());
	// struct curl_slist *slist = curl_slist_append(NULL, "Content-Type: text/xml; charset=utf-8");
	// curl_easy_setopt(curlHandle, CURLOPT_HTTPHEADER, slist);
	// end experimental stuff

	FILE* file = std::fopen(ms_tempReplyFile.c_str(), "w");
	curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, file);
	CURLcode curlCode = curl_easy_perform(curlHandle);
	std::fclose(file);

	long httpCode = 0;
	curl_easy_getinfo(curlHandle, CURLINFO_RESPONSE_CODE, &httpCode);

	if (httpCode != 200 || curlCode != CURLE_OK) {
		ErrorLog::logError(
				"Server Communication Utility: boinc request failed, curl error code %d http code %li\n",
				curlCode, httpCode);
		return (NULL);
	}

	// create response to return
	QDomDocument* responseDoc = createDocumentFromRequest("Boinc Response");

	// store host id if necessary
	if (!ms_hostIdManager.m_idInitialized) {
		ms_hostIdManager.saveHostId(responseDoc);
	}

	// return boinc response
	return (responseDoc);

}

bool ServerCommunicationManager::downloadFile(const std::string& url,
		const std::string& destinationName, const std::string& md5Checksum) {
	CURL* curlHandle = curl_easy_init();
	if (!curlHandle) {
		ErrorLog::logError("Server Communication Manager: could not init curl");
		return (false);
	}

	curl_easy_setopt(curlHandle, CURLOPT_URL, url.c_str ());

	FILE* file = std::fopen(destinationName.c_str(), "w");
	curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, file);
	CURLcode curlCode = curl_easy_perform(curlHandle);
	std::fclose(file);

	if (curlCode != CURLE_OK) {
		ErrorLog::logError(
				"Server Communication Utility: boinc request failed, curl error code %d\n",
				curlCode);
		return (false);
	}

	// check checksum if provided
	QString origianlChecksumText(md5Checksum.c_str());
	if (!origianlChecksumText.isEmpty()) {

		QCryptographicHash checksumCalculator(QCryptographicHash::Md5);
		QFile fileToCheck(QString(destinationName.c_str()));
		fileToCheck.open(QIODevice::ReadOnly);
		QByteArray fileData = fileToCheck.readAll();
		checksumCalculator.addData(fileData);
		QByteArray fileChecksum = checksumCalculator.result();
		QString checksumText = QString(fileChecksum.toHex());
		if (checksumText != origianlChecksumText) {
			ErrorLog::logError(
					"Server Communication Utility: checksum %s for file %s does not match expected value of %s\n",
					checksumText.toLocal8Bit().data(), destinationName.c_str(),
					md5Checksum.c_str());
			return (false);
		}
	}

	// file downloaded successfully
	return (true);
}

/***********************************************************************************/

bool ServerCommunicationManager::requestJob(const std::string& fileToOutputTo) {
	if (!ms_currentUser.successful) {
		fprintf(stderr, "RequestJob Failed: Current user is not valid.\n");
		return false;
	}

	CURL *curl = curl_easy_init ();
	if (curl)
	{

		curl_easy_setopt(curl, CURLOPT_URL,
				"http://docktest.gcl.cis.udel.edu/docking_cgi/cgi");

		curl_easy_setopt(curl, CURLOPT_POST, true);

		QFile requestXml (
				"./vmd-1.8.7/ExscitechResources/ServerCommunication/JobRequestSkeleton.xml");
      QDomDocument doc ("Request");
      doc.setContent (&requestXml);
      fillBoincRequestEntries (doc);

      std::string requestString = doc.toString ().toStdString ();
      fprintf (stderr, "\n\n%s\n\n", requestString.c_str ());
      //requestString += "\n";

      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, requestString.c_str ());

      FILE* file = std::fopen (fileToOutputTo.c_str (), "w");
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, file);
      CURLcode curlCode = curl_easy_perform (curl);

      std::fclose (file);

      long http_code = 0;
      curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

      if (http_code == 200 && curlCode != CURLE_ABORTED_BY_CALLBACK)
      {
        return true;
      }
      else
      {
        fprintf (stderr,
            "RequestJob Failed: Communication with the Docking CGI has failed.\n");
        fprintf (stderr, "Error: %li\n", http_code);
        return false;
      }
    }
    else
    {
      fprintf (stderr, "RequestJob Failed: Curl failed to initialize.\n");
      return false;
    }

  }

QDomDocument*
ServerCommunicationManager::createStartJobSubmissionRequest() {
	QDomDocument* request = new QDomDocument();
	QDomElement mainElement = request->createElement(
			QString(m_clientRequestTag.c_str()));
	request->appendChild(mainElement);
	addEntryToElement(request, &mainElement, m_requestTypeTag,
			ms_startJobSubmitRequestTypeText);
	addEntryToElement(request, &mainElement, ms_authTagText,
			ms_currentUser.authenticator);
	addEntryToElement(request, &mainElement, m_appIdTag, "");
	return request;
}

void ServerCommunicationManager::writeResponseAsXmlToTempFile(
		const std::string& sourceFilePath) {
	std::vector<std::string> lines;
	std::vector<std::string>::const_iterator i;
	char line[1000];

	FILE* responseTempFile = std::fopen(sourceFilePath.c_str(), "r");
	while (fgets(line, sizeof(line), responseTempFile))
		lines.push_back(line);
	fclose(responseTempFile);

	FILE* xmlTempFile = fopen(ms_tempReplyFile.c_str(), "w");
	int lineNumber = 0;
	for (i = lines.begin(); i != lines.end(); ++i) {
		// Remove the first two lines
		if (lineNumber > 2)
			fputs((*i).c_str(), xmlTempFile);
		lineNumber++;
	}
	fclose(xmlTempFile);
}

std::string ServerCommunicationManager::intToString(int value) {
	std::stringstream ss;
	ss << value;
	return ss.str();
}

std::string ServerCommunicationManager::floatToString(float value) {
	std::stringstream ss;
	ss << value;
	return ss.str();
}

QDomDocument*
ServerCommunicationManager::makeStartJobSubmitRequest() {
	// Init CURL.  If this fails, no use in doing anything else.
	CURL *curlHandle;
	curlHandle = curl_easy_init();

	if (!curlHandle) {
		ErrorLog::logError("Server Communication Manager: could not init curl");
		return (NULL);
	}

	// Build the request XML
	QDomDocument* request = createStartJobSubmissionRequest();

	// Begin setting up CURL
	curl_easy_setopt(curlHandle, CURLOPT_URL,
			ms_exscitechRequestHandlerUrl.c_str ());
	curl_easy_setopt(curlHandle, CURLOPT_POST, true);

	// Set the CURL post data and header
	QString requestString = request->toString(1).trimmed();
	curl_easy_setopt(curlHandle, CURLOPT_COPYPOSTFIELDS,
			requestString.toLocal8Bit ().data ());
	struct curl_slist *slist = curl_slist_append(NULL,
			"Content-Type: text/xml; charset=utf-8");
	curl_easy_setopt(curlHandle, CURLOPT_HTTPHEADER, slist);

	// The server sends an invalid header for an xml that we must remove.  To do this, we store the response in a second temp file, and then
	// write the correct xml into the file specified by the static data member.
	std::string tempResponseFilePath =
			ms_gameControllerInstance->getExscitechDirectory().append("/").append(
					ms_gameControllerInstance->m_serverDataFolderName).append(
					"/temp2.txt");

	// Opens the second temp file
	FILE* tempFile = std::fopen(tempResponseFilePath.c_str(), "w");

	// Sets up CURL to print the resposne into that file
	curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, tempFile);

	char* errBuffer;
	curl_easy_setopt(curlHandle, CURLOPT_ERRORBUFFER, &errBuffer);

	CURLcode result = curl_easy_perform(curlHandle);
	if (result != CURLE_OK) {
		std::string errorText(errBuffer);
		ErrorLog::logError("Server Communication Manager: curl problem %s",
				errorText.c_str());
		return (NULL);
	}

	//curl_formfree (slist);
	curl_easy_cleanup(curlHandle);
	std::fclose(tempFile);

	// The second temp file is now populated, time to write the correct XML to the first temp file
	writeResponseAsXmlToTempFile(tempResponseFilePath);

	// put the response in a QDomDocument for game to use
	// TODO: any modifications to these files first??
	QDomDocument* responseDoc = createDocumentFromRequest(
			"Job Submission Response");
	return (responseDoc);

}

QDomDocument*
ServerCommunicationManager::submitJob(const std::string& sessionId,
		const std::string& proteinId, const std::string& ligandId,
		const std::string& conformationId, float rotationX, float rotationY,
		float rotationZ, float rotationPhi, int minTemp, int maxTemp,
		int totalTime, int heatPercent, int coolPercent) {
	// Init CURL.  If this fails, no use in doing anything else.
	CURL *curlHandle;
	curlHandle = curl_easy_init();

	if (!curlHandle) {
		ErrorLog::logError("Server Communication Manager: could not init curl");
		return (NULL);
	}

	// Build the request XML
	QDomDocument* request = new QDomDocument();
	QDomElement mainElement = request->createElement(
			QString(m_clientRequestTag.c_str()));
	request->appendChild(mainElement);
	addEntryToElement(request, &mainElement, m_requestTypeTag,
			ms_submitJobRequestTypeText);
	addEntryToElement(request, &mainElement, ms_authTagText,
			ms_currentUser.authenticator);
	addEntryToElement(request, &mainElement, m_sessionIdTag, sessionId);
	addEntryToElement(request, &mainElement, m_appIdTag, "");
	addEntryToElement(request, &mainElement, m_proteinIdTag, proteinId);
	addEntryToElement(request, &mainElement, m_ligandIdTag, ligandId);
	addEntryToElement(request, &mainElement, m_conformationIdTag,
			conformationId);
	addEntryToElement(request, &mainElement, m_rotationXTag,
			floatToString(rotationX));
	addEntryToElement(request, &mainElement, m_rotationYTag,
			floatToString(rotationY));
	addEntryToElement(request, &mainElement, m_rotationZTag,
			floatToString(rotationZ));
	addEntryToElement(request, &mainElement, m_rotationPhiTag,
			floatToString(rotationPhi));
	addEntryToElement(request, &mainElement, m_minTempTag,
			intToString(minTemp));
	addEntryToElement(request, &mainElement, m_maxTempTag,
			intToString(maxTemp));
	addEntryToElement(request, &mainElement, m_totalTimeTag,
			intToString(totalTime));
	addEntryToElement(request, &mainElement, m_heatPercentTag,
			intToString(heatPercent));
	addEntryToElement(request, &mainElement, m_coolPercentTag,
			intToString(coolPercent));

	// Begin setting up CURL
	curl_easy_setopt(curlHandle, CURLOPT_URL,
			ms_exscitechRequestHandlerUrl.c_str ());
	curl_easy_setopt(curlHandle, CURLOPT_POST, true);

	// Set the CURL post data and header
	QString requestString = request->toString(1).trimmed();
	curl_easy_setopt(curlHandle, CURLOPT_COPYPOSTFIELDS,
			requestString.toLocal8Bit ().data ());
	struct curl_slist *slist = curl_slist_append(NULL,
			"Content-Type: text/xml; charset=utf-8");
	curl_easy_setopt(curlHandle, CURLOPT_HTTPHEADER, slist);

	// The server sends an invalid header for an xml that we must remove.  To do this, we store the response in a second temp file, and then
	// write the correct xml into the file specified by the static data member.
	std::string tempResponseFilePath =
			ms_gameControllerInstance->getExscitechDirectory().append("/").append(
					ms_gameControllerInstance->m_serverDataFolderName).append(
					"/temp2.txt");

	// Opens the second temp file
	FILE* tempFile = std::fopen(tempResponseFilePath.c_str(), "w");

	// Sets up CURL to print the resposne into that file
	curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, tempFile);

	char* errBuffer;
	curl_easy_setopt(curlHandle, CURLOPT_ERRORBUFFER, &errBuffer);

	CURLcode result = curl_easy_perform(curlHandle);
	if (result != CURLE_OK) {
		std::string errorText(errBuffer);
		ErrorLog::logError("Server Communication Manager: curl problem %s",
				errorText.c_str());
		return (NULL);
	}

	//curl_formfree (slist);
	curl_easy_cleanup(curlHandle);
	std::fclose(tempFile);

	// The second temp file is now populated, time to write the correct XML to the first temp file
	writeResponseAsXmlToTempFile(tempResponseFilePath);

	// put the response in a QDomDocument for game to use
	// TODO: any modifications to these files first??
	QDomDocument* responseDoc = createDocumentFromRequest("Job Submission");
	return (responseDoc);
}

}
