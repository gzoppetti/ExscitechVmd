#include "HostIdManager.hpp"

#include "Exscitech/Utilities/ServerCommunicationManager.hpp"

#include <QtCore/QFile>
#include <QtCore/QTextStream>

namespace Exscitech {

HostIdManager::HostIdManager() :
		m_hostId("0"), m_idInitialized(false) {
	QString hostFileName = QString(hostSaveLocation().c_str());
	QFile hostFile(hostFileName);
	if (hostFile.exists()) {
		hostFile.open(QFile::ReadOnly);
		QTextStream hostReader(&hostFile);
		QString hostId = hostReader.readAll().trimmed();
		hostFile.close();

		bool isNumeric;
		hostId.toInt(&isNumeric);
		if (isNumeric) {
			m_hostId = hostId.toLocal8Bit().data();
			m_idInitialized = true;
		} else {
			QFile::remove(hostFileName);
		}
	} else {
		ErrorLog::logMessage("Host Id Manager: no host id data available");
	}
}

void HostIdManager::saveHostId(QDomDocument* boincReply) {
	static ServerCommunicationManager* instance =
			ServerCommunicationManager::acquire();

	QDomNodeList hostNodes = boincReply->elementsByTagName(
			QString(instance->m_boincHostIdTag.c_str()));
	if (hostNodes.length() > 0) {
		QDomElement hostIdSegment = hostNodes.at(0).toElement();
		m_hostId = hostIdSegment.text().toStdString();
		m_idInitialized = true;

		QFile hostWriter(QString(hostSaveLocation().c_str()));
		hostWriter.open(QFile::WriteOnly);
		hostWriter.write(m_hostId.c_str());
		hostWriter.close();

		ErrorLog::logMessage("Host Id Manager: saved host id %s",
				m_hostId.c_str());
	}

}

std::string HostIdManager::hostSaveLocation() {
	static GameController* instance = GameController::acquire();
	static std::string hostSaveLocation =
			instance->getExscitechDirectory().append("/").append(
					instance->m_serverDataFolderName).append("/hostid.data");
	return hostSaveLocation;
}
}
