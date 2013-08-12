#include "ErrorLog.hpp"

#include <cstdio>

#include <QtCore/QDateTime>

#include "Exscitech/Games/GameController.hpp"

namespace Exscitech
{
  const std::string ErrorLog::ms_errorLogLocation =
      GameController::acquire()->getExscitechDirectory ().append ("/ErrorLog.txt");
  const std::string ErrorLog::ms_errorLogHeader = "EXSCITECH ERROR LOG FILE ";
  const std::string ErrorLog::ms_errorLogClose = "ERROR LOG CLOSED";

  const std::string ErrorLog::ms_messageHeaders[3] =
    { "Error: ", "Warning: ", "" };

  bool ErrorLog::m_logfileOpen = false;
  std::ofstream ErrorLog::m_errorLogFile;

  void
  ErrorLog::openErrorLog ()
  {
    fprintf (stderr, "|%s|", ms_errorLogLocation.c_str ());
    // remove existing error file to prevent long use creating a huge amount of log info
    remove (ms_errorLogLocation.c_str ());

    // open new file to write errors to
    m_errorLogFile.open (ms_errorLogLocation.c_str ());

    if (!m_errorLogFile)
    {
      // cannot open error log file to record errors
      // future errors sent to the log will be redirected to stderr
      fprintf (stderr, "Error: problem opening error log file\n");
    }
    else
    {
      // record that error logging can occur
      m_logfileOpen = true;

      // print opening header
      m_errorLogFile << ms_errorLogHeader;
      std::string currentDateTime =
          QDateTime::currentDateTime ().toString ().toStdString ();
      m_errorLogFile << currentDateTime << std::endl;
    }
  }

  void
  ErrorLog::logWarning (const std::string& message, ...)
  {
    va_list arguments;
    va_start (arguments, message);
    writeMessage (message, arguments, Warning);
    va_end (arguments);
  }

  void
  ErrorLog::logError (const std::string& message, ...)
  {
    va_list arguments;
    va_start (arguments, message);
    writeMessage (message, arguments, Error);
    va_end (arguments);
  }

  void
  ErrorLog::logMessage (const std::string& message, ...)
  {
    va_list arguments;
    va_start (arguments, message);
    writeMessage (message, arguments, None);
    va_end (arguments);
  }

  void
  ErrorLog::writeMessage (const std::string& message, va_list arguments,
      int messageType)
  {
    std::string formattedMessage = QString ().vsprintf (message.c_str (),
        arguments).toStdString ();

    if (m_logfileOpen)
    {
      m_errorLogFile << ms_messageHeaders[messageType];
      m_errorLogFile << formattedMessage << std::endl;
    }
    else
    {
      fprintf (stderr, "%s%s\n", ms_messageHeaders[messageType].c_str (),
          formattedMessage.c_str ());
    }
  }

  void
  ErrorLog::closeErrorLog ()
  {
    if (m_logfileOpen)
    {
      m_errorLogFile << ms_errorLogClose << std::endl;
      m_errorLogFile.close ();
    }
  }
}
