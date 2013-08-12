#ifndef ERROR_LOG_HPP
#define ERROR_LOG_HPP

#include <string>

#ifdef WIN32
#include <stdarg.h>
#endif

#include <iostream>
#include <fstream>

namespace Exscitech
{

  class ErrorLog
  {
  public:

    enum LogTag
    {
    	Error, Warning, None
    };

  public:

    static void
    openErrorLog ();

    static void
    logMessage (const std::string& message, ...);

    static void
    logWarning (const std::string& message, ...);

    static void
    logError (const std::string& message, ...);

    static void
    closeErrorLog ();

  private:

    static void
    writeMessage (const std::string& message, va_list arguments, int messageType);

    ErrorLog ();

  private:

    static std::ofstream m_errorLogFile;
    static bool m_logfileOpen;

  private:

    static const std::string ms_errorLogLocation;
    static const std::string ms_errorLogHeader;
    static const std::string ms_errorLogClose;

    static const std::string ms_messageHeaders[3];

  };
}
#endif
