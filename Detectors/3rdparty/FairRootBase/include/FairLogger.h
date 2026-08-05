/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/*
 * FairLogger.h
 *
 *  Created on: Mar 03, 2011
 *      Author: f.uhlig
 */

#ifndef BASE_FAIRLOGGER_H_
#define BASE_FAIRLOGGER_H_

#include <Rtypes.h>   // for bool, FairLogger::Class, etc
#include <fairlogger/Logger.h>
#include <stdarg.h>   // for va_list
#include <string>     // for operator<<
#include <vector>     // for vector

#define MESSAGE_ORIGIN __FILE__, CONVERTTOSTRING(__LINE__), __FUNCTION__

// backwards-compatibility with older FairLogger calls, TODO: delete me as soon as possible
enum FairLogLevel
{
    FATAL,
    ERROR,
    WARNING,
    INFO,
    DEBUG,
    DEBUG1,
    DEBUG2,
    DEBUG3,
    DEBUG4
};

// backwards-compatibility with older FairLogger calls, TODO: delete me as soon as possible
static const char* const LogLevelString[] =
    {"FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "DEBUG1", "DEBUG2", "DEBUG3", "DEBUG4"};

class FairLogger
{
  public:
    static FairLogger* GetLogger();

    void SetLogToScreen(bool enabled)
    {
        if (enabled) {
            fair::Logger::SetConsoleSeverity(fConsoleSeverity);
        } else {
            fair::Logger::SetConsoleSeverity(fair::Severity::fatal);
        }
    }

    void SetLogScreenLevel(const char* severity)
    {
        fConsoleSeverity = severity;
        fair::Logger::SetConsoleSeverity(fConsoleSeverity);
    }

    void SetLogToFile(bool enabled)
    {
        if (enabled) {
            fair::Logger::SetFileSeverity(fLogFileSeverity);
        } else {
            fair::Logger::SetFileSeverity(fair::Severity::fatal);
        }
    }

    void SetLogFileLevel(const char* severity)
    {
        fLogFileSeverity = severity;
        fair::Logger::SetFileSeverity(fLogFileSeverity);
    }

    void SetLogFileName(const std::string& name)
    {
        fLogFileName = name;
        OpenLogFile();
    }

    void SetColoredLog(bool enabled) { fair::Logger::SetConsoleColor(enabled); }

    void SetLogVerbosityLevel(const char* verbosity) { fair::Logger::SetVerbosity(verbosity); }

    bool IsLogNeeded(fair::Severity severity);
    bool IsLogNeeded(FairLogLevel level) __attribute__((
        deprecated("Use 'IsLogNeeded(fair::Severity severity)' or 'fair::Logger::Logging(const fair::Severity "
                   "severity)' or 'fair::Logger::Logging(const std::string& severityStr)'")));

    void Fatal(const char* file, const char* line, const char* func, const char* format, ...)
        __attribute__((deprecated("Use 'LOG(fatal) << content;' macro interface instead.")));
    void Error(const char* file, const char* line, const char* func, const char* format, ...)
        __attribute__((deprecated("Use 'LOG(error) << content;' macro interface instead.")));
    void Warning(const char* file, const char* line, const char* func, const char* format, ...)
        __attribute__((deprecated("Use 'LOG(warn) << content;' macro interface instead.")));
    void Info(const char* file, const char* line, const char* func, const char* format, ...)
        __attribute__((deprecated("Use 'LOG(info) << content;' macro interface instead.")));
    void Debug(const char* file, const char* line, const char* func, const char* format, ...)
        __attribute__((deprecated("Use 'LOG(debug) << content;' macro interface instead.")));
    void Debug1(const char* file, const char* line, const char* func, const char* format, ...)
        __attribute__((deprecated("Use 'LOG(debug1) << content;' macro interface instead.")));
    void Debug2(const char* file, const char* line, const char* func, const char* format, ...)
        __attribute__((deprecated("Use 'LOG(debug2) << content;' macro interface instead.")));
    void Debug3(const char* file, const char* line, const char* func, const char* format, ...)
        __attribute__((deprecated("Use 'LOG(debug3) << content;' macro interface instead.")));
    void Debug4(const char* file, const char* line, const char* func, const char* format, ...)
        __attribute__((deprecated("Use 'LOG(debug4) << content;' macro interface instead.")));

    static char const endl __attribute__((
        deprecated("Line break is now added automatically by the LOG macro, this variable only adds empty space.")));
    static char const flush __attribute__((
        deprecated("Data is now flushed automatically by the LOG macro, this variable only adds empty space.")));

    void SetScreenStreamToCerr(bool /* useCerr */) {}

  private:
    static FairLogger* instance;

    FairLogger();
    FairLogger(const FairLogger&);
    FairLogger operator=(const FairLogger&);

    ~FairLogger() {}

    void Log(fair::Severity level,
             const char* file,
             const char* line,
             const char*,
             const char* format,
             va_list arglist);

    const char* ConvertLogLevelToString(FairLogLevel level) const { return LogLevelString[level]; }

    static void LogFatalMessage();

    void OpenLogFile() { fair::Logger::InitFileSink(fLogFileSeverity, fLogFileName, false); }

    void CloseLogFile() { fair::Logger::RemoveFileSink(); }

    std::string fLogFileName;
    std::string fLogFileSeverity;
    std::string fConsoleSeverity;

    int fBufferSize;
    int fBufferSizeNeeded;
    std::vector<char> fDynamicBuffer;
    char* fBufferPointer;

    ClassDef(FairLogger, 4);
};

#define gLogger (FairLogger::GetLogger())

#endif   // BASE_FAIRLOGGER_H_
