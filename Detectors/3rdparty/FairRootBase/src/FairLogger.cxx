/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/*
 * FairLogger.cxx
 *
 *  Created on: Mar 03, 2011
 *      Author: f.uhlig
 */

#include "FairLogger.h"

#include <TString.h>   // for TString, operator==, etc
#include <TSystem.h>   // for gSystem, TSystem
#include <iostream>    // for cout, cerr
#include <stdio.h>     // for fclose, freopen, remove, etc

FairLogger* FairLogger::instance = nullptr;

const char FairLogger::endl = ' ';
const char FairLogger::flush = ' ';

FairLogger::FairLogger()
    : fLogFileName()
    , fLogFileSeverity("info")
    , fConsoleSeverity("info")
    , fBufferSize(1024)
    , fBufferSizeNeeded(-1)
    , fDynamicBuffer(fBufferSize)
    , fBufferPointer(&fDynamicBuffer[0])
{
    fair::Logger::OnFatal(LogFatalMessage);
}

FairLogger* FairLogger::GetLogger()
{
    if (!instance) {
        instance = new FairLogger();
    }

    return instance;
}

void FairLogger::Fatal(const char* file, const char* line, const char* func, const char* format, ...)
{
    va_list ap;
    va_start(ap, format);
    Log(fair::Severity::fatal, file, line, func, format, ap);
    va_end(ap);
}

void FairLogger::Error(const char* file, const char* line, const char* func, const char* format, ...)
{
    if (IsLogNeeded(fair::Severity::error)) {
        va_list ap;
        va_start(ap, format);
        Log(fair::Severity::error, file, line, func, format, ap);
        va_end(ap);
    }
}

void FairLogger::Warning(const char* file, const char* line, const char* func, const char* format, ...)
{
    if (IsLogNeeded(fair::Severity::warning)) {
        va_list ap;
        va_start(ap, format);
        Log(fair::Severity::warning, file, line, func, format, ap);
        va_end(ap);
    }
}

void FairLogger::Info(const char* file, const char* line, const char* func, const char* format, ...)
{
    if (IsLogNeeded(fair::Severity::info)) {
        va_list ap;
        va_start(ap, format);
        Log(fair::Severity::info, file, line, func, format, ap);
        va_end(ap);
    }
}

void FairLogger::Debug(const char* file, const char* line, const char* func, const char* format, ...)
{
    if (IsLogNeeded(fair::Severity::debug)) {
        va_list ap;
        va_start(ap, format);
        Log(fair::Severity::debug, file, line, func, format, ap);
        va_end(ap);
    }
}

void FairLogger::Debug1(const char* file, const char* line, const char* func, const char* format, ...)
{
    if (IsLogNeeded(fair::Severity::debug1)) {
        va_list ap;
        va_start(ap, format);
        Log(fair::Severity::debug1, file, line, func, format, ap);
        va_end(ap);
    }
}

void FairLogger::Debug2(const char* file, const char* line, const char* func, const char* format, ...)
{
    if (IsLogNeeded(fair::Severity::debug2)) {
        va_list ap;
        va_start(ap, format);
        Log(fair::Severity::debug2, file, line, func, format, ap);
        va_end(ap);
    }
}

void FairLogger::Debug3(const char* file, const char* line, const char* func, const char* format, ...)
{
    if (IsLogNeeded(fair::Severity::debug3)) {
        va_list ap;
        va_start(ap, format);
        Log(fair::Severity::debug3, file, line, func, format, ap);
        va_end(ap);
    }
}

void FairLogger::Debug4(const char* file, const char* line, const char* func, const char* format, ...)
{
    if (IsLogNeeded(fair::Severity::debug4)) {
        va_list ap;
        va_start(ap, format);
        Log(fair::Severity::debug4, file, line, func, format, ap);
        va_end(ap);
    }
}

void FairLogger::Log(fair::Severity severity,
                     const char* file,
                     const char* line,
                     const char* func,
                     const char* format,
                     va_list arglist)
{
    // If the format string together with the argument list was used once it can't
    // be used another time. Don't know why but if we do so the arguments are not
    // correct any longer and the program may crash (On Mac OSX). Even if it does
    // not crash it does not work as expected.
    // To vercome the problem the output is written to a buffer which then can be
    // used several times.

    while (true) {
        fBufferSizeNeeded = vsnprintf(fBufferPointer, fBufferSize, format, arglist);

        // Try to vsnprintf into our buffer.
        //    int needed = vsnprintf (buf, size, fmt, ap);
        // NB. C99 (which modern Linux and OS X follow) says vsnprintf
        // failure returns the length it would have needed.  But older
        // glibc and current Windows return -1 for failure, i.e., not
        // telling us how much was needed.

        if (fBufferSizeNeeded <= static_cast<int>(fBufferSize) && fBufferSizeNeeded >= 0) {
            // It fits fine so we're done.
            LOGD(severity, file, line, func) << std::string(fBufferPointer, static_cast<size_t>(fBufferSizeNeeded));
            break;
        }

        // vsnprintf reported that it wanted to write more characters
        // than we allotted.  So try again using a dynamic buffer.  This
        // doesn't happen very often if we chose our initial size well.
        fBufferSize = (fBufferSizeNeeded > 0) ? (fBufferSizeNeeded + 1) : (fBufferSize * 2);
        fDynamicBuffer.resize(fBufferSize);
    }
}

bool FairLogger::IsLogNeeded(fair::Severity severity) { return fair::Logger::Logging(severity); }

bool FairLogger::IsLogNeeded(FairLogLevel level) { return fair::Logger::Logging(ConvertLogLevelToString(level)); }

void FairLogger::LogFatalMessage()
{
    // Since Fatal indicates a fatal error it is maybe usefull to have
    // system information from the incident. Since the output on the screen
    // does not help the noraml user, the stderr is redirected into a special
    // core dump file for later usage.
    // Fatal also indicates a problem which is so severe that the process should
    // not go on, so the process is aborted.

    if (gSystem) {
        TString corefile = "core_dump_";
        int pid = gSystem->GetPid();
        corefile += pid;

        std::cerr << "For later analysis we write a core dump to " << corefile << std::endl;

        if (freopen(corefile, "w", stderr)) {
            gSystem->StackTrace();
            fclose(stderr);
        }
    }

    throw fair::FatalException("Fatal error occured. Exiting...");
}

ClassImp(FairLogger);
