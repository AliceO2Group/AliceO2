// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/RuntimeError.h"

#include <cstdio>
#include <atomic>
#include <cstdarg>
#include <execinfo.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>

namespace o2::framework
{

namespace
{
static RuntimeError gError[RuntimeError::MAX_RUNTIME_ERRORS];
static std::atomic<bool> gErrorBooking[RuntimeError::MAX_RUNTIME_ERRORS];

bool canDumpBacktrace()
{
#ifdef DPL_ENABLE_BACKTRACE
  return true;
#else
  return false;
#endif
}
} // namespace

void clean_all_runtime_errors()
{
  for (size_t i = 0; i < RuntimeError::MAX_RUNTIME_ERRORS; ++i) {
    gErrorBooking[i].store(false);
  }
}

void clean_runtime_error(int i)
{
  gErrorBooking[i].store(false);
}

RuntimeError& error_from_ref(RuntimeErrorRef ref)
{
  return gError[ref.index];
}

RuntimeErrorRef runtime_error_f(const char* format, ...)
{
  int i = 0;
  bool expected = false;
  while (gErrorBooking[i].compare_exchange_strong(expected, true) == false) {
    ++i;
  }
  va_list args;
  va_start(args, format);
  vsnprintf(gError[i].what, RuntimeError::MAX_RUNTIME_ERROR_SIZE, format, args);
  va_end(args);
  gError[i].maxBacktrace = canDumpBacktrace() ? backtrace(gError[i].backtrace, RuntimeError::MAX_BACKTRACE_SIZE) : 0;
  return RuntimeErrorRef{i};
}

RuntimeErrorRef runtime_error(const char* s)
{
  int i = 0;
  bool expected = false;
  while (gErrorBooking[i].compare_exchange_strong(expected, true) == false) {
    ++i;
  }
  strncpy(gError[i].what, s, RuntimeError::MAX_RUNTIME_ERROR_SIZE);
  gError[i].maxBacktrace = canDumpBacktrace() ? backtrace(gError[i].backtrace, RuntimeError::MAX_BACKTRACE_SIZE) : 0;
  return RuntimeErrorRef{i};
}

void throw_error(RuntimeErrorRef ref)
{
  throw ref;
}

void demangled_backtrace_symbols(void** stackTrace, unsigned int stackDepth, int fd)
{
  char** stackStrings;
  stackStrings = backtrace_symbols(stackTrace, stackDepth);
  for (size_t i = 1; i < stackDepth; i++) {

    size_t sz = 64000; // 64K ought to be enough for our templates...
    char* function = static_cast<char*>(malloc(sz));
    char *begin = nullptr, *end = nullptr;
    // find the last space and address offset surrounding the mangled name
#if __APPLE__
    for (char* j = stackStrings[i]; *j; ++j) {
      if (*j == ' ' && *(j + 1) != '+') {
        begin = j;
      } else if (*j == ' ' && *(j + 1) == '+') {
        end = j;
        break;
      }
    }
#else
    for (char* j = stackStrings[i]; *j; ++j) {
      if (*j == '(') {
        begin = j;
      } else if (*j == '+') {
        end = j;
        break;
      }
    }
#endif
    if (begin && end) {
      *begin++ = '\0';
      *end = '\0';
      // found our mangled name, now in [begin, end)

      int status;
      char* ret = abi::__cxa_demangle(begin, function, &sz, &status);
      if (ret) {
        // return value may be a realloc() of the input
        function = ret;
      } else {
        // demangling failed, just pretend it's a C function with no args
        std::strncpy(function, begin, sz);
        std::strncat(function, "()", sz);
        function[sz - 1] = '\0';
      }
      dprintf(fd, "    %s: %s\n", stackStrings[i], function);
    } else {
      // didn't find the mangled name, just print the whole line
      dprintf(fd, "    %s\n", stackStrings[i]);
    }
    free(function);
  }
  free(stackStrings); // malloc()ed by backtrace_symbols
  fsync(fd);
}

} // namespace o2::framework
