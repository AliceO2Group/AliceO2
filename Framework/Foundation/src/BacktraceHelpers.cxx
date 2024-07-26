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
#include "Framework/BacktraceHelpers.h"
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cxxabi.h>
#include <execinfo.h>

namespace o2::framework
{

void BacktraceHelpers::demangled_backtrace_symbols(void** stackTrace, unsigned int stackDepth, int fd)
{
  char** stackStrings;
  stackStrings = backtrace_symbols(stackTrace, stackDepth);
  char exe[PATH_MAX];
  bool hasExe = false;
  int exeSize = 0;

#if __linux__
  exeSize = readlink("/proc/self/exe", exe, PATH_MAX);
  if (exeSize == -1) {
    dprintf(fd, "Unable to detect exectuable name\n");
    hasExe = false;
  } else {
    dprintf(fd, "Executable is %.*s\n", exeSize, exe);
    hasExe = true;
  }
#endif

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
    bool tryAddr2Line = false;
#else
    for (char* j = stackStrings[i]; j && *j; ++j) {
      if (*j == '(') {
        begin = j;
      } else if (*j == '+') {
        end = j;
        break;
      }
    }
    bool tryAddr2Line = true;
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
        dprintf(fd, "    %s: %s\n", stackStrings[i], function);
        tryAddr2Line = false;
      }
    }
    if (tryAddr2Line) {
      // didn't find the mangled name, just print the whole line
      dprintf(fd, "    %s: ", stackStrings[i]);
      if (stackTrace[i] && hasExe) {
        char syscom[4096 + PATH_MAX];

        // Find c++filt from the environment
        // This is needed for platforms where we still need c++filt -r
        char const* cxxfilt = getenv("CXXFILT");
        if (cxxfilt == nullptr) {
          cxxfilt = "c++filt";
        }
        // Do the same for addr2line, just in case we wanted to pass some options
        char const* addr2line = getenv("ADDR2LINE");
        if (addr2line == nullptr) {
          addr2line = "addr2line";
        }
        snprintf(syscom, 4096, "%s %p -p -s -f -e %.*s 2>/dev/null | %s ", addr2line, stackTrace[i], exeSize, exe, cxxfilt); // last parameter is the name of this app

        FILE* fp;
        char path[1024];

        fp = popen(syscom, "r");
        if (fp == nullptr) {
          dprintf(fd, "-- no source could be retrieved --\n");
          continue;
        }

        while (fgets(path, sizeof(path) - 1, fp) != nullptr) {
          dprintf(fd, "    %s", path);
        }

        pclose(fp);
      } else {
        dprintf(fd, "-- no source avaliable --\n");
      }
    }
    free(function);
  }
  free(stackStrings); // malloc()ed by backtrace_symbols
  fsync(fd);
}
} // namespace o2::framework
