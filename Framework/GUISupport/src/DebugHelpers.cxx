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

#include "DebugHelpers.h"
#include <string>

namespace o2::framework::gui
{
auto DebugHelpers::attachDebugger(int pid) -> void
{
  std::string pidStr = std::to_string(pid);
  setenv("O2DEBUGGEDPID", pidStr.c_str(), 1);
#ifdef __APPLE__
  std::string defaultAppleDebugCommand =
    "osascript -e 'tell application \"Terminal\"'"
    " -e 'activate'"
    " -e 'do script \"lldb -p \" & (system attribute \"O2DEBUGGEDPID\") & \"; exit\"'"
    " -e 'end tell'";
  setenv("O2DPLDEBUG", defaultAppleDebugCommand.c_str(), 0);
#else
  std::string gdbPath = "gdb";
  char* path = getenv("PATH");
  if (path && strlen(path) > 0) {
    path = strdup(path);
    char* dir;
    for (dir = strtok(path, ":"); dir; dir = strtok(NULL, ":")) {
      std::string fullPath = dir + "/" + gdbPath;
      if (access(fullPath.c_str(), F_OK) == 0) {
        gdbPath = fullPath;
        break;
      }
    }
    free(path);
  }
  std::string defaultGdbDebugCommand = fmt::format("xterm -hold -e {} attach $O2DEBUGGEDPID &", gdbPath);
  setenv("O2DPLDEBUG", defaultGdbDebugCommand.c_str(), 0);
#endif
  int retVal = system(getenv("O2DPLDEBUG"));
  (void)retVal;
};
} // namespace o2::framework::gui
