// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_Signpost_H_INCLUDED
#define o2_framework_Signpost_H_INCLUDED

/// Compatibility layer for kdebug_signpost like API. This will improve
/// profiling experience in Instruments.  I think something similar can be
/// achieved on Linux using sys/sdt.h.
#if __has_include(<sys/kdebug_signpost.h>) // This will be true Mojave onwards
#include <sys/kdebug_signpost.h>
#define O2_SIGNPOST(code, arg1, arg2, arg3, arg4) kdebug_signpost(code, arg1, arg2, arg3, arg4)
#define O2_SIGNPOST_START(code, arg1, arg2, arg3, arg4) kdebug_signpost_start(code, arg1, arg2, arg3, arg4)
#define O2_SIGNPOST_END(code, arg1, arg2, arg3, arg4) kdebug_signpost_end(code, arg1, arg2, arg3, arg4)
#else // by default we do not do anything
#define O2_SIGNPOST(code, arg1, arg2, arg3, arg4)
#define O2_SIGNPOST_START(code, arg1, arg2, arg3, arg4)
#define O2_SIGNPOST_END(code, arg1, arg2, arg3, arg4)
#endif

/// Helper class which allows the user to track
template <typename S>
struct StateMonitoring {
 public:
  static void start()
  {
    O2_SIGNPOST_START(S::ID, StateMonitoring<S>::count, 0, 0, StateMonitoring<S>::level);
  }

  static void moveTo(S newLevel)
  {
    if (newLevel == StateMonitoring<S>::level) {
      return;
    }
    O2_SIGNPOST_END(S::ID, StateMonitoring<S>::count, 0, 0, StateMonitoring<S>::level);
    StateMonitoring<S>::count++;
    StateMonitoring<S>::level = newLevel;
    O2_SIGNPOST_START(S::ID, StateMonitoring<S>::count, 0, 0, StateMonitoring<S>::level);
  }

  static void end()
  {
    O2_SIGNPOST_START(S::ID, StateMonitoring<S>::count, 0, 0, 0);
  }

  inline static int count = 0;
  inline static int level = 0;
};

#endif // o2_framework_Signpost_H_INCLUDED
