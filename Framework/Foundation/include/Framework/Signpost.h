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

#include <cstdint>

/// Compatibility layer for kdebug_signpost like API. This will improve
/// profiling experience in Instruments.  I think something similar can be
/// achieved on Linux using sys/sdt.h.
#if __has_include(<sys/kdebug_signpost.h>) // This will be true Mojave onwards
#include <sys/kdebug_signpost.h>
#define O2_SIGNPOST(code, arg1, arg2, arg3, color) kdebug_signpost((uint32_t)code, (uintptr_t)arg1, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)color)
#define O2_SIGNPOST_START(code, interval_id, arg2, arg3, color) kdebug_signpost_start((uint32_t)code, (uintptr_t)interval_id, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)color)
#define O2_SIGNPOST_END(code, interval_id, arg2, arg3, color) kdebug_signpost_end((uint32_t)code, (uintptr_t)interval_id, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)color)
#elif __has_include(<sys/kdebug.h>) // Compatibility with old API
#include <sys/kdebug.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifndef SYS_kdebug_trace
#define SYS_kdebug_trace 180
#endif
#define O2_SIGNPOST(code, arg1, arg2, arg3, arg4) syscall(SYS_kdebug_trace, APPSDBG_CODE(DBG_MACH_CHUD, (uint32_t)code) | DBG_FUNC_NONE, (uintptr_t)arg1, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)arg4);
#define O2_SIGNPOST_START(code, arg1, arg2, arg3, arg4) syscall(SYS_kdebug_trace, APPSDBG_CODE(DBG_MACH_CHUD, (uint32_t)code) | DBG_FUNC_START, (uintptr_t)arg1, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)arg4);
#define O2_SIGNPOST_END(code, arg1, arg2, arg3, arg4) syscall(SYS_kdebug_trace, APPSDBG_CODE(DBG_MACH_CHUD, (uintptr_t)code) | DBG_FUNC_END, (uintptr_t)arg1, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)arg4);
#elif __has_include(<sys/sdt.h>) // This will be true on Linux if systemtap-std-dev / systemtap-std-devel
#include <sys/sdt.h>
#define O2_SIGNPOST(code, arg1, arg2, arg3, arg4) STAP_PROBE4(dpl, probe##code, arg1, arg2, arg3, arg4)
#define O2_SIGNPOST_START(code, arg1, arg2, arg3, arg4) STAP_PROBE4(dpl, start_probe##code, arg1, arg2, arg3, arg4)
#define O2_SIGNPOST_END(code, arg1, arg2, arg3, arg4) STAP_PROBE4(dpl, stop_probe##code, arg1, arg2, arg3, arg4)
#else // by default we do not do anything
#define O2_SIGNPOST(code, arg1, arg2, arg3, arg4)
#define O2_SIGNPOST_START(code, arg1, arg2, arg3, arg4)
#define O2_SIGNPOST_END(code, arg1, arg2, arg3, arg4)
#endif

/// Colors for the signpost while shown in instruments.
/// Notice we use defines and not enums becasue STAP / DTRACE seems not
/// to play well with it, due to macro expansion tricks.
///
/// FIXME: we should have sensible defaults also for some DTrace / STAP
/// GUI.
#define O2_SIGNPOST_BLUE 0
#define O2_SIGNPOST_GREEN 1
#define O2_SIGNPOST_PURPLE 2
#define O2_SIGNPOST_ORANGE 3
#define O2_SIGNPOST_RED 4

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
    if ((uint32_t)newLevel == StateMonitoring<S>::level) {
      return;
    }
    O2_SIGNPOST_END(S::ID, StateMonitoring<S>::count, 0, 0, StateMonitoring<S>::level);
    StateMonitoring<S>::count++;
    StateMonitoring<S>::level = (uint32_t)newLevel;
    O2_SIGNPOST_START(S::ID, StateMonitoring<S>::count, 0, 0, StateMonitoring<S>::level);
  }

  static void end()
  {
    O2_SIGNPOST_END(S::ID, StateMonitoring<S>::count, 0, 0, 0);
  }

  inline static uint32_t count = 0;
  inline static uint32_t level = 0;
};

#endif // o2_framework_Signpost_H_INCLUDED
