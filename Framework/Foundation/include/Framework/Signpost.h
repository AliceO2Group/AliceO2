// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_SIGNPOST_H_
#define O2_FRAMEWORK_SIGNPOST_H_

#include <cstdint>

/// Signpost API implemented using different techonologies:
///
/// * macOS 10.15 onwards os_signpost
/// * macOS 10.14 and below (either kdebug_signpost or kdebug)
/// * linux SystemTap
///
/// Supported systems will have O2_SIGNPOST_API_AVAILABLE defined.
///
/// In order to use it, one must define O2_SIGNPOST_DEFINE_CONTEXT in at least one cxx file,
/// include "Framework/Signpost.h" and invoke O2_SIGNPOST_INIT().
#if defined(__APPLE__) && __has_include(<os/signpost.h>) && (__MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_10_15)
#include <os/signpost.h>
#include <os/log.h>
#ifdef OS_LOG_TARGET_HAS_10_15_FEATURES
#define O2_SIGNPOST_TYPE OS_LOG_CATEGORY_DYNAMIC_TRACING
#else
#define O2_SIGNPOST_TYPE OS_LOG_CATEGORY_POINTS_OF_INTEREST
#endif
#ifdef O2_SIGNPOST_DEFINE_CONTEXT
os_log_t gDPLLog = 0;
#else
static os_log_t gDPLLog;
#endif
#define O2_SIGNPOST_INIT() gDPLLog = os_log_create("ch.cern.alice.dpl", O2_SIGNPOST_TYPE);
#define O2_SIGNPOST(code, arg1, arg2, arg3, color) os_signpost_event_emit(gDPLLog, OS_SIGNPOST_ID_EXCLUSIVE, "##code", "%lu %lu %lu %lu", (uintptr_t)arg1, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)color)
#define O2_SIGNPOST_START(code, interval_id, arg2, arg3, color) os_signpost_interval_begin(gDPLLog, (os_signpost_id_t)interval_id, "##code", "%lu %lu %lu", (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)color)
#define O2_SIGNPOST_END(code, interval_id, arg2, arg3, color) os_signpost_interval_end(gDPLLog, (os_signpost_id_t)interval_id, "##code", "%lu %lu %lu", (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)color)
#define O2_SIGNPOST_API_AVAILABLE
#elif defined(__APPLE__) && __has_include(<sys/kdebug_signpost.h>) && (__MAC_OS_X_VERSION_MAX_ALLOWED < __MAC_10_15) // Deprecated in Catalina
#include <sys/kdebug_signpost.h>
#define O2_SIGNPOST_INIT()
#define O2_SIGNPOST(code, arg1, arg2, arg3, color) kdebug_signpost((uint32_t)code, (uintptr_t)arg1, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)color)
#define O2_SIGNPOST_START(code, interval_id, arg2, arg3, color) kdebug_signpost_start((uint32_t)code, (uintptr_t)interval_id, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)color)
#define O2_SIGNPOST_END(code, interval_id, arg2, arg3, color) kdebug_signpost_end((uint32_t)code, (uintptr_t)interval_id, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)color)
#define O2_SIGNPOST_API_AVAILABLE
#elif defined(__APPLE__) && __has_include(<sys/kdebug.h>) && (__MAC_OS_X_VERSION_MAX_ALLOWED < __MAC_10_15) // Compatibility with old API
#include <sys/kdebug.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifndef SYS_kdebug_trace
#define SYS_kdebug_trace 180
#endif
#define O2_SIGNPOST_INIT()
#define O2_SIGNPOST(code, arg1, arg2, arg3, arg4) syscall(SYS_kdebug_trace, APPSDBG_CODE(DBG_MACH_CHUD, (uint32_t)code) | DBG_FUNC_NONE, (uintptr_t)arg1, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)arg4);
#define O2_SIGNPOST_START(code, arg1, arg2, arg3, arg4) syscall(SYS_kdebug_trace, APPSDBG_CODE(DBG_MACH_CHUD, (uint32_t)code) | DBG_FUNC_START, (uintptr_t)arg1, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)arg4);
#define O2_SIGNPOST_END(code, arg1, arg2, arg3, arg4) syscall(SYS_kdebug_trace, APPSDBG_CODE(DBG_MACH_CHUD, (uintptr_t)code) | DBG_FUNC_END, (uintptr_t)arg1, (uintptr_t)arg2, (uintptr_t)arg3, (uintptr_t)arg4);
#define O2_SIGNPOST_API_AVAILABLE
#elif (!defined(__APPLE__)) && __has_include(<sys/sdt.h>) // Dtrace support is being dropped by Apple
#include <sys/sdt.h>
#define O2_SIGNPOST_INIT()
#define O2_SIGNPOST(code, arg1, arg2, arg3, arg4) STAP_PROBE4(dpl, probe##code, arg1, arg2, arg3, arg4)
#define O2_SIGNPOST_START(code, arg1, arg2, arg3, arg4) STAP_PROBE4(dpl, start_probe##code, arg1, arg2, arg3, arg4)
#define O2_SIGNPOST_END(code, arg1, arg2, arg3, arg4) STAP_PROBE4(dpl, stop_probe##code, arg1, arg2, arg3, arg4)
#define O2_SIGNPOST_API_AVAILABLE
#else // by default we do not do anything
#define O2_SIGNPOST_INIT()
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

#endif // O2_FRAMEWORK_SIGNPOST_H_
