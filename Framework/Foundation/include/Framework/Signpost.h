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
#ifndef O2_FRAMEWORK_SIGNPOST_H_
#define O2_FRAMEWORK_SIGNPOST_H_

#include "Framework/CompilerBuiltins.h"
#include <atomic>
#include <array>
#ifdef __APPLE__
#include <os/log.h>
#endif

struct o2_log_handle_t {
  char const* name = nullptr;
  void* log = nullptr;
  o2_log_handle_t* next = nullptr;
};

// Helper function which replaces engineering types with a printf
// compatible format string.
// FIXME: make this consteval when available in C++20
template <auto N>
constexpr auto remove_engineering_type(char const (&src)[N])
{
  std::array<char, N> res = {};
  // do whatever string manipulation you want in res.
  char* t = res.data();
  for (int i = 0; i < N; ++i) {
    if (src[i] == '%' && src[i + 1] == '{') {
      *t++ = src[i];
      while (src[i] != '}' && src[i] != 0) {
        ++i;
      }
      if (src[i] == 0) {
        *t = 0;
        return res;
      }
    } else {
      *t++ = src[i];
    }
  }
  return res;
}

// Loggers registry is actually a feature available to all platforms
// We use this to register the loggers and to walk over them.
// So that also on mac we can have a list of all the registered loggers.
std::atomic<o2_log_handle_t*>& o2_get_logs_tail();
o2_log_handle_t* o2_walk_logs(bool (*callback)(char const* name, void* log, void* context), void* context = nullptr);

#ifdef O2_SIGNPOST_IMPLEMENTATION
// The first log of the list. We make it atomic,
// so that we can add new logs from different threads.
std::atomic<o2_log_handle_t*>& o2_get_logs_tail()
{
  static std::atomic<o2_log_handle_t*> first = nullptr;
  return first;
}

/// Walk over the logs and call the callback for each log.
/// If the callback returns false, the loop is broken and
/// the current log is returned.
/// Returns the last log otherwise.
/// This way we can use this to iterate over the logs or
/// to insert new logs if none matches.
o2_log_handle_t* o2_walk_logs(bool (*callback)(char const* name, void* log, void* context), void* context)
{
  // This might skip newly inserted logs, but that is ok.
  o2_log_handle_t* current = o2_get_logs_tail().load();
  while (current) {
    bool cont = callback(current->name, current->log, context);
    // In case we should not continue, break out of the loop.
    if (cont == false) {
      return current;
    }
    current = current->next;
  }
  return current;
}
#endif

#if defined(__APPLE__)
#include <os/log.h>
#include <os/signpost.h>
#include <cstring>
#define O2_LOG_DEBUG_MAC(log, ...) os_log_debug(private_o2_log_##log, __VA_ARGS__)
// FIXME: use __VA_OPT__ when available in C++20
#define O2_SIGNPOST_EVENT_EMIT_MAC(log, id, name, format, ...) os_signpost_event_emit(private_o2_log_##log->os_log, (uint64_t)id.value, name, format, ##__VA_ARGS__)
#define O2_SIGNPOST_START_MAC(log, id, name, format, ...) os_signpost_interval_begin(private_o2_log_##log->os_log, (uint64_t)id.value, name, format, ##__VA_ARGS__)
#define O2_SIGNPOST_END_MAC(log, id, name, format, ...) os_signpost_interval_end(private_o2_log_##log->os_log, (uint64_t)id.value, name, format, ##__VA_ARGS__)
#define O2_SIGNPOST_ENABLED_MAC(log) os_signpost_enabled(private_o2_log_##log->os_log)
#else
// These are no-ops on linux.
#define O2_DECLARE_LOG_MAC(x, category)
#define O2_LOG_DEBUG_MAC(log, ...)
#define O2_SIGNPOST_EVENT_EMIT_MAC(log, id, name, format, ...)
#define O2_SIGNPOST_START_MAC(log, id, name, format, ...)
#define O2_SIGNPOST_END_MAC(log, id, name, format, ...)
#define O2_SIGNPOST_ENABLED_MAC(log) false
#endif // __APPLE__

// Unless we are on apple we enable checking for signposts only if in debug mode or if we force them.
#if defined(__APPLE__) || defined(O2_FORCE_SIGNPOSTS) || !defined(O2_NSIGNPOSTS)
#define O2_LOG_ENABLED(log) private_o2_log_##log->stacktrace
#else
#define O2_LOG_ENABLED(log) false
#endif

#if !defined(O2_LOG_MACRO) && __has_include("Framework/Logger.h")
#include "Framework/Logger.h"
#define O2_LOG_MACRO(...) LOGF(info, __VA_ARGS__)
#elif !defined(O2_LOG_MACRO)
// If we do not have the fairlogger, we simply print out the signposts to the console.
// This is useful for things like the tests, which this way do not need to depend on the FairLogger.
#define O2_LOG_MACRO(...) \
  do {                    \
    printf(__VA_ARGS__);  \
    printf("\n");         \
  } while (0)
#else
#define O2_LOG_MACRO(...)
#endif // O2_LOG_MACRO

// This is the linux implementation, it is not as nice as the apple one and simply prints out
// the signpost information to the log.
#include <atomic>
#include <array>
#include <cassert>
#include <cinttypes>
#include <cstddef>

struct _o2_lock_free_stack {
  static constexpr size_t N = 1024;
  std::atomic<size_t> top = 0;
  int stack[N];
};

// A log is simply an inbox which keeps track of the available id, so that we can print out different signposts
// with different indentation levels.
// supports up to 1024 paralle signposts before it spinlocks.
using _o2_signpost_index_t = int;

struct _o2_activity_t {
  // How much the activity is indented in the output log.
  unsigned char indentation = 0;
  char const* name = nullptr;
};

struct _o2_signpost_id_t {
  // The id of the activity.
  int64_t value = -1;
};

struct _o2_log_t {
#ifdef __APPLE__
  os_log_t os_log = nullptr;
#endif
  // A circular buffer of available slots. Each unique interval pulls an id from this buffer.
  _o2_lock_free_stack slots;
  // Up to 256 activities can be active at the same time.
  std::array<_o2_signpost_id_t, 256> ids = {};
  // The intervals associated with each slot.
  // We use this to keep track of the indentation level for messages associated to it
  std::array<_o2_activity_t, 256> activities = {};
  std::atomic<int64_t> current_indentation = 0;
  // Each thread needs to maintain a stack for the intervals, so that
  // you can have nested intervals.
  std::atomic<int64_t> unique_signpost = 1;

  // how many stacktrace levels print per log.
  // 0 means the log is disabled.
  // 1 means only the current signpost is printed.
  // >1 means the current signpost and n levels of the stacktrace are printed.
  int stacktrace = 0;

  // Default stacktrace level for the log, when enabled.
  int defaultStacktrace = 1;
};

bool _o2_lock_free_stack_push(_o2_lock_free_stack& stack, const int& value, bool spin = false);
bool _o2_lock_free_stack_pop(_o2_lock_free_stack& stack, int& value, bool spin = false);
//_o2_signpost_id_t _o2_signpost_id_generate_local(_o2_log_t* log);
//_o2_signpost_id_t _o2_signpost_id_make_with_pointer(_o2_log_t* log, void* pointer);
void* _o2_log_create(char const* name, int stacktrace);
void _o2_signpost_event_emit(_o2_log_t* log, _o2_signpost_id_t id, char const* name, char const* const format, ...);
void _o2_signpost_interval_begin(_o2_log_t* log, _o2_signpost_id_t id, char const* name, char const* const format, ...);
void _o2_signpost_interval_end(_o2_log_t* log, _o2_signpost_id_t id, char const* name, char const* const format, ...);
void _o2_log_set_stacktrace(_o2_log_t* log, int stacktrace);

// This generates a unique id for a signpost. Do not use this directly, use O2_SIGNPOST_ID_GENERATE instead.
// Notice that this is only valid on a given computer.
// This is guaranteed to be unique at 5 GHz for at least 63 years, if my math is correct.
// I doubt we will have a job running for that long or that CPU scaling will shorten that period too much.
// If you want to use this on the Nostromo, please think twice about it.
// We use odd numbers so that pointers to things which are not bytes are not confused with
// the generated ones.
inline _o2_signpost_id_t _o2_signpost_id_generate_local(_o2_log_t* log)
{
  return {(log->unique_signpost++ * 2) + 1};
}

// Generate a unique id for a signpost. Do not use this directly, use O2_SIGNPOST_ID_FROM_POINTER instead.
// Notice that this will fail for pointers to bytes as it might overlap with the id above.
inline _o2_signpost_id_t _o2_signpost_id_make_with_pointer(_o2_log_t* log, void* pointer)
{
  assert(((int64_t)pointer & 1) != 1);
  _o2_signpost_id_t uniqueId{(int64_t)pointer};
  return uniqueId;
}

// Implementation start here. Include this file with O2_SIGNPOST_IMPLEMENTATION defined in one file of your
// project.
#ifdef O2_SIGNPOST_IMPLEMENTATION
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include "Framework/RuntimeError.h"
void _o2_signpost_interval_end_v(_o2_log_t* log, _o2_signpost_id_t id, char const* name, char const* const format, va_list args);

// returns true if the push was successful, false if the stack was full
// @param spin if true, will spin until the stack is not full
bool _o2_lock_free_stack_push(_o2_lock_free_stack& stack, const int& value, bool spin)
{
  size_t currentTop = stack.top.load(std::memory_order_relaxed);
  while (true) {
    if (currentTop == _o2_lock_free_stack::N && spin == false) {
      return false;
    } else if (currentTop == _o2_lock_free_stack::N) {
// Avoid consuming too much CPU time if we are spinning.
#if defined(__x86_64__) || defined(__i386__)
      __asm__ __volatile__("pause" ::
                             : "memory");
#elif defined(__aarch64__)
      __asm__ __volatile__("yield" ::
                             : "memory");
#endif
      continue;
    }

    if (stack.top.compare_exchange_weak(currentTop, currentTop + 1,
                                        std::memory_order_release,
                                        std::memory_order_relaxed)) {
      stack.stack[currentTop] = value;
      return true;
    }
  }
}

bool _o2_lock_free_stack_pop(_o2_lock_free_stack& stack, int& value, bool spin)
{
  size_t currentTop = stack.top.load(std::memory_order_relaxed);
  while (true) {
    if (currentTop == 0 && spin == false) {
      return false;
    } else if (currentTop == 0) {
// Avoid consuming too much CPU time if we are spinning.
#if defined(__x86_64__) || defined(__i386__)
      __asm__ __volatile__("pause" ::
                             : "memory");
#elif defined(__aarch64__)
      __asm__ __volatile__("yield" ::
                             : "memory");
#endif
      continue;
    }

    if (stack.top.compare_exchange_weak(currentTop, currentTop - 1,
                                        std::memory_order_acquire,
                                        std::memory_order_relaxed)) {
      value = stack.stack[currentTop - 1];
      return true;
    }
  }
}

void* _o2_log_create(char const* name, int defaultStacktrace)
{
  // iterate over the list of logs and check if we already have
  // one with the same name.
  o2_log_handle_t* handle = o2_walk_logs([](char const* currentName, void* log, void* context) -> bool {
    char const* name = (char const*)context;
    if (strcmp(name, currentName) == 0) {
      return false;
    }
    return true;
  },
                                         (void*)name);

  // If we found one, return it.
  if (handle) {
    return handle->log;
  }
  // Otherwise, create a new one and add it to the end of the list.
  auto* log = new _o2_log_t();
  // Write the initial 256 ids to the inbox, in reverse, so that the
  // linear search below is just for an handful of elements.
  int n = _o2_lock_free_stack::N;
  for (int i = 0; i < n; i++) {
    _o2_signpost_index_t signpost_index{n - 1 - i};
    _o2_lock_free_stack_push(log->slots, signpost_index, true);
  }
  log->defaultStacktrace = defaultStacktrace;
  auto* newHandle = new o2_log_handle_t();
  newHandle->log = log;
#ifdef __APPLE__
  // On macOS, we use the os_signpost API so that when we are
  // using instruments we can see the messages there.
  if (defaultStacktrace > 1) {
    log->os_log = os_log_create(name, OS_LOG_CATEGORY_DYNAMIC_STACK_TRACING);
  } else {
    log->os_log = os_log_create(name, OS_LOG_CATEGORY_DYNAMIC_TRACING);
  }
#endif
  newHandle->name = strdup(name);
  newHandle->next = o2_get_logs_tail().load();
  // Until I manage to replace the log I have in next, keep trying.
  // Notice this does not protect against two threads trying to insert
  // a log with the same name. I should probably do a sorted insert for that.
  while (!o2_get_logs_tail().compare_exchange_weak(newHandle->next, newHandle,
                                                   std::memory_order_release,
                                                   std::memory_order_relaxed)) {
    newHandle->next = o2_get_logs_tail();
  }

  return log;
}

// This will look at the slot in the log associated to the ID.
// If the slot is empty, it will return the id and increment the indentation level.
void _o2_signpost_event_emit(_o2_log_t* log, _o2_signpost_id_t id, char const* name, char const* const format, ...)
{
  va_list args;
  va_start(args, format);

  // Find the index of the activity
  int leading = 0;

  // This is the equivalent of exclusive
  if (id.value != 0) {
    int i = 0;
    for (i = 0; i < log->ids.size(); ++i) {
      if (log->ids[i].value == id.value) {
        break;
      }
    }
    // If the id is not in the list, then we consider it as a standalone event and we print
    // it at toplevel.
    if (i != log->ids.size()) {
      // we found an interval associated to this id.
      _o2_activity_t* activity = &log->activities[i];
      leading = activity->indentation * 2;
    }
  }

  char prebuffer[4096];
  int s = snprintf(prebuffer, 4096, "id%.16" PRIx64 ":%-16s*>%*c", id.value, name, leading, ' ');
  vsnprintf(prebuffer + s, 4096 - s, format, args);
  va_end(args);
  O2_LOG_MACRO("%s", prebuffer);
}

// This will look at the slot in the log associated to the ID.
// If the slot is empty, it will return the id and increment the indentation level.
void _o2_signpost_interval_begin(_o2_log_t* log, _o2_signpost_id_t id, char const* name, char const* const format, ...)
{
  va_list args;
  va_start(args, format);
  // This is a unique slot for this interval.
  _o2_signpost_index_t signpost_index;
  _o2_lock_free_stack_pop(log->slots, signpost_index, true);
  // Put the id in the slot, to close things or to attach signposts to a given interval
  log->ids[signpost_index].value = id.value;
  auto* activity = &log->activities[signpost_index];
  activity->indentation = log->current_indentation++;
  activity->name = name;
  int leading = activity->indentation * 2;
  char prebuffer[4096];
  int s = snprintf(prebuffer, 4096, "id%.16" PRIx64 ":%-16sS>%*c", id.value, name, leading, ' ');
  vsnprintf(prebuffer + s, 4096 - s, format, args);
  va_end(args);
  O2_LOG_MACRO("%s", prebuffer);
}

void _o2_signpost_interval_end_v(_o2_log_t* log, _o2_signpost_id_t id, char const* name, char const* const format, va_list args)
{
  if (log->stacktrace == 0) {
    return;
  }
  // Find the index of the activity
  int i = 0;
  for (i = 0; i < log->ids.size(); ++i) {
    if (log->ids[i].value == id.value) {
      break;
    }
  }
  // If we do not find a matching id, then we just emit this as an event in then log.
  // We should not make this an error because one could have enabled the log after the interval
  // was started.
  if (i == log->ids.size()) {
    _o2_signpost_event_emit(log, id, name, format, args);
    return;
  }
  // i is the slot index
  _o2_activity_t* activity = &log->activities[i];
  int leading = activity->indentation * 2;
  char prebuffer[4096];
  int s = snprintf(prebuffer, 4096, "id%.16" PRIx64 ":%-16sE>%*c", id.value, name, leading, ' ');
  vsnprintf(prebuffer + s, 4096 - s, format, args);
  O2_LOG_MACRO("%s", prebuffer);
  // Clear the slot
  activity->indentation = -1;
  activity->name = nullptr;
  log->ids[i].value = -1;
  // Put back the slot
  log->current_indentation--;
  _o2_signpost_index_t signpost_index{i};
  _o2_lock_free_stack_push(log->slots, signpost_index, true);
}

// We separate this so that we can still emit the end signpost when the log is not enabled.
void _o2_signpost_interval_end(_o2_log_t* log, _o2_signpost_id_t id, char const* name, char const* const format, ...)
{
  va_list args;
  va_start(args, format);
  _o2_signpost_interval_end_v(log, id, name, format, args);
  va_end(args);
  return;
}

void _o2_log_set_stacktrace(_o2_log_t* log, int stacktrace)
{
  log->stacktrace = stacktrace;
}
// A C function which can be used to enable the signposts
extern "C" {
void o2_debug_log_set_stacktrace(_o2_log_t* log, int stacktrace)
{
  log->stacktrace = stacktrace;
}
}
#endif // O2_SIGNPOST_IMPLEMENTATION

#if defined(__APPLE__) || defined(O2_FORCE_SIGNPOSTS) || !defined(O2_NSIGNPOSTS)
/// Dynamic logs need to be enabled via the O2_LOG_ENABLE macro. Notice this will only work
/// for the logger based logging, since the Apple version needs instruments to enable them.
#define O2_DECLARE_DYNAMIC_LOG(name) static _o2_log_t* private_o2_log_##name = (_o2_log_t*)_o2_log_create("ch.cern.aliceo2." #name, 1)
/// For the moment we do not support logs with a stacktrace.
#define O2_DECLARE_DYNAMIC_STACKTRACE_LOG(name) static _o2_log_t* private_o2_log_##name = (_o2_log_t*)_o2_log_create("ch.cern.aliceo2." #name, 64)
#define O2_DECLARE_LOG(name, category) static _o2_log_t* private_o2_log_##name = (_o2_log_t*)_o2_log_create("ch.cern.aliceo2." #name, 1)
// When we enable the log, we set the stacktrace to the default value.
#define O2_LOG_ENABLE(log) _o2_log_set_stacktrace(private_o2_log_##log, private_o2_log_##log->defaultStacktrace)
#define O2_LOG_DISABLE(log) _o2_log_set_stacktrace(private_o2_log_##log, 0)
// For the moment we simply use LOG DEBUG. We should have proper activities so that we can
// turn on and off the printing.
#define O2_LOG_DEBUG(log, ...) __extension__({                        \
  if (O2_BUILTIN_UNLIKELY(O2_LOG_ENABLED(log))) {                     \
    O2_LOG_MACRO(__VA_ARGS__);                                        \
  } else if (O2_BUILTIN_UNLIKELY(private_o2_log_##log->stacktrace)) { \
    O2_LOG_MACRO(__VA_ARGS__);                                        \
  }                                                                   \
})
#define O2_SIGNPOST_ID_FROM_POINTER(name, log, pointer) _o2_signpost_id_t name = _o2_signpost_id_make_with_pointer(private_o2_log_##log, pointer)
#define O2_SIGNPOST_ID_GENERATE(name, log) _o2_signpost_id_t name = _o2_signpost_id_generate_local(private_o2_log_##log)
// In case Instruments is attached, we switch to the Apple signpost API otherwise, both one
// mac and on linux we use our own implementation, using the logger. We can use the same ids because
// they are compatible between the two implementations, we also use remove_engineering_type to remove
// the engineering types from the format string, so that we can use the same format string for both.
#define O2_SIGNPOST_EVENT_EMIT(log, id, name, format, ...) __extension__({                                          \
  if (O2_BUILTIN_UNLIKELY(O2_SIGNPOST_ENABLED_MAC(log))) {                                                          \
    O2_SIGNPOST_EVENT_EMIT_MAC(log, id, name, format, ##__VA_ARGS__);                                               \
  } else if (O2_BUILTIN_UNLIKELY(private_o2_log_##log->stacktrace)) {                                               \
    _o2_signpost_event_emit(private_o2_log_##log, id, name, remove_engineering_type(format).data(), ##__VA_ARGS__); \
  }                                                                                                                 \
})
#define O2_SIGNPOST_START(log, id, name, format, ...)                                                                   \
  if (O2_BUILTIN_UNLIKELY(O2_SIGNPOST_ENABLED_MAC(log))) {                                                              \
    O2_SIGNPOST_START_MAC(log, id, name, format, ##__VA_ARGS__);                                                        \
  } else if (O2_BUILTIN_UNLIKELY(private_o2_log_##log->stacktrace)) {                                                   \
    _o2_signpost_interval_begin(private_o2_log_##log, id, name, remove_engineering_type(format).data(), ##__VA_ARGS__); \
  }
#define O2_SIGNPOST_END(log, id, name, format, ...)                                                                   \
  if (O2_BUILTIN_UNLIKELY(O2_SIGNPOST_ENABLED_MAC(log))) {                                                            \
    O2_SIGNPOST_END_MAC(log, id, name, format, ##__VA_ARGS__);                                                        \
  } else if (O2_BUILTIN_UNLIKELY(private_o2_log_##log->stacktrace)) {                                                 \
    _o2_signpost_interval_end(private_o2_log_##log, id, name, remove_engineering_type(format).data(), ##__VA_ARGS__); \
  }
#else // This is the release implementation, it does nothing.
#define O2_DECLARE_DYNAMIC_LOG(x)
#define O2_DECLARE_DYNAMIC_STACKTRACE_LOG(x)
#define O2_DECLARE_LOG(x, category)
#define O2_LOG_ENABLE(log)
#define O2_LOG_DISABLE(log)
#define O2_LOG_DEBUG(log, ...)
#define O2_SIGNPOST_ID_FROM_POINTER(name, log, pointer)
#define O2_SIGNPOST_ID_GENERATE(name, log)
#define O2_SIGNPOST_EVENT_EMIT(log, id, name, format, ...)
#define O2_SIGNPOST_START(log, id, name, format, ...)
#define O2_SIGNPOST_END(log, id, name, format, ...)
#endif

#endif // O2_FRAMEWORK_SIGNPOST_H_
