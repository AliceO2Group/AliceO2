// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/RuntimeError.h"

#include <cstdio>
#include <atomic>
#include <cstdarg>
#include <execinfo.h>
#include <cstring>

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

} // namespace o2::framework
