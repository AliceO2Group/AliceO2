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
#ifndef O2_FRAMEWORK_RUNTIMEERROR_H_
#define O2_FRAMEWORK_RUNTIMEERROR_H_

namespace o2::framework
{

struct RuntimeError {
  static constexpr unsigned int MAX_RUNTIME_ERRORS = 64;
  static constexpr unsigned int MAX_RUNTIME_ERROR_SIZE = 1024;
  static constexpr unsigned int MAX_BACKTRACE_SIZE = 100;
  char what[MAX_RUNTIME_ERROR_SIZE];
  void* backtrace[MAX_BACKTRACE_SIZE];
  int maxBacktrace = 0;
};

struct RuntimeErrorRef {
  int index = 0;
};

void demangled_backtrace_symbols(void** backtrace, unsigned int total, int fd);

RuntimeErrorRef runtime_error(const char*);
RuntimeErrorRef runtime_error_f(const char*, ...);
RuntimeError& error_from_ref(RuntimeErrorRef);

void throw_error(RuntimeErrorRef);
void clean_all_runtime_errors();

} // namespace o2::framework

#endif // O2_FRAMEWORK_RUNTIMEERROR_H_
