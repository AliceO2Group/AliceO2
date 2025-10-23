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

/// \file GPUErrors.h
/// \author David Rohr

#ifndef GPUERRORS_H
#define GPUERRORS_H

#include "GPUCommonDef.h"
#ifndef GPUCA_GPUCODE
#include <unordered_map>
#endif

namespace o2::gpu
{

class GPUErrors
{
 public:
  enum errorNumbers {
#define GPUCA_ERROR_CODE(num, name, ...) name = num,
#include "GPUErrorCodes.h"
#undef GPUCA_ERROR_CODE
  };

  GPUd() void raiseError(uint32_t code, uint32_t param1 = 0, uint32_t param2 = 0, uint32_t param3 = 0) const;
  GPUd() bool hasError() { return *mErrors > 0; }
  void setMemory(GPUglobalref() uint32_t* m) { mErrors = m; }
  void clear();
  bool printErrors(bool silent = false, uint64_t mask = 0);
#ifndef GPUCA_GPUCODE
  static const std::unordered_map<uint32_t, const char*>& getErrorNames();
#endif
  uint32_t getNErrors() const;
  const uint32_t* getErrorPtr() const;
  static uint32_t getMaxErrors();

 private:
  GPUglobalref() uint32_t* mErrors;
};

} // namespace o2::gpu

#endif
