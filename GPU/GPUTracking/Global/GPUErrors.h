// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUErrors.h
/// \author David Rohr

#ifndef GPUERRORS_H
#define GPUERRORS_H

#include "GPUDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUErrors
{
 public:
  enum errorNumbers {
#define GPUCA_ERROR_CODE(num, name) name = num,
#include "GPUErrorCodes.h"
#undef GPUCA_ERROR_CODE
  };

  GPUd() void raiseError(unsigned int code, unsigned int param1 = 0, unsigned int param2 = 0, unsigned int param3 = 0) const;
  GPUd() bool hasError() { return *mErrors > 0; }
  void setMemory(GPUglobalref() unsigned int* m) { mErrors = m; }
  void clear();
  void printErrors();
  static unsigned int getMaxErrors();

 private:
  GPUglobalref() unsigned int* mErrors;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
