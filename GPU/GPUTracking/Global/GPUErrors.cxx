// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUErrors.cxx
/// \author David Rohr

#include "GPUErrors.h"
#include "GPUDataTypes.h"
#include "GPUCommonMath.h"
#include "GPUDefMacros.h"
#include "GPULogging.h"

using namespace GPUCA_NAMESPACE::gpu;

#define GPUCA_MAX_ERRORS 255u

GPUd() void GPUErrors::raiseError(unsigned int code, unsigned int param1, unsigned int param2, unsigned int param3) const
{
  unsigned int pos = CAMath::AtomicAdd(mErrors, 1u);
  if (pos < GPUCA_MAX_ERRORS) {
    mErrors[4 * pos + 1] = code;
    mErrors[4 * pos + 2] = param1;
    mErrors[4 * pos + 3] = param2;
    mErrors[4 * pos + 4] = param3;
  }
}

#ifndef GPUCA_GPUCODE

#include <cstring>
#include <unordered_map>

unsigned int GPUErrors::getMaxErrors()
{
  return GPUCA_MAX_ERRORS;
}

void GPUErrors::clear()
{
  memset(mErrors, 0, GPUCA_MAX_ERRORS * sizeof(*mErrors));
}

static std::unordered_map<unsigned int, const char*> errorNames = {
#define GPUCA_ERROR_CODE(num, name) {num, GPUCA_M_STR(name)},
#include "GPUErrorCodes.h"
#undef GPUCA_ERROR_CODE
};

void GPUErrors::printErrors()
{
  for (unsigned int i = 0; i < std::min(*mErrors, GPUCA_MAX_ERRORS); i++) {
    unsigned int errorCode = mErrors[4 * i + 1];
    const auto& it = errorNames.find(errorCode);
    const char* errorName = it == errorNames.end() ? "INVALID ERROR CODE" : it->second;
    GPUError("GPU Error Code (%u:%u) %s : %u / %u / %u", i, errorCode, errorName, mErrors[4 * i + 2], mErrors[4 * i + 3], mErrors[4 * i + 4]);
  }
  if (*mErrors > GPUCA_MAX_ERRORS) {
    GPUError("Additional errors occured (codes not stored)");
  }
}

#endif
