// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//  @file   MCStepLoggerImpl.cxx
//  @author Sandro Wenzel
//  @since  2017-06-29
//  @brief  A logging service for MCSteps (hooking into Stepping of TVirtualMCApplication's)

#include "MCStepLogger/StepLogger.h"
#include "MCStepLogger/StepInfo.h"
#include "MCStepLogger/StepLoggerUtilities.h"
//#include "SimulationDataFormat/Stack.h"

#include <dlfcn.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

namespace o2
{
namespace data
{

class Stack
{
 public:
  void addHit(int iDet);
};

} // namespace data
} // namespace o2

// a generic function that can dispatch to the original method of a TVirtualMCApplication
extern "C" void dispatchOriginalAddHit(o2::data::Stack* stack, char const* libname, char const* origFunctionName, int iDet)
{
  using StepMethodType = void (o2::data::Stack::*)(int);
  o2::mcsteploggerutilities::dispatchOriginalKernel<o2::data::Stack, StepMethodType>(stack, libname, origFunctionName, iDet);
}

extern "C" void logHitDetector(int iDet)
{
  o2::StepLogger::Instance().addHit(iDet);
}

void o2::data::Stack::addHit(int iDet)
{
  // /auto baseptr = reinterpret_cast<o2::data::Stack*>(this);
  logHitDetector(iDet);
  dispatchOriginalAddHit(this, "libSimulationDataFormat", "_ZN2o24data5Stack6addHitEi", iDet);
}
