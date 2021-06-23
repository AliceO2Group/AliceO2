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
#ifndef O2_FRAMEWORK_ERROR_CONTEXT_H_
#define O2_FRAMEWORK_ERROR_CONTEXT_H_

#include "Framework/InputRecord.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/RuntimeError.h"

namespace o2::framework
{

// This is a utility class to reduce the amount of boilerplate when defining
// an error callback.
class ErrorContext
{
 public:
  ErrorContext(InputRecord& inputs, ServiceRegistry& services, RuntimeErrorRef e)
    : mInputs{inputs},
      mServices{services},
      mExceptionRef{e}
  {
  }

  InputRecord const& inputs() { return mInputs; }
  ServiceRegistry const& services() { return mServices; }
  RuntimeErrorRef exception() { return mExceptionRef; }

 private:
  InputRecord& mInputs;
  ServiceRegistry& mServices;
  RuntimeErrorRef mExceptionRef;
};

} // namespace o2::framework

#endif // o2::framework
