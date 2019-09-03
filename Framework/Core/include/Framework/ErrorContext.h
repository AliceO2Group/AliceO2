// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_ERROR_CONTEXT_H
#define FRAMEWORK_ERROR_CONTEXT_H

#include "Framework/InputRecord.h"
#include "Framework/ServiceRegistry.h"

#include <exception>

namespace o2
{
namespace framework
{

// This is a utility class to reduce the amount of boilerplate when defining
// an error callback.
class ErrorContext
{
 public:
  ErrorContext(InputRecord& inputs, ServiceRegistry& services, std::exception& e)
    : mInputs{inputs},
      mServices{services},
      mException{e}
  {
  }

  InputRecord const& inputs() { return mInputs; }
  ServiceRegistry const& services() { return mServices; }
  std::exception const& exception() { return mException; }

  InputRecord& mInputs;
  ServiceRegistry& mServices;
  std::exception& mException;
};

} // namespace framework
} // namespace o2

#endif
