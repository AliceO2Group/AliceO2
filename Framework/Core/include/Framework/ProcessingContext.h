// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_PROCESSING_CONTEXT_H
#define FRAMEWORK_PROCESSING_CONTEXT_H

#include "Framework/InputRecord.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DataAllocator.h"

namespace o2 {
namespace framework {

// This is a utility class to reduce the amount of boilerplate when defining
// an algorithm.
class ProcessingContext {
public:
  ProcessingContext(InputRecord &inputs, ServiceRegistry &services, DataAllocator &allocator)
  :mInputs(inputs),
   mServices(services),
   mAllocator(allocator)
  {
  }

  InputRecord & inputs() {return mInputs;}
  ServiceRegistry & services() {return mServices;}
  DataAllocator& outputs() { return mAllocator; }

  InputRecord &mInputs;
  ServiceRegistry &mServices;
  DataAllocator &mAllocator;
};

} // namespace framework
} // namespace o2

#endif
