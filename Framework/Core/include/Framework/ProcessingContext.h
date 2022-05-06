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
#ifndef O2_FRAMEWORK_PROCESSINGCONTEXT_H_
#define O2_FRAMEWORK_PROCESSINGCONTEXT_H_

namespace o2::framework
{

struct ServiceRegistry;
struct DataAllocator;
struct InputRecord;

// This is a utility class to reduce the amount of boilerplate when  defining
// an algorithm.
class ProcessingContext
{
 public:
  ProcessingContext(InputRecord& inputs, ServiceRegistry& services, DataAllocator& allocator)
    : mInputs(inputs),
      mServices(services),
      mAllocator(allocator)
  {
  }

  /// The inputs associated with this processing context.
  InputRecord& inputs() { return mInputs; }
  /// The services registry associated with this processing context.
  ServiceRegistry& services() { return mServices; }
  /// The data allocator is used to allocate memory for the output data.
  DataAllocator& outputs() { return mAllocator; }

  InputRecord& mInputs;
  ServiceRegistry& mServices;
  DataAllocator& mAllocator;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_PROCESSINGCONTEXT_H_
