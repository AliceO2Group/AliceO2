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
#ifndef O2_FRAMEWORK_DATAPROCESSORCONTEXT_H_
#define O2_FRAMEWORK_DATAPROCESSORCONTEXT_H_

#include "Framework/DataRelayer.h"
#include "Framework/AlgorithmSpec.h"
#include <functional>

namespace o2::framework
{

struct DeviceContext;
struct ServiceRegistry;
struct DataAllocator;
struct DataProcessorSpec;

struct DataProcessorContext {
  // These are specific of a given context and therefore
  // not shared by threads.
  bool* wasActive = nullptr;
  bool allDone = false;

  // These are pointers to the one owned by the DataProcessingDevice
  // but they are fully reentrant / thread safe and therefore can
  // be accessed without a lock.

  // FIXME: move stuff here from the list below... ;-)
  ServiceRegistry* registry = nullptr;
  std::vector<DataRelayer::RecordAction> completed;
  std::vector<ExpirationHandler> expirationHandlers;
  AlgorithmSpec::InitCallback init;
  AlgorithmSpec::ProcessCallback statefulProcess;
  AlgorithmSpec::ProcessCallback statelessProcess;
  AlgorithmSpec::ErrorCallback error = nullptr;

  DataProcessorSpec* spec = nullptr;

  /// Wether or not the associated DataProcessor can forward things early
  bool canForwardEarly = true;
  bool isSink = false;
  bool balancingInputs = true;

  std::function<void(o2::framework::RuntimeErrorRef e, InputRecord& record)> errorHandling = nullptr;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_DATAPROCESSINGCONTEXT_H_
