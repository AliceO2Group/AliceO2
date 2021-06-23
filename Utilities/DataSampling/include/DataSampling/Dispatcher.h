// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Dispatcher.h
/// \brief Declaration of Dispatcher for O2 Data Sampling
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#ifndef ALICEO2_DISPATCHER_H
#define ALICEO2_DISPATCHER_H

#include <string>
#include <vector>
#include <memory>

#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Task.h"

class FairMQDevice;

namespace o2::monitoring
{
class Monitoring;
}

namespace o2::utilities
{

class DataSamplingHeader;
class DataSamplingPolicy;

class Dispatcher : public framework::Task
{
 public:
  /// \brief Constructor
  Dispatcher(const std::string name, const std::string reconfigurationSource);
  /// \brief Destructor
  ~Dispatcher() override;

  /// \brief Dispatcher init callback
  void init(framework::InitContext& ctx) override;
  /// \brief Dispatcher process callback
  void run(framework::ProcessingContext& ctx) override;

  /// \brief Register a Data Sampling Policy
  void registerPolicy(std::unique_ptr<DataSamplingPolicy>&&);
  /// \brief Returns the number of registered policies.
  size_t numberOfPolicies();

  const std::string& getName();
  /// \brief Assembles InputSpecs of all registered policies in a single vector, removing overlapping entries.
  framework::Inputs getInputSpecs();
  framework::Outputs getOutputSpecs();
  framework::Options getOptions();

 private:
  DataSamplingHeader prepareDataSamplingHeader(const DataSamplingPolicy& policy, const framework::DeviceSpec& spec);
  header::Stack extractAdditionalHeaders(const char* inputHeaderStack) const;
  void reportStats(monitoring::Monitoring& monitoring) const;
  void send(framework::DataAllocator& dataAllocator, const framework::DataRef& inputData, framework::Output&& output) const;

  std::string mName;
  std::string mReconfigurationSource;
  // policies should be shared between all pipeline threads
  std::vector<std::shared_ptr<DataSamplingPolicy>> mPolicies;
};

} // namespace o2::utilities

#endif //ALICEO2_DISPATCHER_H
