// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DISPATCHERFLPPROTO_H
#define ALICEO2_DISPATCHERFLPPROTO_H

#include "Framework/Dispatcher.h"
#include "Framework/DataSamplingConfig.h"
#include "FairMQDevice.h"

using namespace o2::framework;
using namespace o2::framework::DataSamplingConfig;

// class Dispatcher;

namespace o2
{
namespace framework
{

class DispatcherFlpProto : public Dispatcher
{
 public:
  DispatcherFlpProto() = delete;
  DispatcherFlpProto(const SubSpecificationType dispatcherSubSpec, const QcTaskConfiguration& task,
                     const InfrastructureConfig& cfg);
  ~DispatcherFlpProto() override;

  void addSource(const DataProcessorSpec& externalDataProcessor, const OutputSpec& externalOutput,
                 const std::string& binding) override;

 private:
  enum class FlpProtoState { Idle, ExpectingHeaderOrEOM, ExpectingPayload };
  /// Dispatcher init callback
  static AlgorithmSpec::ProcessCallback initCallback(InitContext& ctx, const std::string& channel, double fraction);
  /// Dispatcher callback with FairMQ output
  static void processCallback(ProcessingContext& ctx, Dispatcher::BernoulliGenerator& bernoulliGenerator,
                              FairMQDevice* device, const std::string& channel, FlpProtoState& state);
};

} // namespace framework
} // namespace o2

#endif // ALICEO2_DISPATCHERFLPPROTO_H
