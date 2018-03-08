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

/// \file DispatcherFlpProto.h
/// \brief Definition of DispatcherFlpProto for O2 Data Sampling
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/Dispatcher.h"
#include "Framework/DataSamplingConfig.h"
#include "FairMQDevice.h"

using namespace o2::framework;
using namespace o2::framework::DataSamplingConfig;

namespace o2
{
namespace framework
{

/// \brief A special dispatcher for QC tasks that are FairMQ devices, using FLP Proto data model.
class DispatcherFlpProto : public Dispatcher
{
 public:
  /// \brief Constructor.
  DispatcherFlpProto(const SubSpecificationType dispatcherSubSpec, const QcTaskConfiguration& task,
                     const InfrastructureConfig& cfg);
  /// \brief Destructor
  ~DispatcherFlpProto() override;

  /// \brief Function responsible for adding new data source to dispatcher.
  void addSource(const DataProcessorSpec& externalDataProcessor, const OutputSpec& externalOutput,
                 const std::string& binding) override;

 private:
  /// Class used to keep track of protocol state. Expected message order: Header-Payload-Header-Payload-...-EOM
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
