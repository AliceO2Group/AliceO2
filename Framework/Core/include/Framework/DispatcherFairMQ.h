// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DISPATCHERFAIRMQ_H
#define ALICEO2_DISPATCHERFAIRMQ_H

/// \file DispatcherFairMQ.h
/// \brief Definition of DispatcherFairMQ for O2 Data Sampling
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/Dispatcher.h"
#include "Framework/DataSamplingConfig.h"

namespace o2
{
namespace framework
{

using namespace o2::framework;
using namespace o2::framework::DataSamplingConfig;

/// \brief A dispatcher for clients that are FairMQDevices using O2 Data Model.
///
/// A dispatcher for clients outside of Data Processing Layer, specifically FairMQ devices using O2 Data Model.
/// For now, it sends only the payload, without the header - it will be added later probably.

class DispatcherFairMQ : public Dispatcher
{
 public:
  /// \brief Constructor.
  DispatcherFairMQ(const SubSpecificationType dispatcherSubSpec, const QcTaskConfiguration& task,
                   const InfrastructureConfig& cfg);
  /// \brief Destructor
  ~DispatcherFairMQ() override;

  /// \brief Function responsible for adding new data source to dispatcher.
  void addSource(const DataProcessorSpec& externalDataProcessor, const OutputSpec& externalOutput,
                 const std::string& binding) override;

 private:
  /// Dispatcher init callback
  static AlgorithmSpec::ProcessCallback initCallback(InitContext& ctx, const std::string& channel, double fraction);
  /// Dispatcher callback with FairMQ output
  static void processCallback(ProcessingContext& ctx, Dispatcher::BernoulliGenerator& bernoulliGenerator,
                              FairMQDevice* device, const std::string& channel);
};

} // namespace framework
} // namespace o2

#endif // ALICEO2_DISPATCHERFAIRMQ_H
