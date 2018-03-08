// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DISPATCHERDPL_H
#define ALICEO2_DISPATCHERDPL_H

/// \file DispatcherDPL.h
/// \brief Definition of DispatcherDPL for O2 Data Sampling
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/Dispatcher.h"
#include "Framework/DataSamplingConfig.h"

using namespace o2::framework;
using namespace o2::framework::DataSamplingConfig;

namespace o2
{
namespace framework
{

/// \brief A dispatcher for clients inside Data Processing Layer.
class DispatcherDPL : public Dispatcher
{

 public:
  /// \brief Constructor.
  DispatcherDPL(const SubSpecificationType dispatcherSubSpec, const QcTaskConfiguration& task,
                const InfrastructureConfig& cfg);
  /// \brief Destructor
  ~DispatcherDPL() override;

  /// \brief Function responsible for adding new data source to dispatcher. It has different behaviour dependent
  /// on infrastructure configuration.
  void addSource(const DataProcessorSpec& externalDataProcessor, const OutputSpec& externalOutput,
                 const std::string& binding) override;

 private:
  /// Dispatcher callback with DPL outputs
  static void processCallback(ProcessingContext& ctx, Dispatcher::BernoulliGenerator& bernoulliGenerator);
};

} // namespace framework
} // namespace o2

#endif // ALICEO2_DISPATCHERDPL_H
