// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DATASAMPLINGCONFIG_H
#define ALICEO2_DATASAMPLINGCONFIG_H

/// \file DataSamplingConfig.h
/// \brief Helper structures for O2 Data Sampling configuration
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"

#include "Headers/DataHeader.h"

#include <vector>
#include <string>

namespace o2
{
namespace framework
{
// consider: make that just a boost variable map?

/// A bunch of helper structures for DataSampling configuration.
namespace DataSamplingConfig
{
/// Structure that holds requirements for external FairMQ data. Probably temporary.
struct FairMqInput {
  OutputSpec outputSpec;
  std::string channelConfig;
  std::string converterType;
};

/// Structure that holds QC task requirements for sampled data.
struct QcTaskConfiguration {
  std::string name;
  std::vector<FairMqInput> desiredFairMqData; // for temporary feature
  std::vector<InputSpec> desiredDataSpecs;
  header::DataHeader::SubSpecificationType subSpec;
  double fractionOfDataToSample;
  std::string dispatcherType;
  std::string fairMqOutputChannelConfig;
};
using QcTaskConfigurations = std::vector<QcTaskConfiguration>;

/// Structure that holds general data sampling infrastructure configuration
struct InfrastructureConfig {
  bool enableTimePipeliningDispatchers;
  bool enableParallelDispatchers;
  bool enableProxy;

  InfrastructureConfig()
    : enableTimePipeliningDispatchers(false), enableParallelDispatchers(false), enableProxy(false){};
};

} // namespace DataSamplingConfig

} // namespace framework
} // namespace o2

#endif // ALICEO2_DATASAMPLINGCONFIG_H
