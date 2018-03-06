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

#include <vector>
#include <string>

#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Headers/DataHeader.h"

namespace o2
{
namespace framework
{

namespace DataSamplingConfig
{
using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;

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
  SubSpecificationType subSpec;
  double fractionOfDataToSample;
  std::string fairMqOutputChannelConfig;
};

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
