// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitReaderSpec.h
/// @author Matthias Richter
/// @since  2018-12-06
/// @brief  Processor spec for a reader of TPC data from ROOT file

#include "Framework/DataProcessorSpec.h"
#include "Framework/OutputSpec.h"
#include <vector>
#include <string>

namespace o2
{
namespace TPC
{

using OutputSpec = framework::OutputSpec;

struct PublisherConf {
  struct BranchOptionConfig {
    std::string option;
    std::string defval;
    std::string help;
  };

  std::string processName;
  std::string defaultTreeName;
  BranchOptionConfig databranch;
  BranchOptionConfig mcbranch;
  OutputSpec dataoutput;
  OutputSpec mcoutput;
  std::vector<int> tpcSectors;
  std::vector<int> outputIds;
};

/// create a processor spec
/// read data from multiple tree branches from ROOT file and publish
framework::DataProcessorSpec getPublisherSpec(PublisherConf const& config, bool propagateMC = true);

} // end namespace TPC
} // end namespace o2
