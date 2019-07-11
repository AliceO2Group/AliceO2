// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataProcessorSpec.h"
#include "Framework/OutputSpec.h"
#include <string>
#include <vector>

namespace o2
{

namespace emcal
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
};

framework::DataProcessorSpec getPublisherSpec(PublisherConf const& config, bool propagateMC = true);

} // namespace emcal
} // end namespace o2
