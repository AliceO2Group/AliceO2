// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DPLUtils/RootTreeReader.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Output.h"
#include "Framework/OutputSpec.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
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
  BranchOptionConfig triggerrecordbranch;
  BranchOptionConfig mcbranch;
  OutputSpec dataoutput;
  OutputSpec triggerrecordoutput;
  OutputSpec mcoutput;
};

template <typename T = void>
framework::DataProcessorSpec getPublisherSpec(PublisherConf const& config, bool propagateMC = true)
{
  using Reader = o2::framework::RootTreeReader;
  using Output = o2::framework::Output;
  using TriggerInputType = std::vector<o2::emcal::TriggerRecord>;
  auto dto = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.dataoutput);
  auto tro = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.triggerrecordoutput);
  auto mco = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.mcoutput);

  // a creator callback for the actual reader instance
  auto creator = [dto, tro, mco, propagateMC](const char* treename, const char* filename, int nofEvents, Reader::PublishingMode publishingMode, const char* branchname, const char* triggerbranchname, const char* mcbranchname) {
    constexpr auto persistency = o2::framework::Lifetime::Timeframe;
    if (propagateMC) {
      return std::make_shared<Reader>(treename,
                                      filename,
                                      nofEvents,
                                      publishingMode,
                                      Output{mco.origin, mco.description, 0, persistency},
                                      mcbranchname,
                                      Reader::BranchDefinition<T>{Output{dto.origin, dto.description, 0, persistency}, branchname},
                                      Reader::BranchDefinition<TriggerInputType>{Output{tro.origin, tro.description, 0, persistency}, triggerbranchname});
    } else {
      return std::make_shared<Reader>(treename,
                                      filename,
                                      nofEvents,
                                      publishingMode,
                                      Reader::BranchDefinition<T>{Output{dto.origin, dto.description, 0, persistency}, branchname},
                                      Reader::BranchDefinition<TriggerInputType>{Output{tro.origin, tro.description, 0, persistency}, triggerbranchname});
    }
  };

  return createPublisherSpec(config, propagateMC, creator);
}

namespace workflow_reader
{
using Reader = o2::framework::RootTreeReader;
using Creator = std::function<std::shared_ptr<Reader>(const char*, const char*, int, Reader::PublishingMode, const char*, const char*, const char*)>;
} // namespace workflow_reader

framework::DataProcessorSpec createPublisherSpec(PublisherConf const& config, bool propagateMC, workflow_reader::Creator creator);

} // namespace emcal
} // end namespace o2
