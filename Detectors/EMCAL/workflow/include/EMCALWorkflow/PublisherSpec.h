// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#include "DataFormatsEMCAL/Cell.h"
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
  std::string defaultFileName;
  BranchOptionConfig databranch;
  BranchOptionConfig triggerrecordbranch;
  BranchOptionConfig mcbranch;
  OutputSpec dataoutput;
  OutputSpec triggerrecordoutput;
  OutputSpec mcoutput;
};

template <typename T = void>
framework::DataProcessorSpec getPublisherSpec(PublisherConf const& config, uint32_t subspec = 0, bool propagateMC = true)
{
  using Reader = o2::framework::RootTreeReader;
  using Output = o2::framework::Output;
  using TriggerInputType = std::vector<o2::emcal::TriggerRecord>;
  auto dto = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.dataoutput);
  auto tro = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.triggerrecordoutput);
  auto mco = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.mcoutput);

  // a creator callback for the actual reader instance
  auto creator = [dto, tro, mco, subspec, propagateMC](const char* treename, const char* filename, int nofEvents, Reader::PublishingMode publishingMode, const char* branchname, const char* triggerbranchname, const char* mcbranchname) {
    constexpr auto persistency = o2::framework::Lifetime::Timeframe;
    if (propagateMC) {
      return std::make_shared<Reader>(treename,
                                      filename,
                                      nofEvents,
                                      publishingMode,
                                      Output{mco.origin, mco.description, subspec, persistency},
                                      mcbranchname,
                                      Reader::BranchDefinition<T>{Output{dto.origin, dto.description, subspec, persistency}, branchname},
                                      Reader::BranchDefinition<TriggerInputType>{Output{tro.origin, tro.description, subspec, persistency}, triggerbranchname});
    } else {
      return std::make_shared<Reader>(treename,
                                      filename,
                                      nofEvents,
                                      publishingMode,
                                      Reader::BranchDefinition<T>{Output{dto.origin, dto.description, subspec, persistency}, branchname},
                                      Reader::BranchDefinition<TriggerInputType>{Output{tro.origin, tro.description, subspec, persistency}, triggerbranchname});
    }
  };

  return createPublisherSpec(config, subspec, propagateMC, creator);
}

inline framework::DataProcessorSpec getCellReaderSpec(bool propagateMC)
{
  using cellInputType = std::vector<o2::emcal::Cell>;
  return getPublisherSpec<cellInputType>(PublisherConf{"emcal-cell-reader",
                                                       "o2sim",
                                                       "emccells.root",
                                                       {"cellbranch", "EMCALCell", "Cell branch"},
                                                       {"celltriggerbranch", "EMCALCellTRGR", "Trigger record branch"},
                                                       {"mcbranch", "EMCALCellMCTruth", "MC label branch"},
                                                       o2::framework::OutputSpec{"EMC", "CELLS"},
                                                       o2::framework::OutputSpec{"EMC", "CELLSTRGR"},
                                                       o2::framework::OutputSpec{"EMC", "CELLSMCTR"}},
                                         0,
                                         propagateMC);
}

namespace workflow_reader
{
using Reader = o2::framework::RootTreeReader;
using Creator = std::function<std::shared_ptr<Reader>(const char*, const char*, int, Reader::PublishingMode, const char*, const char*, const char*)>;
} // namespace workflow_reader

framework::DataProcessorSpec createPublisherSpec(PublisherConf const& config, uint32_t subspec, bool propagateMC, workflow_reader::Creator creator);

} // namespace emcal
} // end namespace o2
