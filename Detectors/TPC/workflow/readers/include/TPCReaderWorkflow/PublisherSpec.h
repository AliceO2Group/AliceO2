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
#include "Framework/DataSpecUtils.h"
#include "Framework/Output.h"
#include "DPLUtils/RootTreeReader.h"
#include <vector>
#include <string>
#include <functional>

namespace o2
{
namespace tpc
{

using OutputSpec = framework::OutputSpec;
using Reader = o2::framework::RootTreeReader;

struct PublisherConf {
  struct BranchOptionConfig {
    std::string option;
    std::string defval;
    std::string help;
  };

  std::string processName;
  std::string defaultFileName;
  std::string defaultTreeName;
  BranchOptionConfig databranch;
  BranchOptionConfig mcbranch;
  OutputSpec dataoutput;
  OutputSpec mcoutput;
  std::vector<int> tpcSectors;
  std::vector<int> outputIds;
  Reader::SpecialPublishHook* hook = nullptr;
};

/// create a processor spec
/// read data from multiple tree branches from ROOT file and publish
template <typename T = void>
framework::DataProcessorSpec getPublisherSpec(PublisherConf const& config, bool propagateMC = true)
{
  using Reader = o2::framework::RootTreeReader;
  using Output = o2::framework::Output;
  auto dto = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.dataoutput);
  auto mco = o2::framework::DataSpecUtils::asConcreteDataTypeMatcher(config.mcoutput);

  // a creator callback for the actual reader instance
  auto creator = [dto, mco, propagateMC](const char* treename, const char* filename, int nofEvents, Reader::PublishingMode publishingMode, o2::header::DataHeader::SubSpecificationType subSpec, const char* branchname, const char* mcbranchname, Reader::SpecialPublishHook* publishhook = nullptr) {
    constexpr auto persistency = o2::framework::Lifetime::Timeframe;
    if (propagateMC) {
      return std::make_shared<Reader>(treename,
                                      filename,
                                      nofEvents,
                                      publishingMode,
                                      Output{mco.origin, mco.description, subSpec, persistency},
                                      mcbranchname,
                                      Reader::BranchDefinition<T>{Output{dto.origin, dto.description, subSpec, persistency}, branchname},
                                      publishhook);
    } else {
      return std::make_shared<Reader>(treename,
                                      filename,
                                      nofEvents,
                                      publishingMode,
                                      Reader::BranchDefinition<T>{Output{dto.origin, dto.description, subSpec, persistency}, branchname},
                                      publishhook);
    }
  };

  return createPublisherSpec(config, propagateMC, creator);
}

namespace workflow_reader
{
using Reader = o2::framework::RootTreeReader;
using Creator = std::function<std::shared_ptr<Reader>(const char*, const char*, int, Reader::PublishingMode, o2::header::DataHeader::SubSpecificationType, const char*, const char*, Reader::SpecialPublishHook*)>;
} // namespace workflow_reader

framework::DataProcessorSpec createPublisherSpec(PublisherConf const& config, bool propagateMC, workflow_reader::Creator creator);

} // end namespace tpc
} // end namespace o2
