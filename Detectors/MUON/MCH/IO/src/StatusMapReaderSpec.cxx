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

#include "StatusMapReaderSpec.h"

#include <memory>
#include <string>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ControlService.h"
#include "Framework/InitContext.h"
#include "Framework/Lifetime.h"
#include "Framework/OutputSpec.h"
#include "Framework/ProcessingContext.h"
#include "Framework/Task.h"

#include "DPLUtils/RootTreeReader.h"
#include "CommonUtils/StringUtils.h"
#include "MCHStatus/StatusMap.h"

using namespace o2::framework;

namespace o2::mch
{

struct StatusMapReader {
  std::unique_ptr<RootTreeReader> mTreeReader;

  void init(InitContext& ic)
  {
    auto fileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")), ic.options().get<std::string>("infile"));
    mTreeReader = std::make_unique<RootTreeReader>(
      "o2sim",
      fileName.c_str(),
      -1,
      RootTreeReader::PublishingMode::Single,
      RootTreeReader::BranchDefinition<StatusMap>{Output{"MCH", "STATUSMAP", 0}, "statusmaps"});
  }

  void run(ProcessingContext& pc)
  {
    if (mTreeReader->next()) {
      (*mTreeReader)(pc);
    } else {
      pc.services().get<ControlService>().endOfStream();
    }
  }
};

DataProcessorSpec getStatusMapReaderSpec(const char* specName)
{
  return DataProcessorSpec{
    specName,
    Inputs{},
    Outputs{OutputSpec{{"statusmaps"}, "MCH", "STATUSMAP", 0, Lifetime::Timeframe}},
    adaptFromTask<StatusMapReader>(),
    Options{{"infile", VariantType::String, "mchstatusmaps.root", {"name of the input status map file"}},
            {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace o2::mch
