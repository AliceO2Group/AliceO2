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

#include "MCHWorkflow/ErrorReaderSpec.h"

#include <memory>
#include <string>
#include <vector>

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
#include "MCHBase/Error.h"

using namespace o2::framework;

namespace o2::mch
{

struct ErrorReader {
  std::unique_ptr<RootTreeReader> mTreeReader;

  void init(InitContext& ic)
  {
    auto fileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")), ic.options().get<std::string>("infile"));
    mTreeReader = std::make_unique<RootTreeReader>(
      "o2sim",
      fileName.c_str(),
      -1,
      RootTreeReader::PublishingMode::Single,
      RootTreeReader::BranchDefinition<std::vector<Error>>{Output{"MCH", "ERRORS", 0}, "errors"});
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

DataProcessorSpec getErrorReaderSpec(const char* specName)
{
  return DataProcessorSpec{
    specName,
    Inputs{},
    Outputs{OutputSpec{{"errors"}, "MCH", "ERRORS", 0, Lifetime::Timeframe}},
    adaptFromTask<ErrorReader>(),
    Options{{"infile", VariantType::String, "mcherrors.root", {"name of the input error file"}},
            {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace o2::mch
