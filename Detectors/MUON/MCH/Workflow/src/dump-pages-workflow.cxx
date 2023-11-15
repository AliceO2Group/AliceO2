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

///
/// \file    cru-page-reader-workflow.cxx
/// \author  Andrea Ferrero
///
/// \brief This is an executable that reads a data file from disk and sends the individual CRU pages via DPL.
///
/// This is an executable that reads a data file from disk and sends the individual CRU pages via the Data Processing Layer.
/// It can be used as a data source for O2 development. For example, one can do:
/// \code{.sh}
/// o2-mch-cru-page-reader-workflow --infile=some_data_file | o2-mch-raw-to-digits-workflow
/// \endcode
///

#include <random>
#include <iostream>
#include <queue>
#include <fstream>
#include <stdexcept>
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/runDataProcessing.h"

#include "DPLUtils/DPLRawParser.h"
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::raw;

namespace o2
{
namespace mch
{
namespace raw
{

using RDH = o2::header::RDHAny;

class DumpPagesTask
{
 public:
  DumpPagesTask(std::string spec) : mInputSpec(spec) {}
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the input file and other options from the context
    LOG(info) << "initializing pager dumper";

    auto outputFileName = ic.options().get<std::string>("outfile");
    mOutputFile.open(outputFileName, std::ios::binary);
    if (!mOutputFile.is_open()) {
      throw std::invalid_argument("Cannot open output file \"" + outputFileName + "\"");
    }

    auto stop = [this]() {
      /// close the input file
      LOG(info) << "stop file reader";
      this->mOutputFile.close();
    };
    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    // get the input buffer
    auto& inputs = pc.inputs();
    DPLRawParser parser(inputs, o2::framework::select(mInputSpec.c_str()));
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      auto const* raw = it.raw();
      if (!raw) {
        continue;
      }
      size_t payloadSize = it.size();

      mOutputFile.write(reinterpret_cast<const char*>(raw), sizeof(RDH) + payloadSize);
    }
  }

 private:
  std::string mInputSpec{"TF:MCH/RAWDATA"}; /// selection string for the input data
  std::ofstream mOutputFile{};              ///< input file
};

//_________________________________________________________________________________________________
// clang-format off
o2::framework::DataProcessorSpec getDumpPagesSpec(const char* specName)
{
  auto inputs = o2::framework::select("TF:MCH/RAWDATA");
  //o2::mch::raw::DumpPagesTask task("TF:MCH/RAWDATA");
  return DataProcessorSpec{
    specName,
    inputs,
    Outputs{},
    //AlgorithmSpec{adaptFromTask<o2::mch::raw::DumpPagesTask>(std::move(task))},
    AlgorithmSpec{adaptFromTask<o2::mch::raw::DumpPagesTask>("TF:MCH/RAWDATA")},
    Options{{"outfile", VariantType::String, "data.raw", {"output file name"}}}};
}
// clang-format on

} // end namespace raw
} // end namespace mch
} // end namespace o2

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext&)
{
  WorkflowSpec specs;

  // The producer to generate some data in the workflow
  DataProcessorSpec producer = mch::raw::getDumpPagesSpec("mch-page-dumper");
  specs.push_back(producer);

  return specs;
}
