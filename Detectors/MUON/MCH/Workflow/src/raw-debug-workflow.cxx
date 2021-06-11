// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    raw-parser.cxx
/// \author  Andrea Ferrero
///
/// \brief This is an executable that receives the TimeFrames from the raw proxy and prints the sequence of RDHs.
///
/// This is an executable that receives the TimeFrames from the raw proxy and prints the sequence of RDHs.
/// Useful for debugging the DPL workflows involving input RAW data.
///

#include <iostream>
#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"

#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"dataspec", VariantType::String, "TF:MCH/RAWDATA", {"selection string for the input data"}});
}

#include "Framework/runDataProcessing.h"

using namespace o2;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

namespace o2
{
namespace mch
{
namespace raw
{

//=======================
// Data parser
class DataParserTask
{
 public:
  DataParserTask(std::string spec) : mInputSpec(spec) {}

  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
  }

  void decodeBuffer(gsl::span<const std::byte> page){};

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    // get the input buffer
    DPLRawParser parser(pc.inputs(), o2::framework::select(mInputSpec.c_str()));

    int nRDH = 0;

    const std::byte* raw = nullptr;
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      raw = reinterpret_cast<const std::byte*>(it.raw());
      if (!raw) {
        continue;
      }

      auto* rdh = reinterpret_cast<const RDH*>(raw);

      if (nRDH == 0) {
        std::cout << std::endl
                  << "---------------" << std::endl;
        o2::raw::RDHUtils::printRDH(rdh);
        //std::cout << "......." << std::endl;
      }
      nRDH += 1;
    }

    if (false && raw) {
      auto* rdh = reinterpret_cast<const RDH*>(raw);
      o2::raw::RDHUtils::printRDH(rdh);
    }
    std::cout << "---------------" << std::endl;
  }

 private:
  std::string mInputSpec;
};

} // namespace raw
} // namespace mch
} // end namespace o2

WorkflowSpec defineDataProcessing(const ConfigContext& config)
{
  auto inputSpec = config.options().get<std::string>("dataspec");

  WorkflowSpec specs;

  o2::mch::raw::DataParserTask task(inputSpec);
  DataProcessorSpec parser{
    "RawParser",
    o2::framework::select(inputSpec.c_str()),
    Outputs{},
    AlgorithmSpec{adaptFromTask<o2::mch::raw::DataParserTask>(std::move(task))},
    Options{}};

  specs.push_back(parser);

  return specs;
}
