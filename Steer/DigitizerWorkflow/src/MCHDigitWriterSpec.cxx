// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHDigitWriterSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "MCHBase/Digit.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace mch
{

class MCHDPLDigitWriterTask
{
 public:
  void init(framework::InitContext& ic)
  {
    LOG(INFO) << "initializing MCH digit writer";
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }

    LOG(INFO) << "Doing MCH digit IO";
    auto digits = pc.inputs().get<std::vector<o2::mch::Digit>*>("mchdigits");
    LOG(INFO) << "MCH received " << digits->size() << " digits";
    for (auto& digit : *digits) {
      LOG(INFO) << "Have digit with charge " << digit.getCharge();
    }

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(false);
    finished = true;
  }
};

o2::framework::DataProcessorSpec getMCHDigitWriterSpec()
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  return DataProcessorSpec{
    "MCHDigitWriternn",
    Inputs{ InputSpec{ "mchdigits", "MCH", "DIGITS", 0, Lifetime::Timeframe } },
    {},
    AlgorithmSpec{ adaptFromTask<MCHDPLDigitWriterTask>() },
    Options{
      { "digitFile", VariantType::String, "mchdigits.root", { "filename for MCH digits" } } }
  };
}

} // end namespace mch
} // end namespace o2
