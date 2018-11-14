// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "HMPIDDigitWriterSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "HMPIDBase/Digit.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace hmpid
{

class HMPIDDPLDigitWriterTask
{
 public:
  void init(framework::InitContext& ic)
  {
    LOG(INFO) << "initializing HMPID digit writer";
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }

    LOG(INFO) << "Doing HMPID digit IO";
    auto digits = pc.inputs().get<std::vector<o2::hmpid::Digit>*>("hmpiddigits");
    LOG(INFO) << "HMPID received " << digits->size() << " digits";
    for (auto& digit : *digits) {
      LOG(INFO) << "Have digit with charge " << digit.getCharge();
    }

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(false);
    finished = true;
  }
};

o2::framework::DataProcessorSpec getHMPIDDigitWriterSpec()
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  return DataProcessorSpec{
    "HMPIDDigitWriternn",
    Inputs{ InputSpec{ "hmpiddigits", "HMP", "DIGITS", 0, Lifetime::Timeframe } },
    {},
    AlgorithmSpec{ adaptFromTask<HMPIDDPLDigitWriterTask>() },
    Options{
      { "digitFile", VariantType::String, "hmpiddigits.root", { "filename for HMPID digits" } } }
  };
}

} // end namespace hmpid
} // end namespace o2
