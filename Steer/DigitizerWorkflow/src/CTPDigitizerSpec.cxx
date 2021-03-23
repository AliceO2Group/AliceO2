// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CTPDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DataFormatsCTP/Digits.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "CTPSimulation/Digitizer.h"

#include <TChain.h>
#include <TStopwatch.h>

using namespace o2::framework;
namespace o2
{
namespace ctp
{
class CTPDPLDigitizerTask : public o2::base::BaseDPLDigitizer
{
  using GRP = o2::parameters::GRPObject;

 public:
  CTPDPLDigitizerTask() : o2::base::BaseDPLDigitizer(), mDigitizer() {}
  ~CTPDPLDigitizerTask() override = default;
  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }
    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    //context->initSimChains(o2::detectors::DetID::CTP, mSimChains);
    const bool withQED = context->isQEDProvided();
    auto& timesview = context->getEventRecords(withQED);
    // if there is nothing to do ... return
    if (timesview.size() == 0) {
      return;
    }

    TStopwatch timer;
    timer.Start();
    LOG(INFO) << "CALLING FT0 DIGITIZATION";
    for (const auto& coll : timesview) {
      mDigitizer.setInteractionRecord(coll);
      mDigitizer.process(mDigits);
      mDigitizer.flush(mDigits);
    }
    // send out to next stage
    pc.outputs().snapshot(Output{"CTP", "DIGITS", 0, Lifetime::Timeframe}, mDigits);
    timer.Stop();
    LOG(INFO) << "CTP Digitization took " << timer.CpuTime() << "s";
    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    mFinished = true;
  }

 protected:
  bool mFinished = false;
  std::vector<o2::ctp::CTPDigit> mDigits;

  Bool_t mContinuous = kFALSE;   ///< flag to do continuous simulation
  double mFairTimeUnitInNS = 1;  ///< Fair time unit in ns
  o2::ctp::Digitizer mDigitizer; ///< Digitizer
};
o2::framework::DataProcessorSpec getCTPDigitizerSpec(int channel, bool mctruth)
{
}
} // namespace ctp
} // namespace o2