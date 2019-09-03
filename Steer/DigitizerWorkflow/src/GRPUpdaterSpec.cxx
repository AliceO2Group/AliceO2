// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief  Processor to modify detectors triggered/continous RO status in the simulated GRP

#include "GRPUpdaterSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "Framework/Lifetime.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Headers/DataHeader.h"
#include <TFile.h>
#include <FairLogger.h>
#include <memory> // for make_shared, make_unique, unique_ptr
#include <string>

using namespace o2::framework;

namespace o2
{
namespace parameters
{

using SubSpecificationType = framework::DataAllocator::SubSpecificationType;

static std::vector<o2::detectors::DetID> sDetList;

class GRPDPLUpdatedTask
{
  using GRP = o2::parameters::GRPObject;

 public:
  void init(framework::InitContext& ic)
  {
    // nothing special to be set up
  }

  void run(framework::ProcessingContext& pc)
  {
    const std::string inputGRP = "o2sim_grp.root";
    const std::string grpName = "GRP";
    if (mFinished) {
      return;
    }

    TFile flGRP(inputGRP.c_str(), "update");
    if (flGRP.IsZombie()) {
      LOG(ERROR) << "Failed to open  in update mode " << inputGRP;
      return;
    }
    std::unique_ptr<GRP> grp(static_cast<GRP*>(flGRP.GetObjectChecked(grpName.c_str(), GRP::Class())));
    for (auto det : sDetList) { // get readout mode data from different detectors
      auto roMode = pc.inputs().get<o2::parameters::GRPObject::ROMode>(det.getName());
      if (!(roMode & o2::parameters::GRPObject::PRESENT)) {
        LOG(ERROR) << "Detector " << det.getName() << " is read out while processor set ABSENT";
        continue;
      }
      grp->setDetROMode(det, roMode);
    }
    LOG(INFO) << "Updated GRP in " << inputGRP << " for detectors RO mode";
    grp->print();
    flGRP.WriteObjectAny(grp.get(), grp->Class(), grpName.c_str());
    flGRP.Close();
    mFinished = true;

    pc.services().get<ControlService>().readyToQuit(false);
  }

 private:
  bool mFinished = false;
};

/// create the processor spec
o2::framework::DataProcessorSpec getGRPUpdaterSpec(const std::vector<o2::detectors::DetID>& detList)
{
  sDetList = detList;
  static constexpr std::array<o2::header::DataOrigin, o2::detectors::DetID::nDetectors> sOrigins = {
    o2::header::gDataOriginITS, o2::header::gDataOriginTPC, o2::header::gDataOriginTRD,
    o2::header::gDataOriginTOF, o2::header::gDataOriginPHS, o2::header::gDataOriginCPV,
    o2::header::gDataOriginEMC, o2::header::gDataOriginHMP, o2::header::gDataOriginMFT,
    o2::header::gDataOriginMCH, o2::header::gDataOriginMID, o2::header::gDataOriginZDC,
    o2::header::gDataOriginFT0, o2::header::gDataOriginFV0, o2::header::gDataOriginFDD,
    o2::header::gDataOriginACO};

  // prepare specs
  std::vector<InputSpec> inputs;
  for (const auto det : detList) {
    inputs.emplace_back(InputSpec{det.getName(), sOrigins[det], "ROMode",
                                  static_cast<SubSpecificationType>(0 /*det.second*/), Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    "GRPUpdater",
    inputs, // input status from each detector
    {},     // no output
    AlgorithmSpec{adaptFromTask<GRPDPLUpdatedTask>()},
    Options{/* for the moment no options */}};
}

} // end namespace parameters
} // end namespace o2
