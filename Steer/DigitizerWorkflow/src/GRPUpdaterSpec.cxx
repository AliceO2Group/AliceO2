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
#include "DetectorsRaw/HBFUtils.h"

// this is for some process synchronization, since
// we need to prevent writing concurrently to the same global GRP file
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <filesystem>
#include <unordered_map> // for the hashing utility

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
  GRPDPLUpdatedTask(const std::string& grpfilename) : mGRPFileName{grpfilename} {}

  void init(framework::InitContext& ic)
  {
    // nothing special to be set up
  }

  void run(framework::ProcessingContext& pc)
  {
    const std::string grpName = "GRP";

    // a standardized semaphore convention --> taking the current execution path should be enough
    // (the user enables this via O2_USEGRP_SEMA environment)
    bool use_sema = false;
    boost::interprocess::named_semaphore* sem = nullptr;
    if (auto semaname = getenv("O2_USEGRP_SEMA")) {
      try {
        const auto semname = std::filesystem::current_path().string() + mGRPFileName;
        std::hash<std::string> hasher;
        const auto semhashedstring = "alice_grp_" + std::to_string(hasher(semname));
        sem = new boost::interprocess::named_semaphore(boost::interprocess::open_or_create_t{}, semhashedstring.c_str(), 1);
      } catch (std::exception e) {
        LOG(WARN) << "Exception occurred during GRP semaphore setup; Continuing without";
        sem = nullptr;
      }
    }
    try {
      if (sem) {
        sem->wait(); // wait until we can enter (no one else there)
      }

      auto postSem = [sem] {
        if (sem) {
          sem->post();
          delete sem;
        }
      };

      TFile flGRP(mGRPFileName.c_str(), "update");
      if (flGRP.IsZombie()) {
        LOG(ERROR) << "Failed to open in update mode " << mGRPFileName;
        postSem();
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
      grp->setFirstOrbit(o2::raw::HBFUtils::Instance().orbitFirst);
      grp->setNHBFPerTF(o2::raw::HBFUtils::Instance().nHBFPerTF);
      LOG(INFO) << "Updated GRP in " << mGRPFileName << " for detectors RO mode and 1st orbit of the run";
      grp->print();
      flGRP.WriteObjectAny(grp.get(), grp->Class(), grpName.c_str());
      flGRP.Close();

      postSem();
    } catch (boost::interprocess::interprocess_exception e) {
      LOG(ERROR) << "Caught semaphore exception " << e.what();
    }
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

 private:
  std::string mGRPFileName = "o2sim_grp.root";
};

/// create the processor spec
o2::framework::DataProcessorSpec getGRPUpdaterSpec(const std::string& grpfilename, const std::vector<o2::detectors::DetID>& detList)
{
  sDetList = detList;

  // prepare specs
  std::vector<InputSpec> inputs;
  for (const auto det : detList) {
    inputs.emplace_back(InputSpec{det.getName(), det.getDataOrigin(), "ROMode",
                                  static_cast<SubSpecificationType>(0), Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    "GRPUpdater",
    inputs, // input status from each detector
    {},     // no output
    AlgorithmSpec{adaptFromTask<GRPDPLUpdatedTask>(grpfilename)},
    Options{/* for the moment no options */}};
}

} // end namespace parameters
} // end namespace o2
