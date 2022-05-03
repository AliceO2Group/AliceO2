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

/// @brief  Processor to modify detectors triggered/continous RO status in the simulated GRP

#include "GRPUpdaterSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "Framework/Lifetime.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include <DataFormatsParameters/GRPLHCIFData.h>
#include "CommonUtils/NameConf.h"
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

 public:
  GRPDPLUpdatedTask(const std::string& prefix) : mPrefix{prefix} {}

  void init(framework::InitContext& ic)
  {
    // nothing special to be set up
  }

  void run(framework::ProcessingContext& pc)
  {
    const std::string grpName{o2::base::NameConf::CCDBOBJECT};
    auto grpFileName = o2::base::NameConf::getGRPFileName(mPrefix);
    auto grpECSFileName = o2::base::NameConf::getGRPECSFileName(mPrefix);
    // a standardized semaphore convention --> taking the current execution path should be enough
    // (the user enables this via O2_USEGRP_SEMA environment)
    boost::interprocess::named_semaphore* sem = nullptr;
    std::string semhashedstring;
    try {
      const auto semname = std::filesystem::current_path().string() + grpECSFileName;
      std::hash<std::string> hasher;
      semhashedstring = "alice_grp_" + std::to_string(hasher(semname)).substr(0, 16);
      sem = new boost::interprocess::named_semaphore(boost::interprocess::open_or_create_t{}, semhashedstring.c_str(), 1);
    } catch (std::exception e) {
      LOG(warn) << "Could not setup GRP semaphore; Continuing without";
      sem = nullptr;
    }
    try {
      if (sem) {
        sem->wait(); // wait until we can enter (no one else there)
      }

      auto postSem = [sem, &semhashedstring] {
        if (sem) {
          sem->post();
          if (sem->try_wait()) {
            // if nobody else is waiting remove the semaphore resource
            sem->post();
            boost::interprocess::named_semaphore::remove(semhashedstring.c_str());
          }
          delete sem;
        }
      };
      updateECSData<o2::parameters::GRPECSObject>(grpECSFileName, pc);
      updateECSData<o2::parameters::GRPObject>(grpFileName, pc); // RS FIXME: suppress once we completely switch to GRPs triplet
      updateLHCIFData(pc);
      postSem();
    } catch (boost::interprocess::interprocess_exception e) {
      LOG(error) << "Caught semaphore exception " << e.what();
    }
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

 private:
  template <typename GRP>
  void updateECSData(const std::string& grpFileName, framework::ProcessingContext& pc)
  {
    const std::string grpName{o2::base::NameConf::CCDBOBJECT};
    TFile flGRP(grpFileName.c_str(), "update");
    if (flGRP.IsZombie()) {
      LOG(error) << "Failed to open in update mode " << grpFileName;
      return;
    }
    std::unique_ptr<GRP> grp(static_cast<GRP*>(flGRP.GetObjectChecked(grpName.c_str(), GRP::Class())));
    for (auto det : sDetList) { // get readout mode data from different detectors
      auto roMode = pc.inputs().get<typename GRP::ROMode>(det.getName());
      if (!(roMode & GRP::PRESENT)) {
        LOG(error) << "Detector " << det.getName() << " is read out while processor set ABSENT";
        continue;
      }
      grp->setDetROMode(det, roMode);
    }
    grp->setIsMC(true);
    grp->setNHBFPerTF(o2::raw::HBFUtils::Instance().nHBFPerTF);
    LOG(info) << "Updated " << grpFileName << " with detectors RO modes";
    grp->print();
    flGRP.WriteObjectAny(grp.get(), grp->Class(), grpName.c_str());
    flGRP.Close();
  }

  void updateLHCIFData(framework::ProcessingContext& pc)
  {
    using GRPLHCIF = o2::parameters::GRPLHCIFData;
    auto grpFileName = o2::base::NameConf::getGRPLHCIFFileName(mPrefix);
    const std::string grpName{o2::base::NameConf::CCDBOBJECT};
    if (!std::filesystem::exists(grpFileName)) {
      LOGP(info, "GRPLHCIF file {} is absent, abandon setting bunch-filling", grpFileName);
      return;
    }
    TFile flGRP(grpFileName.c_str(), "update");
    if (flGRP.IsZombie()) {
      LOG(fatal) << "Failed to open in update mode " << grpFileName;
    }
    std::unique_ptr<GRPLHCIF> grp(static_cast<GRPLHCIF*>(flGRP.GetObjectChecked(grpName.c_str(), GRPLHCIF::Class())));
    grp->setBunchFillingWithTime(grp->getBeamEnergyPerZTime(), pc.inputs().get<o2::BunchFilling>("bunchfilling")); // borrow the time from the existing entry
    flGRP.WriteObjectAny(grp.get(), grp->Class(), grpName.c_str());
    flGRP.Close();
    LOG(info) << "Updated " << grpFileName << " with bunch filling";
  }

  std::string mPrefix = "o2sim";
};

/// create the processor spec
o2::framework::DataProcessorSpec getGRPUpdaterSpec(const std::string& prefix, const std::vector<o2::detectors::DetID>& detList)
{
  sDetList = detList;

  // prepare specs
  std::vector<InputSpec> inputs;
  for (const auto det : detList) {
    inputs.emplace_back(InputSpec{det.getName(), det.getDataOrigin(), "ROMode", 0, Lifetime::Timeframe});
  }
  inputs.emplace_back(InputSpec{"bunchfilling", "SIM", "BUNCHFILLING", 0, Lifetime::Timeframe});

  return DataProcessorSpec{
    "GRPUpdater",
    inputs, // input status from each detector
    {},     // no output
    AlgorithmSpec{adaptFromTask<GRPDPLUpdatedTask>(prefix)},
    Options{/* for the moment no options */}};
}

} // end namespace parameters
} // end namespace o2
