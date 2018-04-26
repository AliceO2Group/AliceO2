// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TPCDriftTimeDigitizerSpec.h"
#include <FairMQLogger.h>
#include <TMessage.h> // object serialization
#include <cassert>
#include <cstring> // memcpy
#include <memory>  // std::unique_ptr
#include <string>  // std::string
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "Steer/HitProcessingManager.h"
#include <TPCSimulation/Digitizer.h>
#include <TPCSimulation/DigitizerTask.h>
#include <functional>
#include "ITSMFTSimulation/Hit.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/HitDriftFilter.h"
#include "TStopwatch.h"
#include <sstream>
#include <algorithm>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
namespace o2
{
namespace steer
{

DataProcessorSpec getTPCDriftTimeDigitizer(int sector, int channel, bool cachehits)
{
  TChain* simChain = new TChain("o2sim");
  std::stringstream branchnamestreamleft;
  branchnamestreamleft << "TPCHitsShiftedSector" << int(o2::TPC::Sector::getLeft(o2::TPC::Sector(sector)));
  std::string branchnameleft = branchnamestreamleft.str();
  std::stringstream branchnamestreamright;
  branchnamestreamright << "TPCHitsShiftedSector" << sector;
  std::string branchnameright = branchnamestreamright.str();

  auto digitizertask = std::make_shared<o2::TPC::DigitizerTask>();
  digitizertask->Init2();
  // the task takes the ownership of digit array + mc truth array
  // TODO: make this clear in the API

  auto digitArray = std::make_shared<std::vector<o2::TPC::Digit>>();
  auto mcTruthArray = std::make_shared<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
  digitizertask->setOutputData(digitArray.get(), mcTruthArray.get());

  auto doit = [simChain, branchnameleft, branchnameright, sector, digitizertask, digitArray,
               mcTruthArray](ProcessingContext& pc) {
    static int callcounter = 0;
    callcounter++;

    // ===| open file and register branches |=====================================
    // this is done at the moment for each worker function invocation
    // TODO: make this nicer or let the write service be handled outside
    auto file = std::make_unique<TFile>(Form("tpc_digi_%i_instance%i.root", sector, callcounter), "recreate");
    auto outtree = std::make_unique<TTree>("o2sim", "TPC digits");
    outtree->SetDirectory(file.get());
    auto digitArrayRaw = digitArray.get();
    auto mcTruthArrayRaw = mcTruthArray.get();
    auto digitBranch = outtree->Branch(Form("TPCDigit_%i", sector), &digitArrayRaw);
    auto mcTruthBranch = outtree->Branch(Form("TPCDigitMCTruth_%i", sector), &mcTruthArrayRaw);

    // obtain collision contexts
    auto dataref = pc.inputs().get("timeinput");
    auto header = o2::header::get<const o2::header::DataHeader*>(dataref.header);

    auto context = pc.inputs().get<o2::steer::RunContext>("timeinput");
    auto timesview = context->getEventRecords();
    LOG(DEBUG) << "GOT " << timesview.size() << " COLLISSION TIMES";

    // if there is nothing ... return
    if (timesview.size() == 0) {
      return;
    }

    TStopwatch timer;
    timer.Start();

    // detect number of possible drift times (remember that a drift
    // time is convenient unit for digit pileup), any multiple of this
    // unit should also be ok
    const auto TPCDRIFT = 100000;
    double maxtime = 0;
    for (auto e : timesview) {
      maxtime = std::max(maxtime, e.timeNS);
    }

    // minimum 2 drifts is a safe bet; an electron might
    // need 1 full drift and might hence land in the second drift time
    auto ndrifts = 2 + (int)(maxtime / TPCDRIFT);

    std::vector<std::vector<o2::TPC::HitGroup>*> hitvectorsleft;  // "TPCHitVector"
    std::vector<o2::TPC::TPCHitGroupID> hitidsleft;               // "TPCHitIDs"
    std::vector<std::vector<o2::TPC::HitGroup>*> hitvectorsright; // "TPCHitVector"
    std::vector<o2::TPC::TPCHitGroupID> hitidsright;              // "TPCHitIDs"
    hitvectorsleft.resize(timesview.size(), nullptr);
    hitvectorsright.resize(timesview.size(), nullptr);

    // have to do a loop over drift times
    for (int drift = 1; drift <= ndrifts; ++drift) {
      auto starttime = (drift - 1) * TPCDRIFT;
      auto endtime = drift * TPCDRIFT;
      LOG(DEBUG) << "STARTTIME " << starttime << " ENDTIME " << endtime;
      digitizertask->setStartTime(starttime);
      digitizertask->setEndTime(endtime);

      // obtain candidate hit(ids) for this time range --> left
      o2::TPC::getHits(*simChain, timesview, hitvectorsleft, hitidsleft, branchnameleft.c_str(), starttime, endtime,
                       o2::TPC::calcDriftTime);
      // --> right
      o2::TPC::getHits(*simChain, timesview, hitvectorsright, hitidsright, branchnameright.c_str(), starttime, endtime,
                       o2::TPC::calcDriftTime);

      LOG(DEBUG) << "DRIFTTIME " << drift << " SECTOR " << sector << " : SELECTED LEFT " << hitidsleft.size() << " IDs"
                 << " SELECTED RIGHT " << hitidsright.size();

      // invoke digitizer if anything to digitize within this drift interval
      if (hitidsleft.size() > 0 || hitidsright.size() > 0) {
        digitizertask->setData(&hitvectorsleft, &hitvectorsright, &hitidsleft, &hitidsright, context.get());
        digitizertask->setupSector(sector);
        digitizertask->Exec2("");

        // write digits + MC truth
        outtree->Fill();
      }
    }
    outtree->SetDirectory(file.get());
    file->Write();
    timer.Stop();
    LOG(INFO) << "Digitization took " << timer.CpuTime() << "s";

    pc.services().get<ControlService>().readyToQuit(false);
  };

  // init function return a lambda taking a ProcessingContext
  auto initIt = [simChain, doit](InitContext& ctx) {
    // setup the input chain
    simChain->AddFile("o2sim.root");
    return doit;
  };

  std::stringstream id;
  id << "TPCDigitizer" << sector;
  return DataProcessorSpec{
    id.str().c_str(), Inputs{ InputSpec{ "timeinput", "SIM", "EVENTTIMES", static_cast<SubSpecificationType>(channel),
                                         Lifetime::Timeframe } },
    Outputs{
      // define channel by triple of (origin, type id of data to be sent on this channel, subspecification)
    },
    AlgorithmSpec{ initIt }, Options{ /*{ "simFile", VariantType::String, "o2sim.root", { "Sim input filename" } }*/ }
  };
}
}
}
