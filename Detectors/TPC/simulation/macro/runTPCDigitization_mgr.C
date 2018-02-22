/*
 * filterHits.C
 *
 *  Created on: Jan 30, 2018
 *      Author: swenzel
 */
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <functional>
#include <TPCSimulation/Digitizer.h>
#include <TPCSimulation/DigitizerTask.h>
#include <Steer/HitProcessingManager.h>
#include "ITSMFTSimulation/Hit.h"
#include <cassert>
#endif

void getHits(const o2::steer::RunContext& context, std::vector<std::vector<o2::TPC::HitGroup>*>& hitvectors,
             std::vector<TPCHitGroupID>& hitids,

             const char* branchname, float tmin /*NS*/, float tmax /*NS*/,
             std::function<float(float, float, float)>&& f)
{
  // f is some function taking event time + z of hit and returns final "digit" time

  // ingredients
  // *) tree

  auto br = context.getBranch(branchname);
  if (!br) {
    std::cerr << "No branch found\n";
    return;
  }

  auto& eventrecords = context.getEventRecords();

  auto nentries = br->GetEntries();
  hitvectors.resize(nentries, nullptr);

  // do the filtering
  for (int entry = 0; entry < nentries; ++entry) {
    if (tmin > f(eventrecords[entry].timeNS, 0, 0)) {
      continue;
    }
    if (tmax < f(eventrecords[entry].timeNS, 0, 250)) {
      break;
    }

    br->SetAddress(&hitvectors[entry]);
    br->GetEntry(entry);

    int groupid = -1;
    auto groups = hitvectors[entry];
    for (auto& singlegroup : *groups) {
      groupid++;
      const auto zmax = singlegroup.mZAbsMax;
      const auto zmin = singlegroup.mZAbsMin;
      assert(zmin <= zmax);
      // auto tof = singlegroup.
      float tmaxtrack = f(eventrecords[entry].timeNS, 0., zmin);
      float tmintrack = f(eventrecords[entry].timeNS, 0., zmax);
      std::cout << tmintrack << " & " << tmaxtrack << "\n";
      assert(tmaxtrack > tmintrack);
      if (tmin > tmaxtrack || tmax < tmintrack) {
        std::cout << "DISCARDING " << groupid << " OF ENTRY " << entry << "\n";
        continue;
      }
      // need to record index of the group
      hitids.emplace_back(entry, groupid);
    }
  }
}

// TPC hit selection lambda
auto fTPC = [](float tNS, float tof, float z) {
  // returns time in NS
  return tNS + o2::TPC::Digitizer::getTime(z) * 1000 + tof;
};

void runTPCDigitization(const o2::steer::RunContext& context)
{
  std::vector<std::vector<o2::TPC::HitGroup>*> hitvectorsleft;  // "TPCHitVector"
  std::vector<std::vector<o2::TPC::HitGroup>*> hitvectorsright; // "TPCHitVector"
  std::vector<o2::TPC::TPCHitGroupID> hitidsleft;               // "TPCHitIDs"
  std::vector<o2::TPC::TPCHitGroupID> hitidsright;

  o2::TPC::DigitizerTask task;

  const auto TPCDRIFT = 100000;
  for (int driftinterval = 0;; ++driftinterval) {
    auto tmin = driftinterval * TPCDRIFT;
    auto tmax = (driftinterval + 1) * TPCDRIFT;

    hitvectorsleft.clear();
    hitidsleft.clear();
    hitvectorsright.clear();
    hitidsright.clear();

    bool hasData = false;
    // loop over sectors
    for (int s = 0; s < 36; ++s) {
      task.setSector(s);
      std::stringstream sectornamestreamleft;
      sectornamestreamleft << "TPCHitsShiftedSector" << s;
      getHits(context, hitvectorsleft, hitidsleft, sectornamestreamleft.str().c_str(), tmin, tmax, fTPC);

      std::stringstream sectornamestreamright;
      sectornamestreamright << "TPCHitsShiftedSector" << Shifted(s);
      getHits(context, hitvectorsright, hitidsright, sectornamestreamright.str().c_str(), tmin, tmax, fTPC);

      task.setData(&hitvectorsleft, &hitvectorsright, &hitidsleft, &hitidsright, &context);
      task.Exec("");

      hasData |= hitids.size();
    }

    // condition to end:
    if (!hasData) {
      break;
    }
    task.FinishTask();
  }
}

void runITSDigitization(const o2::steer::RunContext& context)
{
  std::vector<o2::ITSMFT::Hit>* hitvector = nullptr;
  o2::ITS::DigitizerTask task(true);
  task.Init();
  auto br = context.getBranch("ITSHit");
  assert(br);
  br->SetAddress(&hitvector);
  for (int entry = 0; entry < context.getNEntries(); ++entry) {
    br->GetEntry(entry);
    task.setData(hitvector, &context);
    task.Exec("");
  }
  task.FinishTask();
}

void runTPCDigitization_mgr()
{
  auto& hitrunmgr = o2::steer::HitProcessingManager::instance();
  hitrunmgr.addInputFile("o2sim.root");
  hitrunmgr.registerRunFunction(runTPCDigitization);

  // hitrunmgr.registerRunFunction(runITSDigitization);
  using Data_t = std::vector<o2::ITSMFT::Hit>;

  o2::ITS::DigitizerTask task;
  TGeoManager::Import("O2geometry.root");
  task.Init();
  hitrunmgr.registerDefaultRunFunction(o2::steer::defaultRunFunction(task, "ITSHit"));

  hitrunmgr.run();
}
