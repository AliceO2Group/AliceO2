/*
 * filterHits.C
 *
 *  Created on: Jan 30, 2018
 *      Author: swenzel
 */
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <functional>
#include "TStopwatch.h"
#include "TPCSimulation/Point.h"
#include "TPCBase/Sector.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/DigitizerTask.h"
#include "Steer/HitProcessingManager.h"
#include "FairSystemInfo.h"
#endif

void getHits(const o2::steer::RunContext& context, std::vector<std::vector<o2::TPC::HitGroup>*>& hitvectors,
             std::vector<o2::TPC::TPCHitGroupID>& hitids, const char* branchname, float tmin /*NS*/, float tmax /*NS*/,
             std::function<float(float, float, float)>&& f)
{
  // f is some function taking event time + z of hit and returns final "digit" time
  auto br = context.getBranch(branchname);
  if (!br) {
    std::cerr << "No branch found\n";
    return;
  }

  auto& eventrecords = context.getEventRecords();

  auto nentries = br->GetEntries();
  hitvectors.resize(nentries, nullptr);
  hitids.clear();

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
      auto zmax = singlegroup.mZAbsMax;
      auto zmin = singlegroup.mZAbsMin;
      // in case of secondaries, the time ordering may be reversed
      if (zmax < zmin) {
        std::swap(zmax, zmin);
      }
      assert(zmin <= zmax);
      float tmaxtrack = f(eventrecords[entry].timeNS, 0., zmin);
      float tmintrack = f(eventrecords[entry].timeNS, 0., zmax);
      assert(tmaxtrack >= tmintrack);
      if (tmin > tmaxtrack || tmax < tmintrack) {
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
  return tNS + o2::TPC::ElectronTransport::getDriftTime(z) * 1000 + tof;
};

void getHits(const o2::steer::RunContext& context, std::vector<std::vector<o2::TPC::HitGroup>*>& hitvectors,
             std::vector<o2::TPC::TPCHitGroupID>& hitids, const char* branchname, const int iEvent)
{
  auto br = context.getBranch(branchname);
  if (!br) {
    std::cerr << "No branch found\n";
    return;
  }

  auto& eventrecords = context.getEventRecords();

  auto nentries = br->GetEntries();
  hitvectors.resize(nentries, nullptr);
  hitids.clear();

  br->SetAddress(&hitvectors[iEvent]);
  br->GetEntry(iEvent);

  int groupid = -1;
  auto groups = hitvectors[iEvent];
  for (auto& singlegroup : *groups) {
    groupid++;
    hitids.emplace_back(iEvent, groupid);
  }
}

void runTPCDigitizationChunkwise(const o2::steer::RunContext& context)
{
  std::vector<std::vector<o2::TPC::HitGroup>*> hitvectorsleft;  // "TPCHitVector"
  std::vector<std::vector<o2::TPC::HitGroup>*> hitvectorsright; // "TPCHitVector"
  std::vector<o2::TPC::TPCHitGroupID> hitidsleft;               // "TPCHitIDs"
  std::vector<o2::TPC::TPCHitGroupID> hitidsright;              // "TPCHitIDs

  o2::TPC::DigitizerTask task;
  task.Init2();

  // ===| open file and register branches |=====================================
  std::unique_ptr<TFile> file = std::unique_ptr<TFile>(TFile::Open("tpc_digi.root", "recreate"));
  TTree outtree("o2sim", "TPC digits");

  FairSystemInfo sysinfo;

  using digitType = std::vector<o2::TPC::Digit>;
  using mcType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  std::array<digitType*, o2::TPC::Sector::MAXSECTOR> digitArrays;
  std::array<mcType*, o2::TPC::Sector::MAXSECTOR> mcTruthArrays;
  std::array<TBranch*, o2::TPC::Sector::MAXSECTOR> digitBranches;
  std::array<TBranch*, o2::TPC::Sector::MAXSECTOR> mcTruthBranches;

  for (int sector = 0; sector < o2::TPC::Sector::MAXSECTOR; ++sector) {
    digitArrays[sector] = new digitType;
    digitBranches[sector] = outtree.Branch(Form("TPCDigit_%i", sector), &digitArrays[sector]);

    // Register MC Truth container
    mcTruthArrays[sector] = new mcType;
    mcTruthBranches[sector] = outtree.Branch(Form("TPCDigitMCTruth_%i", sector), &mcTruthArrays[sector]);
  }

  const auto TPCDRIFT = 100000;
  for (int driftinterval = 0;; ++driftinterval) {
    auto tmin = driftinterval * TPCDRIFT;
    auto tmax = (driftinterval + 1) * TPCDRIFT;

    std::cout << "============================\n";
    std::cout << "Drift interval " << driftinterval << " times: " << tmin << " - " << tmax << "\n";

    hitvectorsleft.clear();
    hitidsleft.clear();
    hitvectorsright.clear();
    hitidsright.clear();

    bool hasData = false;
    // loop over sectors
    for (int s = 0; s < o2::TPC::Sector::MAXSECTOR; ++s) {
      task.setupSector(s);
      task.setOutputData(digitArrays[s], mcTruthArrays[s]);

      std::stringstream sectornamestreamleft;
      sectornamestreamleft << "TPCHitsShiftedSector" << int(o2::TPC::Sector::getLeft(o2::TPC::Sector(s)));
      std::stringstream sectornamestreamright;
      sectornamestreamright << "TPCHitsShiftedSector" << s;

      hitvectorsright.clear();
      hitidsright.clear();
      // For sector 0 and 18 the hits have to be loaded from scratch, else the hits can be recycled
      if (s == 0 || s == 18) {
        getHits(context, hitvectorsright, hitidsright, sectornamestreamright.str().c_str(), tmin, tmax, fTPC);
      } else {
        hitvectorsright = hitvectorsleft;
        hitidsright = hitidsleft;
      }
      hitvectorsleft.clear();
      hitidsleft.clear();
      getHits(context, hitvectorsleft, hitidsleft, sectornamestreamright.str().c_str(), tmin, tmax, fTPC);

      task.setData(&hitvectorsleft, &hitvectorsright, &hitidsleft, &hitidsright, &context);
      task.setStartTime(tmin);
      task.setEndTime(tmax);
      task.Exec2("");

      outtree.Fill();
      hasData |= hitidsleft.size() || hitidsright.size();
    }

    // condition to end:
    if (!hasData) {
      break;
    }
  }
  file->Write();
}

void runTPCDigitizationEventwise(const o2::steer::RunContext& context)
{
  std::vector<std::vector<o2::TPC::HitGroup>*> hitvectorsleft;  // "TPCHitVector"
  std::vector<std::vector<o2::TPC::HitGroup>*> hitvectorsright; // "TPCHitVector"
  std::vector<o2::TPC::TPCHitGroupID> hitidsleft;               // "TPCHitIDs"
  std::vector<o2::TPC::TPCHitGroupID> hitidsright;              // "TPCHitIDs

  o2::TPC::DigitizerTask task;
  task.Init2();

  // ===| open file and register branches |=====================================
  std::unique_ptr<TFile> file = std::unique_ptr<TFile>(TFile::Open("tpc_digi.root", "recreate"));
  TTree outtree("o2sim", "TPC digits");

  FairSystemInfo sysinfo;

  using digitType = std::vector<o2::TPC::Digit>;
  using mcType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  std::array<digitType*, o2::TPC::Sector::MAXSECTOR> digitArrays;
  std::array<mcType*, o2::TPC::Sector::MAXSECTOR> mcTruthArrays;
  std::array<TBranch*, o2::TPC::Sector::MAXSECTOR> digitBranches;
  std::array<TBranch*, o2::TPC::Sector::MAXSECTOR> mcTruthBranches;

  auto& eventrecords = context.getEventRecords();

  // loop over sectors
  for (int s = 0; s < o2::TPC::Sector::MAXSECTOR; ++s) {

    digitArrays[s] = new digitType;
    digitBranches[s] = outtree.Branch(Form("TPCDigit_%i", s), &digitArrays[s]);

    // Register MC Truth container
    mcTruthArrays[s] = new mcType;
    mcTruthBranches[s] = outtree.Branch(Form("TPCDigitMCTruth_%i", s), &mcTruthArrays[s]);

    task.setupSector(s);
    task.setOutputData(digitArrays[s], mcTruthArrays[s]);

    std::stringstream sectornamestreamleft;
    sectornamestreamleft << "TPCHitsShiftedSector" << int(o2::TPC::Sector::getLeft(o2::TPC::Sector(s)));
    std::stringstream sectornamestreamright;
    sectornamestreamright << "TPCHitsShiftedSector" << s;

    // loop over events
    for (int entry = 0; entry < context.getNEntries(); ++entry) {
      hitvectorsright.clear();
      hitidsright.clear();
      // For sector 0 and 18 the hits have to be loaded from scratch, else the hits can be recycled
      if (s == 0 || s == 18) {
        getHits(context, hitvectorsright, hitidsright, sectornamestreamright.str().c_str(), entry);
      } else {
        hitvectorsright = hitvectorsleft;
        hitidsright = hitidsleft;
      }
      hitvectorsleft.clear();
      hitidsleft.clear();
      getHits(context, hitvectorsleft, hitidsleft, sectornamestreamleft.str().c_str(), entry);

      task.setData(&hitvectorsleft, &hitvectorsright, &hitidsleft, &hitidsright, &context);
      task.setEndTime(eventrecords[entry].timeNS);
      task.Exec2("");
      outtree.Fill();
    }
    task.FinishTask2();
    outtree.Fill();
  }
  file->Write();
}

void runTPCDigitization_mgr(bool isChunk = true)
{
  auto& hitrunmgr = o2::steer::HitProcessingManager::instance();
  hitrunmgr.addInputFile("o2sim.root");

  // Timer
  TStopwatch timer;
  timer.Start();

  if (isChunk)
    hitrunmgr.registerRunFunction(runTPCDigitizationChunkwise);
  else
    hitrunmgr.registerRunFunction(runTPCDigitizationEventwise);
  hitrunmgr.run();

  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  FairSystemInfo sysinfo;
  std::cout << "\n\n";
  std::cout << "Macro finished succesfully.\n";
  std::cout << "Real time " << rtime << " s, CPU time " << ctime << "s\n";
  std::cout << "Memory used " << sysinfo.GetMaxMemory() << " MB\n";
}
