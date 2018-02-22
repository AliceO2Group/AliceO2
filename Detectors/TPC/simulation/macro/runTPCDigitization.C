/*
 * filterHits.C
 *
 *  Created on: Jan 30, 2018
 *      Author: swenzel
 */

#include <functional>
#include <Steer/InteractionSampler.h>
#include <TPCSimulation/Digitizer.h>
#include <TPCSimulation/ToyDigitizerTask.h>

#include <cassert>

// an index to uniquely identify a single hit of TPC
struct TPCHitGroupID {
  TPCHitGroupID() = default;
  TPCHitGroupID(int e, int gid) : entry{ e }, groupID{ gid } {}
  int entry = -1;
  int groupID = -1;
};

TTree* tree;

void getHits(std::vector<std::vector<o2::TPC::HitGroup>*>& hitvectors, std::vector<TPCHitGroupID>& hitids,
             const std::vector<o2::MCInteractionRecord>& times, const char* branchname, float tmin /*NS*/,
             float tmax /*NS*/, std::function<float(float, float, float)>&& f)
{
  // f is some function taking event time + z of hit and returns final "digit" time

  // ingredients
  // *) tree

  auto br = tree->GetBranch(branchname);
  if (!br) {
    std::cerr << "No branch found\n";
    return;
  }

  auto nentries = br->GetEntries();
  hitvectors.resize(nentries, nullptr);
  // *) number entries/events in file

  // do the filtering
  for (int entry = 0; entry < nentries; ++entry) {
    //    std::cout << "ftimes " << f(times[entry].timeNS, 0, 250) << " " << f(times[entry].timeNS, 0, 0) << "\n";
    if (tmin > f(times[entry].timeNS, 0, 0)) {
      continue;
    }
    if (tmax < f(times[entry].timeNS, 0, 250)) {
      break;
    }
    //  std::cout << "Keeping entry " << entry << " with time " << times[entry].timeNS << "\n";

    // no filtering over hitgroups
    // std::vector<o2::TPC::HitGroup>* groups = nullptr;

    // hitvectors.emplace_back(nullptr);

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
      float tmaxtrack = f(times[entry].timeNS, 0., zmin);
      float tmintrack = f(times[entry].timeNS, 0., zmax);
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

void loopHits(const std::vector<std::vector<o2::TPC::HitGroup>*>& hitvectors, const std::vector<TPCHitGroupID>& hitids,
              const std::vector<o2::MCInteractionRecord>& eventtimes)
{
  float maxt = 0;
  float mint = 1E10;

  for (auto& id : hitids) {
    auto entryvector = hitvectors[id.entry];
    auto& group = (*entryvector)[id.groupID];
    auto& MCrecord = eventtimes[id.entry];

    for (size_t hitindex = 0; hitindex < group.getSize(); ++hitindex) {
      const auto& eh = group.getHit(hitindex);
      auto t = fTPC(MCrecord.timeNS, eh.GetTime(), eh.GetZ());
      maxt = std::max(t, maxt);
      mint = std::min(t, mint);
      // std::cout << "Hit from " << id.entry << " : " << eh.GetX() << " " << eh.GetY() << " " << eh.GetZ() << "\n";
    }
  }
  std::cout << " MINT " << mint << " MAXT " << maxt << "\n";
}

void runTPCDigitization()
{
  std::vector<std::vector<o2::TPC::HitGroup>*> hitvectors; // "TPCHitVector"
  std::vector<TPCHitGroupID> hitids;                       // "TPCHitIDs"
  std::vector<o2::MCInteractionRecord> times;

  o2::TPC::ToyDigitizerTask task;

  TFile file("o2sim.root");
  tree = (TTree*)file.Get("o2sim");

  o2::steer::InteractionSampler sampler;
  times.resize(tree->GetEntries());
  sampler.generateCollisionTimes(times);

  const auto TPCDRIFT = 100000;
  for (int driftinterval = 0;; ++driftinterval) {
    auto tmin = driftinterval * TPCDRIFT;
    auto tmax = (driftinterval + 1) * TPCDRIFT;

    hitvectors.clear();
    hitids.clear();
    times.clear();

    bool hasData = false;
    // loop over sectors
    for (int s = 0; s < 36; ++s) {
      std::stringstream sectornamestream;
      sectornamestream << "TPCHitsSector" << s;
      getHits(hitvectors, hitids, times, sectornamestream.str().c_str(), tmin, tmax, fTPC);

      task.setData(&hitvectors, &times, s);
      task.Exec("");

      hasData |= hitids.size();
    }

    // condition to end:
    if (!hasData) {
      break;
    }
  }
}

void filterHits()
{
  //  std::vector<std::vector<o2::TPC::HitGroup>*> hitvectors;
  //  std::vector<TPCHitGroupID> hitids;
  //  std::vector<o2::MCInteractionRecord> times;
  //
  //  // 100us one TPC drift
  //  // 100000ns one TPC drift
  //  const auto TPCDRIFT = 100000;
  //  auto tmin = 2.*TPCDRIFT;
  //  auto tmax = 3.*TPCDRIFT;
  //  getHits(hitvectors, hitids, times, "TPCHitsSector0", tmin, tmax, fTPC);
  //
  //  std::cout << "obtained " << hitids.size() << " ids " << "\n";
  //  std::cout << "spanning " << hitvectors.size() << " entries " << "\n";
  //
  //  loopHits(hitvectors, hitids, times);
  Run();
}
