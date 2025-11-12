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

#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/InteractionSampler.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include <TChain.h>
#include <TFile.h>
#include <iostream>
#include <numeric> // for iota
#include <MathUtils/Cartesian.h>
#include <DataFormatsCalibration/MeanVertexObject.h>
#include <filesystem>

using namespace o2::steer;

void DigitizationContext::printCollisionSummary(bool withQED, int truncateOutputTo) const
{
  std::cout << "Summary of DigitizationContext --\n";
  std::cout << "Maximal parts per collision " << mMaxPartNumber << "\n";
  std::cout << "Collision parts taken from simulations specified by prefix:\n";
  for (int p = 0; p < mSimPrefixes.size(); ++p) {
    std::cout << "Part " << p << " : " << mSimPrefixes[p] << "\n";
  }
  std::cout << "QED information included " << isQEDProvided() << "\n";
  if (withQED) {
    std::cout << "Number of Collisions " << mEventRecords.size() << "\n";
    std::cout << "Number of QED events " << mEventRecordsWithQED.size() - mEventRecords.size() << "\n";
    // loop over combined stuff
    for (int i = 0; i < mEventRecordsWithQED.size(); ++i) {
      if (truncateOutputTo >= 0 && i > truncateOutputTo) {
        std::cout << "--- Output truncated to " << truncateOutputTo << " ---\n";
        break;
      }
      std::cout << "Record " << i << " TIME " << mEventRecordsWithQED[i];
      for (auto& e : mEventPartsWithQED[i]) {
        std::cout << " (" << e.sourceID << " , " << e.entryID << ")";
      }
      std::cout << "\n";
    }
  } else {
    std::cout << "Number of Collisions " << mEventRecords.size() << "\n";
    if (mEventPartsWithQED.size() > 0) {
      auto num_qed_events = mEventPartsWithQED.size() - mEventRecords.size();
      if (num_qed_events > 0) {
        std::cout << "Number of QED events (but not shown) " << num_qed_events << "\n";
        // find first and last QED collision so that we can give the range in orbits where these
        // things are included
        auto firstQEDcoll_iter = std::find_if(mEventPartsWithQED.begin(), mEventPartsWithQED.end(),
                                              [](const std::vector<EventPart>& vec) {
                                                return std::find_if(vec.begin(), vec.end(), [](EventPart const& p) { return p.sourceID == 99; }) != vec.end();
                                              });

        auto lastColl_iter = std::find_if(mEventPartsWithQED.rbegin(), mEventPartsWithQED.rend(),
                                          [](const std::vector<EventPart>& vec) {
                                            return std::find_if(vec.begin(), vec.end(), [](EventPart const& p) { return p.sourceID == 99; }) != vec.end();
                                          });

        auto firstindex = std::distance(mEventPartsWithQED.begin(), firstQEDcoll_iter);
        auto lastindex = std::distance(mEventPartsWithQED.begin(), lastColl_iter.base()) - 1;
        std::cout << "QED from: " << mEventRecordsWithQED[firstindex] << " ---> " << mEventRecordsWithQED[lastindex] << "\n";
      }
    }
    for (int i = 0; i < mEventRecords.size(); ++i) {
      if (truncateOutputTo >= 0 && i > truncateOutputTo) {
        std::cout << "--- Output truncated to " << truncateOutputTo << " ---\n";
        break;
      }
      std::cout << "Collision " << i << " TIME " << mEventRecords[i];
      for (auto& e : mEventParts[i]) {
        std::cout << " (" << e.sourceID << " , " << e.entryID << ")";
      }
      if (i < mInteractionVertices.size()) {
        std::cout << " sampled vertex : " << mInteractionVertices[i];
      }
      std::cout << "\n";
    }
  }
}

void DigitizationContext::setSimPrefixes(std::vector<std::string> const& prefixes)
{
  mSimPrefixes = prefixes;
}

bool DigitizationContext::initSimChains(o2::detectors::DetID detid, std::vector<TChain*>& simchains) const
{
  if (!(simchains.size() == 0)) {
    // nothing to do ... already setup
    return false;
  }

  // check that all files are present, otherwise quit
  for (int source = 0; source < mSimPrefixes.size(); ++source) {
    if (!std::filesystem::exists(o2::base::DetectorNameConf::getHitsFileName(detid, mSimPrefixes[source].data()))) {
      LOG(info) << "Not hit file present for " << detid.getName() << " (exiting SimChain setup)";
      return false;
    }
  }

  simchains.emplace_back(new TChain("o2sim"));
  // add the main (background) file
  simchains.back()->AddFile(o2::base::DetectorNameConf::getHitsFileName(detid, mSimPrefixes[0].data()).c_str());

  for (int source = 1; source < mSimPrefixes.size(); ++source) {
    simchains.emplace_back(new TChain("o2sim"));
    // add signal files
    simchains.back()->AddFile(o2::base::DetectorNameConf::getHitsFileName(detid, mSimPrefixes[source].data()).c_str());
  }

  // QED part
  if (mEventRecordsWithQED.size() > 0) {
    if (mSimPrefixes.size() >= QEDSOURCEID) {
      LOG(fatal) << "Too many signal chains; crashes with QED source ID";
    }

    // it might be better to use an unordered_map for the simchains but this requires interface changes
    simchains.resize(QEDSOURCEID + 1, nullptr);
    simchains[QEDSOURCEID] = new TChain("o2sim");
    simchains[QEDSOURCEID]->AddFile(o2::base::DetectorNameConf::getHitsFileName(detid, mQEDSimPrefix).c_str());
  }

  return true;
}

/// Common functions the setup input TChains for reading kinematics information, given the state (prefixes) encapsulated
/// by this context. The input vector needs to be empty otherwise nothing will be done.
/// return boolean saying if input simchains was modified or not
bool DigitizationContext::initSimKinematicsChains(std::vector<TChain*>& simkinematicschains) const
{
  if (!(simkinematicschains.size() == 0)) {
    // nothing to do ... already setup
    return false;
  }

  simkinematicschains.emplace_back(new TChain("o2sim"));
  // add the main (background) file
  simkinematicschains.back()->AddFile(o2::base::NameConf::getMCKinematicsFileName(mSimPrefixes[0].data()).c_str());

  for (int source = 1; source < mSimPrefixes.size(); ++source) {
    simkinematicschains.emplace_back(new TChain("o2sim"));
    // add signal files
    simkinematicschains.back()->AddFile(o2::base::NameConf::getMCKinematicsFileName(mSimPrefixes[source].data()).c_str());
  }

  // we add QED, if used in the digitization context
  if (mEventRecordsWithQED.size() > 0) {
    if (mSimPrefixes.size() >= QEDSOURCEID) {
      LOG(fatal) << "Too many signal chains; crashes with QED source ID";
    }

    // it might be better to use an unordered_map for the simchains but this requires interface changes
    simkinematicschains.resize(QEDSOURCEID + 1, nullptr);
    simkinematicschains[QEDSOURCEID] = new TChain("o2sim");
    simkinematicschains[QEDSOURCEID]->AddFile(o2::base::DetectorNameConf::getMCKinematicsFileName(mQEDSimPrefix).c_str());
  }

  return true;
}

bool DigitizationContext::checkVertexCompatibility(bool verbose) const
{
  if (mMaxPartNumber == 1) {
    return true;
  }

  auto checkVertexPair = [](math_utils::Point3D<double> const& p1, math_utils::Point3D<double> const& p2) -> bool {
    return (p2 - p1).Mag2() < 1E-6;
  };

  std::vector<TChain*> kinematicschain;
  std::vector<TBranch*> headerbranches;
  std::vector<o2::dataformats::MCEventHeader*> headers;
  std::vector<math_utils::Point3D<double>> vertices;
  initSimKinematicsChains(kinematicschain);
  bool consistent = true;
  if (kinematicschain.size() > 0) {
    headerbranches.resize(kinematicschain.size(), nullptr);
    headers.resize(kinematicschain.size(), nullptr);
    // loop over all collisions in this context
    int collisionID = 0;
    for (auto& collision : getEventParts()) {
      collisionID++;
      vertices.clear();
      for (auto& part : collision) {
        const auto source = part.sourceID;
        const auto entry = part.entryID;
        auto chain = kinematicschain[source];
        if (!headerbranches[source]) {
          headerbranches[source] = chain->GetBranch("MCEventHeader.");
          headerbranches[source]->SetAddress(&headers[source]);
        }
        // get the MCEventHeader to read out the vertex
        headerbranches[source]->GetEntry(entry);
        auto header = headers[source];
        vertices.emplace_back(header->GetX(), header->GetY(), header->GetZ());
      }
      // analyse vertex matching
      bool thiscollision = true;
      const auto& p1 = vertices[0];
      for (int j = 1; j < vertices.size(); ++j) {
        const auto& p2 = vertices[j];
        bool thischeck = checkVertexPair(p1, p2);
        thiscollision &= thischeck;
      }
      if (verbose && !thiscollision) {
        std::stringstream text;
        text << "Found inconsistent vertices for digit collision ";
        text << collisionID << " : ";
        for (auto& p : vertices) {
          text << p << " ";
        }
        LOG(error) << text.str();
      }
      consistent &= thiscollision;
    }
  }
  return consistent;
}

o2::parameters::GRPObject const& DigitizationContext::getGRP() const
{
  if (!mGRP) {
    // we take the GRP from the background file
    // maybe we should add a check that all GRPs are consistent ..
    mGRP = o2::parameters::GRPObject::loadFrom(mSimPrefixes[0]);
  }
  return *mGRP;
}

void DigitizationContext::saveToFile(std::string_view filename) const
{
  // checks if the path content of filename exists ... otherwise it is created before creating the ROOT file
  auto ensure_path_exists = [](std::string_view filename) {
    try {
      // Extract the directory path from the filename
      std::filesystem::path file_path(filename);
      std::filesystem::path dir_path = file_path.parent_path();

      // Check if the directory path is empty (which means filename was just a name without path)
      if (dir_path.empty()) {
        // nothing to do
        return true;
      }

      // Create directories if they do not exist
      if (!std::filesystem::exists(dir_path)) {
        if (std::filesystem::create_directories(dir_path)) {
          // std::cout << "Directories created successfully: " << dir_path.string() << std::endl;
          return true;
        } else {
          std::cerr << "Failed to create directories: " << dir_path.string() << std::endl;
          return false;
        }
      }
      return true;
    } catch (const std::filesystem::filesystem_error& ex) {
      std::cerr << "Filesystem error: " << ex.what() << std::endl;
      return false;
    } catch (const std::exception& ex) {
      std::cerr << "General error: " << ex.what() << std::endl;
      return false;
    }
  };

  if (!ensure_path_exists(filename)) {
    LOG(error) << "Filename contains path component which could not be created";
    return;
  }

  TFile file(filename.data(), "RECREATE");
  if (file.IsOpen()) {
    auto cl = TClass::GetClass(typeid(*this));
    file.WriteObjectAny(this, cl, "DigitizationContext");
    file.Close();
  } else {
    LOG(error) << "Could not write to file " << filename.data();
  }
}

DigitizationContext* DigitizationContext::loadFromFile(std::string_view filename)
{
  std::string tmpFile;
  if (filename == "") {
    tmpFile = o2::base::NameConf::getCollisionContextFileName();
    filename = tmpFile;
  }
  DigitizationContext* incontext = nullptr;
  TFile file(filename.data(), "OPEN");
  file.GetObject("DigitizationContext", incontext);
  return incontext;
}

void DigitizationContext::fillQED(std::string_view QEDprefix, int max_events, double qedrate)
{
  if (mEventRecords.size() <= 1) {
    // nothing to do
    return;
  }

  o2::steer::InteractionSampler qedInteractionSampler;
  qedInteractionSampler.setBunchFilling(mBCFilling);

  // get first and last "hadronic" interaction records and let
  // QED events range from the first bunch crossing to the last bunch crossing
  // in this range
  auto first = mEventRecords.front();
  auto last = mEventRecords.back();
  first.bc = 0;
  last.bc = o2::constants::lhc::LHCMaxBunches;
  LOG(info) << "QED RATE " << qedrate;
  qedInteractionSampler.setInteractionRate(qedrate);
  qedInteractionSampler.setFirstIR(first);
  qedInteractionSampler.init();
  qedInteractionSampler.print();
  std::vector<o2::InteractionTimeRecord> qedinteractionrecords;
  o2::InteractionTimeRecord t;
  LOG(info) << "GENERATING COL TIMES";
  t = qedInteractionSampler.generateCollisionTime();
  while ((t = qedInteractionSampler.generateCollisionTime()) < last) {
    qedinteractionrecords.push_back(t);
  }
  LOG(info) << "DONE GENERATING COL TIMES";
  fillQED(QEDprefix, qedinteractionrecords, max_events, false);
}

void DigitizationContext::fillQED(std::string_view QEDprefix, std::vector<o2::InteractionTimeRecord> const& irecord, int max_events, bool fromKinematics)
{
  mQEDSimPrefix = QEDprefix;

  std::vector<std::vector<o2::steer::EventPart>> qedEventParts;

  int numberQEDevents = max_events; // if this is -1 there will be no limitation
  if (fromKinematics) {
    // we need to fill the QED parts (using a simple round robin scheme)
    auto qedKinematicsName = o2::base::NameConf::getMCKinematicsFileName(mQEDSimPrefix);
    // find out how many events are stored
    TFile f(qedKinematicsName.c_str(), "OPEN");
    auto t = (TTree*)f.Get("o2sim");
    if (!t) {
      LOG(error) << "No QED kinematics found";
      throw std::runtime_error("No QED kinematics found");
    }
    numberQEDevents = t->GetEntries();
  }

  int eventID = 0;
  for (auto& tmp : irecord) {
    std::vector<EventPart> qedpart;
    qedpart.emplace_back(QEDSOURCEID, eventID++);
    qedEventParts.push_back(qedpart);
    if (eventID == numberQEDevents) {
      eventID = 0;
    }
  }

  // we need to do the interleaved event records for detectors consuming both
  // normal and QED events
  // --> merge both; sort first according to times and sort second one according to same order
  auto combinedrecords = mEventRecords;
  combinedrecords.insert(combinedrecords.end(), irecord.begin(), irecord.end());
  auto combinedparts = mEventParts;
  combinedparts.insert(combinedparts.end(), qedEventParts.begin(), qedEventParts.end());

  // get sorted index vector based on event records
  std::vector<size_t> idx(combinedrecords.size());
  std::iota(idx.begin(), idx.end(), 0);

  std::stable_sort(idx.begin(), idx.end(),
                   [&combinedrecords](size_t i1, size_t i2) { return combinedrecords[i1] < combinedrecords[i2]; });

  mEventRecordsWithQED.clear();
  mEventPartsWithQED.clear();
  for (int i = 0; i < idx.size(); ++i) {
    mEventRecordsWithQED.push_back(combinedrecords[idx[i]]);
    mEventPartsWithQED.push_back(combinedparts[idx[i]]);
  }
}

namespace
{
// a common helper for timeframe structure
std::vector<std::pair<int, int>> getTimeFrameBoundaries(std::vector<o2::InteractionTimeRecord> const& irecords, long startOrbit, long orbitsPerTF)
{
  std::vector<std::pair<int, int>> result;

  // the goal is to determine timeframe boundaries inside the interaction record vectors
  // determine if we can do anything
  if (irecords.size() == 0) {
    // nothing to do
    return result;
  }

  if (irecords.back().orbit < startOrbit) {
    LOG(error) << "start orbit larger than last collision entry";
    return result;
  }

  // skip to the first index falling within our constrained
  int left = 0;
  while (left < irecords.size() && irecords[left].orbit < startOrbit) {
    left++;
  }

  // now we can start (2 pointer approach)
  auto right = left;
  int timeframe_count = 1;
  while (right < irecords.size()) {
    if (irecords[right].orbit >= startOrbit + timeframe_count * orbitsPerTF) {
      // we finished one timeframe
      result.emplace_back(std::pair<int, int>(left, right - 1));
      timeframe_count++;
      left = right;
    }
    right++;
  }
  // finished last timeframe
  result.emplace_back(std::pair<int, int>(left, right - 1));
  return result;
}

// a common helper for timeframe structure - includes indices for orbits-early (orbits from last timeframe still affecting current one)
std::vector<std::tuple<int, int, int>> getTimeFrameBoundaries(std::vector<o2::InteractionTimeRecord> const& irecords,
                                                              long startOrbit,
                                                              long orbitsPerTF,
                                                              float orbitsEarly)
{
  // we could actually use the other method first ... then do another pass to fix the early-index ... or impact index
  auto true_indices = getTimeFrameBoundaries(irecords, startOrbit, orbitsPerTF);

  std::vector<std::tuple<int, int, int>> indices_with_early{};
  for (int ti = 0; ti < true_indices.size(); ++ti) {
    // for each timeframe we copy the true indices
    auto& tf_range = true_indices[ti];

    // init new index without fixing the early index yet
    indices_with_early.push_back(std::make_tuple(tf_range.first, tf_range.second, -1));

    // from the second timeframe on we can determine the index in the previous timeframe
    // which matches our criterion
    if (orbitsEarly > 0. && ti > 0) {
      auto& prev_tf_range = true_indices[ti - 1];
      // in this range search the smallest index which precedes
      // timeframe ti by not more than "orbitsEarly" orbits
      // (could probably use binary search, in case optimization becomes necessary)
      int earlyOrbitIndex = -1; // init to start of this timeframe ... there may not be early orbits

      // this is the orbit of the ti-th timeframe start
      auto orbit_timeframe_start = startOrbit + ti * orbitsPerTF;

      auto orbit_timeframe_early_fractional = 1. * orbit_timeframe_start - orbitsEarly;
      auto orbit_timeframe_early_integral = static_cast<long>(std::floor(orbit_timeframe_early_fractional));

      auto bc_early = (uint32_t)((orbit_timeframe_early_fractional - orbit_timeframe_early_integral) * o2::constants::lhc::LHCMaxBunches);

      // this is the interaction record of the ti-th timeframe start
      o2::InteractionRecord timeframe_start_record(0, orbit_timeframe_start);
      // this is the interaction record in some previous timeframe after which interactions could still
      // influence the ti-th timeframe according to orbitsEarly
      o2::InteractionRecord timeframe_early_record(bc_early, orbit_timeframe_early_integral);

      auto differenceInBCNS_max = timeframe_start_record.differenceInBCNS(timeframe_early_record);

      for (int j = prev_tf_range.second; j >= prev_tf_range.first; --j) {
        // determine difference in timing in NS; compare that with the limit given by orbitsEarly
        auto timediff_NS = timeframe_start_record.differenceInBCNS(irecords[j]);
        if (timediff_NS < differenceInBCNS_max) {
          earlyOrbitIndex = j;
        } else {
          break;
        }
      }
      std::get<2>(indices_with_early.back()) = earlyOrbitIndex;
    }
  }
  return indices_with_early;
}

} // namespace

void DigitizationContext::applyMaxCollisionFilter(std::vector<std::tuple<int, int, int>>& timeframeindices, long startOrbit, long orbitsPerTF, int maxColl, double orbitsEarly)
{
  // the idea is to go through each timeframe and throw away collisions beyond a certain count
  // then the indices should be condensed

  std::vector<std::vector<o2::steer::EventPart>> newparts;
  std::vector<o2::InteractionTimeRecord> newrecords;

  std::unordered_map<int, int> currMaxId;                           // the max id encountered for a source
  std::unordered_map<int, std::unordered_map<int, int>> reIndexMap; // for each source, a map of old to new index for the event parts

  if (maxColl == -1) {
    maxColl = mEventRecords.size();
  }

  // the actual first actual timeframe
  int first_timeframe = orbitsEarly > 0. ? 1 : 0;

  // mapping of old to new indices
  std::unordered_map<size_t, size_t> indices_old_to_new;

  // now we can go through the structure timeframe by timeframe
  for (int tf_id = first_timeframe; tf_id < timeframeindices.size(); ++tf_id) {
    auto& tf_indices = timeframeindices[tf_id];

    auto firstindex = std::get<0>(tf_indices); // .first;
    auto lastindex = std::get<1>(tf_indices);  // .second;
    auto previndex = std::get<2>(tf_indices);

    LOG(info) << "timeframe indices " << previndex << " : " << firstindex << " : " << lastindex;

    int collCount = 0; // counting collisions within timeframe
    // copy to new structure
    for (int index = previndex >= 0 ? previndex : firstindex; index <= lastindex; ++index) {
      if (collCount >= maxColl) {
        continue;
      }

      // look if this index was already done?
      // avoid duplicate entries in transformed records
      if (indices_old_to_new.find(index) != indices_old_to_new.end()) {
        continue;
      }

      // we put these events under a certain condition
      bool keep = index < firstindex || collCount < maxColl;

      if (!keep) {
        continue;
      }

      if (index >= firstindex) {
        collCount++;
      }

      // we must also make sure that we don't duplicate the records
      // moreover some records are merely put as precoll of tf2 ---> so they shouldn't be part of tf1 in the final
      // extraction, ouch !
      // maybe we should combine the filter and individual tf extraction in one step !!
      indices_old_to_new[index] = newrecords.size();
      newrecords.push_back(mEventRecords[index]);
      newparts.push_back(mEventParts[index]);

      // reindex the event parts to achieve compactification (and initial linear increase)
      for (auto& part : newparts.back()) {
        auto source = part.sourceID;
        auto entry = part.entryID;
        // have we remapped this entry before?
        if (reIndexMap.find(source) != reIndexMap.end() && reIndexMap[source].find(entry) != reIndexMap[source].end()) {
          part.entryID = reIndexMap[source][entry];
        } else {
          // assign the next free index
          if (currMaxId.find(source) == currMaxId.end()) {
            currMaxId[source] = 0;
          }
          part.entryID = currMaxId[source];
          // cache this assignment
          reIndexMap[source][entry] = currMaxId[source];
          currMaxId[source] += 1;
        }
      }
    } // ends one timeframe

    // correct the timeframe indices
    if (indices_old_to_new.find(firstindex) != indices_old_to_new.end()) {
      std::get<0>(tf_indices) = indices_old_to_new[firstindex]; // start
    }
    if (indices_old_to_new.find(lastindex) != indices_old_to_new.end()) {
      std::get<1>(tf_indices) = indices_old_to_new[lastindex]; // end;
    } else {
      std::get<1>(tf_indices) = newrecords.size(); // end;
    }
    if (indices_old_to_new.find(previndex) != indices_old_to_new.end()) {
      std::get<2>(tf_indices) = indices_old_to_new[previndex]; // previous or "early" index
    }
  }
  // reassignment
  mEventRecords = newrecords;
  mEventParts = newparts;
}

std::vector<std::tuple<int, int, int>> DigitizationContext::calcTimeframeIndices(long startOrbit, long orbitsPerTF, double orbitsEarly) const
{
  auto timeframeindices = getTimeFrameBoundaries(mEventRecords, startOrbit, orbitsPerTF, orbitsEarly);
  LOG(info) << "Fixed " << timeframeindices.size() << " timeframes ";
  for (auto p : timeframeindices) {
    LOG(info) << std::get<0>(p) << " " << std::get<1>(p) << " " << std::get<2>(p);
  }

  return timeframeindices;
}

std::unordered_map<int, int> DigitizationContext::getCollisionIndicesForSource(int source) const
{
  // go through all collisions and pick those that have the give source
  // then keep only the first collision index
  std::unordered_map<int, int> result;
  const auto& parts = getEventParts(false);
  for (int collindex = 0; collindex < parts.size(); ++collindex) {
    for (auto& eventpart : parts[collindex]) {
      if (eventpart.sourceID == source) {
        result[eventpart.entryID] = collindex;
      }
    }
  }
  return result;
}

int DigitizationContext::findSimPrefix(std::string const& prefix) const
{
  auto iter = std::find(mSimPrefixes.begin(), mSimPrefixes.end(), prefix);
  if (iter != mSimPrefixes.end()) {
    return std::distance(mSimPrefixes.begin(), iter);
  }
  return -1;
}

namespace
{
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const
  {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};
} // namespace

int DigitizationContext::setInteractionVertices(std::vector<math_utils::Point3D<float>> const& external_vertices)
{
  if (external_vertices.size() != mEventRecords.size()) {
    LOG(error) << "Size mismatch with event record";
    return 1;
  }
  mInteractionVertices.clear();
  std::copy(external_vertices.begin(), external_vertices.end(), std::back_inserter(mInteractionVertices));
  return 0;
}

void DigitizationContext::sampleInteractionVertices(o2::dataformats::MeanVertexObject const& meanv)
{
  // mapping of source x event --> index into mInteractionVertices
  std::unordered_map<std::pair<int, int>, int, pair_hash> vertex_cache;

  mInteractionVertices.clear();
  int collcount = 0;

  std::unordered_set<int> vset; // used to detect vertex incompatibilities
  for (auto& coll : mEventParts) {
    collcount++;
    vset.clear();

    // first detect if any of these entries already has an associated vertex
    for (auto& part : coll) {
      auto source = part.sourceID;
      auto event = part.entryID;
      auto cached_iter = vertex_cache.find(std::pair<int, int>(source, event));

      if (cached_iter != vertex_cache.end()) {
        vset.emplace(cached_iter->second);
      }
    }

    // make sure that there is no conflict
    if (vset.size() > 1) {
      LOG(fatal) << "Impossible conflict during interaction vertex sampling";
    }

    int cacheindex = -1;
    if (vset.size() == 1) {
      // all of the parts need to be assigned the same existing vertex
      cacheindex = *(vset.begin());
      mInteractionVertices.push_back(mInteractionVertices[cacheindex]);
    } else {
      // we need to sample a new point
      mInteractionVertices.emplace_back(meanv.sample());
      cacheindex = mInteractionVertices.size() - 1;
    }
    // update the cache
    for (auto& part : coll) {
      auto source = part.sourceID;
      auto event = part.entryID;
      vertex_cache[std::pair<int, int>(source, event)] = cacheindex;
    }
  }
}

DigitizationContext DigitizationContext::extractSingleTimeframe(int timeframeid, std::vector<std::tuple<int, int, int>> const& timeframeindices, std::vector<int> const& sources_to_offset)
{
  DigitizationContext r; // make a return object
  if (timeframeindices.size() == 0) {
    LOG(error) << "Timeframe index structure empty; Returning empty object.";
    return r;
  }
  r.mSimPrefixes = mSimPrefixes;
  r.mMuBC = mMuBC;
  r.mBCFilling = mBCFilling;
  try {
    auto tf_ranges = timeframeindices.at(timeframeid);

    auto startindex = std::get<0>(tf_ranges);
    auto endindex = std::get<1>(tf_ranges);
    auto earlyindex = std::get<2>(tf_ranges);

    if (earlyindex >= 0) {
      startindex = earlyindex;
    }
    std::copy(mEventRecords.begin() + startindex, mEventRecords.begin() + endindex, std::back_inserter(r.mEventRecords));
    std::copy(mEventParts.begin() + startindex, mEventParts.begin() + endindex, std::back_inserter(r.mEventParts));
    if (mInteractionVertices.size() >= endindex) {
      std::copy(mInteractionVertices.begin() + startindex, mInteractionVertices.begin() + endindex, std::back_inserter(r.mInteractionVertices));
    }

    // let's assume we want to fix the ids for source = source_id
    // Then we find the first index that has this source_id and take the corresponding number
    // as offset. Thereafter we subtract this offset from all known event parts.
    auto perform_offsetting = [&r](int source_id) {
      auto indices_for_source = r.getCollisionIndicesForSource(source_id);
      int minvalue = std::numeric_limits<int>::max();
      for (auto& p : indices_for_source) {
        if (p.first < minvalue) {
          minvalue = p.first;
        }
      }
      // now fix them
      for (auto& p : indices_for_source) {
        auto index_into_mEventParts = p.second;
        for (auto& part : r.mEventParts[index_into_mEventParts]) {
          if (part.sourceID == source_id) {
            part.entryID -= minvalue;
          }
        }
      }
    };
    for (auto source_id : sources_to_offset) {
      perform_offsetting(source_id);
    }

  } catch (std::exception) {
    LOG(warn) << "No such timeframe id in collision context. Returing empty object";
  }
  // fix number of collisions
  r.setNCollisions(r.mEventRecords.size());
  return r;
}
