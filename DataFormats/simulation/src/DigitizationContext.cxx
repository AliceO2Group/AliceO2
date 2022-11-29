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
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include <TChain.h>
#include <TFile.h>
#include <iostream>
#include <numeric> // for iota
#include <MathUtils/Cartesian.h>

using namespace o2::steer;

void DigitizationContext::printCollisionSummary(bool withQED, int truncateOutputTo) const
{
  std::cout << "Summary of DigitizationContext --\n";
  std::cout << "Maximal parts per collision " << mMaxPartNumber << "\n";
  std::cout << "Collision parts taken from simulations specified by prefix:\n";
  for (int p = 0; p < mSimPrefixes.size(); ++p) {
    std::cout << "Part " << p << " : " << mSimPrefixes[p] << "\n";
  }
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
    for (int i = 0; i < mEventRecords.size(); ++i) {
      if (truncateOutputTo >= 0 && i > truncateOutputTo) {
        std::cout << "--- Output truncated to " << truncateOutputTo << " ---\n";
        break;
      }
      std::cout << "Collision " << i << " TIME " << mEventRecords[i];
      for (auto& e : mEventParts[i]) {
        std::cout << " (" << e.sourceID << " , " << e.entryID << ")";
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
  TFile file(filename.data(), "RECREATE");
  auto cl = TClass::GetClass(typeid(*this));
  file.WriteObjectAny(this, cl, "DigitizationContext");
  file.Close();
}

DigitizationContext const* DigitizationContext::loadFromFile(std::string_view filename)
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

void DigitizationContext::fillQED(std::string_view QEDprefix, std::vector<o2::InteractionTimeRecord> const& irecord)
{
  mQEDSimPrefix = QEDprefix;

  std::vector<std::vector<o2::steer::EventPart>> qedEventParts;

  // we need to fill the QED parts (using a simple round robin scheme)
  auto qedKinematicsName = o2::base::NameConf::getMCKinematicsFileName(mQEDSimPrefix);
  // find out how many events are stored
  TFile f(qedKinematicsName.c_str(), "OPEN");
  auto t = (TTree*)f.Get("o2sim");
  if (!t) {
    LOG(error) << "No QED kinematics found";
    throw std::runtime_error("No QED kinematics found");
  }
  auto numberQEDevents = t->GetEntries();
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

void DigitizationContext::finalizeTimeframeStructure(long startOrbit, long orbitsPerTF)
{
  // the goal is to determine timeframe boundaries inside the interaction record vectors
  // determine if we can do anything
  LOG(info) << "finalizing timeframe structure";
  if (mEventRecords.size() == 0) {
    // nothing to do
    return;
  }

  if (mEventRecords.back().orbit < startOrbit) {
    LOG(error) << "start orbit larger than last collision entry";
    return;
  }

  // skip to the first index falling within our constrained
  int left = 0;
  while (left < mEventRecords.size() && mEventRecords[left].orbit < startOrbit) {
    left++;
  }

  // now we can start (2 pointer approach)
  auto right = left;
  int timeframe_count = 1;
  while (right < mEventRecords.size()) {
    if (mEventRecords[right].orbit >= startOrbit + timeframe_count * orbitsPerTF) {
      // we finished one timeframe
      mTimeFrameStartIndex.emplace_back(std::pair<int, int>(left, right - 1));
      timeframe_count++;
      left = right;
    }
    right++;
  }
  // we finished one timeframe
  mTimeFrameStartIndex.emplace_back(std::pair<int, int>(left, right - 1));
  LOG(info) << "Fixed " << mTimeFrameStartIndex.size() << " timeframes ";
  for (auto p : mTimeFrameStartIndex) {
    LOG(info) << p.first << " " << p.second;
  }
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
