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

#include "CommonUtils/NameConf.h"
#include "Steer/MCKinematicsReader.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/TrackReference.h"
#include <TChain.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <fairlogger/Logger.h>

using namespace o2::steer;

void MCKinematicsReader::reportMissingSource(int source, size_t available)
{
  throw std::out_of_range("MCKinematicsReader: there are " + std::to_string(available) + " sources; source " +
                          std::to_string(source) + " is not one of them");
}

void MCKinematicsReader::reportMissingEvent(const char* what, int source, int event, size_t available)
{
  throw std::out_of_range("MCKinematicsReader: source " + std::to_string(source) + " has " +
                          std::to_string(available) + " " + what + "; there is no event " + std::to_string(event));
}

void MCKinematicsReader::ensureTracksForSourceAndEvent(int source, int event) const
{
  if (static_cast<size_t>(event) >= mTracks[source].size()) {
    reportMissingEvent("events", source, event, mTracks[source].size());
  }
  loadTracksForSourceAndEvent(source, event);
}

MCKinematicsReader::~MCKinematicsReader()
{
  for (auto chain : mInputChains) {
    delete chain;
  }
  mInputChains.clear();

  if (mDigitizationContext && mOwningDigiContext) {
    delete mDigitizationContext;
  }
}

void MCKinematicsReader::initIndexedTrackRefs(std::vector<o2::TrackReference>& refs, o2::dataformats::MCTruthContainer<o2::TrackReference>& indexedrefs) const
{
  // sort trackrefs according to track index then according to track length
  std::sort(refs.begin(), refs.end(), [](const o2::TrackReference& a, const o2::TrackReference& b) {
    if (a.getTrackID() == b.getTrackID()) {
      return a.getLength() < b.getLength();
    }
    return a.getTrackID() < b.getTrackID();
  });

  // make final indexed container for track references
  indexedrefs.clear();
  for (auto& ref : refs) {
    if (ref.getTrackID() >= 0) {
      indexedrefs.addElement(ref.getTrackID(), ref);
    }
  }
}

void MCKinematicsReader::initTracksForSource(int source) const
{
  auto chain = mInputChains[source];
  if (chain) {
    // todo: get name from NameConfig
    auto br = chain->GetBranch("MCTrack");
    mTracks[source].resize(br->GetEntries(), nullptr);
  }
}

void MCKinematicsReader::loadTracksForSourceAndEvent(int source, int event) const
{
  auto chain = mInputChains[source];
  if (chain) {
    // todo: get name from NameConfig
    auto br = chain->GetBranch("MCTrack");
    if (br) {
      std::vector<MCTrack>* loadtracks = nullptr;
      br->SetAddress(&loadtracks);
      br->GetEntry(event);
      // ROOT allocated the vector for us and we own it (we passed a pointer to nullptr): keep it instead of copying it
      mTracks[source][event] = loadtracks;
      br->ResetAddress(); // the branch must not refer to the stored vector (nor to the local pointer) any more
      // free the decompressed baskets (~ the size of the event) if no later entry reads them, i.e. at the end of its cluster
      auto clusterIt = br->GetTree()->GetClusterIterator(event);
      clusterIt.Next();
      if (event + 1 >= clusterIt.GetNextEntry()) {
        br->DropBaskets("all");
      }
    }
  }
}

void MCKinematicsReader::releaseTracksForSourceAndEvent(int source, int eventID)
{
  if (mTracks.at(source).at(eventID) != nullptr) {
    delete mTracks[source][eventID];
    mTracks[source][eventID] = nullptr;
  }
  // the track references of this event as well (reloaded on demand)
  if (static_cast<size_t>(eventID) < mTrackRefsLoaded.at(source).size() && mTrackRefsLoaded[source][eventID]) {
    mIndexedTrackRefs[source][eventID] = o2::dataformats::MCTruthContainer<o2::TrackReference>();
    mTrackRefsLoaded[source][eventID] = false;
  }
}

void MCKinematicsReader::loadHeadersForSource(int source) const
{
  auto chain = mInputChains[source];
  if (chain) {
    // todo: get name from NameConfig
    auto br = chain->GetBranch("MCEventHeader.");
    if (br) {
      o2::dataformats::MCEventHeader* header = nullptr;
      br->SetAddress(&header);
      mHeaders[source].resize(br->GetEntries());
      for (int event = 0; event < br->GetEntries(); ++event) {
        br->GetEntry(event);
        mHeaders[source][event] = *header;
      }
      delete header;
      header = nullptr;
    } else {
      LOG(warn) << "MCHeader branch not found";
    }
  }
}

void MCKinematicsReader::initTrackRefsForSource(int source) const
{
  auto chain = mInputChains[source];
  if (chain) {
    // todo: get name from NameConfig
    auto br = chain->GetBranch("TrackRefs");
    if (br) {
      mIndexedTrackRefs[source].resize(br->GetEntries());
      mTrackRefsLoaded[source].assign(br->GetEntries(), false);
    } else {
      LOG(warn) << "TrackRefs branch not found";
    }
  }
}

void MCKinematicsReader::loadTrackRefsForSourceAndEvent(int source, int event) const
{
  // todo: get name from NameConfig
  auto br = mInputChains[source]->GetBranch("TrackRefs");
  std::vector<o2::TrackReference>* refs = nullptr; // allocated by ROOT, owned by us
  br->SetAddress(&refs);
  br->GetEntry(event);
  if (refs) {
    // we convert the original flat vector into an indexed structure
    initIndexedTrackRefs(*refs, mIndexedTrackRefs[source][event]);
    delete refs;
  }
  br->ResetAddress();
  // free the decompressed baskets if no later entry reads them, i.e. at the end of the cluster of this event
  auto clusterIt = br->GetTree()->GetClusterIterator(event);
  clusterIt.Next();
  if (event + 1 >= clusterIt.GetNextEntry()) {
    br->DropBaskets("all");
  }
  mTrackRefsLoaded[source][event] = true;
}

bool MCKinematicsReader::initFromDigitContext(o2::steer::DigitizationContext const* context)
{
  if (mInitialized) {
    LOG(info) << "MCKinematicsReader already initialized; doing nothing";
    return false;
  }

  mInitialized = true;
  mDigitizationContext = context;

  // get the chains to read
  mDigitizationContext->initSimKinematicsChains(mInputChains);

  // load the kinematics information
  mTracks.resize(mInputChains.size());
  mHeaders.resize(mInputChains.size());
  mIndexedTrackRefs.resize(mInputChains.size());
  mTrackRefsLoaded.resize(mInputChains.size());

  // actual loading will be done only if someone asks
  // the first time for a particular source ...

  return true;
}

bool MCKinematicsReader::initFromDigitContext(std::string_view name)
{
  if (mInitialized) {
    LOG(info) << "MCKinematicsReader already initialized; doing nothing";
    return false;
  }

  auto context = DigitizationContext::loadFromFile(name);
  if (!context) {
    return false;
  }
  mOwningDigiContext = true;
  return initFromDigitContext(context);
}

bool MCKinematicsReader::initFromKinematics(std::string_view name)
{
  if (mInitialized) {
    LOG(info) << "MCKinematicsReader already initialized; doing nothing";
    return false;
  }
  mInputChains.emplace_back(new TChain("o2sim"));
  mInputChains.back()->AddFile(o2::base::NameConf::getMCKinematicsFileName(name.data()).c_str());
  mTracks.resize(1);
  mHeaders.resize(1);
  mIndexedTrackRefs.resize(1);
  mTrackRefsLoaded.resize(1);
  mInitialized = true;

  return true;
}
