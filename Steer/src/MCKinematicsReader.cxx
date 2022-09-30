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
#include <vector>
#include <fairlogger/Logger.h>

using namespace o2::steer;

MCKinematicsReader::~MCKinematicsReader()
{
  for (auto chain : mInputChains) {
    delete chain;
  }
  mInputChains.clear();

  if (mDigitizationContext) {
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
      mTracks[source][event] = new std::vector<o2::MCTrack>;
      *mTracks[source][event] = *loadtracks;
      delete loadtracks;
    }
  }
}

void MCKinematicsReader::releaseTracksForSourceAndEvent(int source, int eventID)
{
  if (mTracks.at(source).at(eventID) != nullptr) {
    delete mTracks[source][eventID];
    mTracks[source][eventID] = nullptr;
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

void MCKinematicsReader::loadTrackRefsForSource(int source) const
{
  auto chain = mInputChains[source];
  if (chain) {
    // todo: get name from NameConfig
    auto br = chain->GetBranch("TrackRefs");
    if (br) {
      std::vector<o2::TrackReference>* refs = nullptr;
      br->SetAddress(&refs);
      mIndexedTrackRefs[source].resize(br->GetEntries());
      for (int event = 0; event < br->GetEntries(); ++event) {
        br->GetEntry(event);
        if (refs) {
          // we convert the original flat vector into an indexed structure
          initIndexedTrackRefs(*refs, mIndexedTrackRefs[source][event]);
          delete refs;
          refs = nullptr;
        }
      }
    } else {
      LOG(warn) << "TrackRefs branch not found";
    }
  }
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
  mInitialized = true;
  mDigitizationContext = context;

  // get the chains to read
  mDigitizationContext->initSimKinematicsChains(mInputChains);

  // load the kinematics information
  mTracks.resize(mInputChains.size());
  mHeaders.resize(mInputChains.size());
  mIndexedTrackRefs.resize(mInputChains.size());

  // actual loading will be done only if someone asks
  // the first time for a particular source ...

  return true;
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
  mInitialized = true;

  return true;
}
