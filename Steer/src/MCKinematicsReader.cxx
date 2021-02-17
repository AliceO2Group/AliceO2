// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsCommonDataFormats/NameConf.h"
#include "Steer/MCKinematicsReader.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/TrackReference.h"
#include <TChain.h>
#include <vector>
#include "FairLogger.h"

using namespace o2::steer;

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

void MCKinematicsReader::loadTracksForSource(int source) const
{
  auto chain = mInputChains[source];
  if (chain) {
    // todo: get name from NameConfig
    auto br = chain->GetBranch("MCTrack");
    if (br) {
      std::vector<MCTrack>* loadtracks = nullptr;
      br->SetAddress(&loadtracks);
      // load all kinematics
      mTracks[source].resize(br->GetEntries());
      for (int event = 0; event < br->GetEntries(); ++event) {
        br->GetEntry(event);
        mTracks[source][event] = *loadtracks;
      }
      delete loadtracks;
      loadtracks = nullptr;
    }
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
      LOG(WARN) << "MCHeader branch not found";
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
      LOG(WARN) << "TrackRefs branch not found";
    }
  }
}

bool MCKinematicsReader::initFromDigitContext(std::string_view name)
{
  if (mInitialized) {
    LOG(INFO) << "MCKinematicsReader already initialized; doing nothing";
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
    LOG(INFO) << "MCKinematicsReader already initialized; doing nothing";
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
