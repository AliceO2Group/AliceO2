// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Steer/MCKinematicsReader.h"
#include <TChain.h>
#include <vector>
#include "FairLogger.h"

using namespace o2::steer;

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
    }
  }
}

bool MCKinematicsReader::init(std::string_view name)
{
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

  // actual loading will be done only if someone asks
  // the first time for a particular source ...

  return true;
}
