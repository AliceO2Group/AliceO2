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
    } else {
      LOG(WARN) << "MCHeader branch not found";
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
  mInitialized = true;

  return true;
}
