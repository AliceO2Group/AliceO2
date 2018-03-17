// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Tracking/src/TrackerDevice.cxx
/// \brief  Implementation of tracker device for the MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   09 May 2017
#include "TrackerDevice.h"

#include "FairMQLogger.h"
#include "options/FairMQProgOptions.h"
#include "MIDBase/Serializer.h"

namespace o2
{
namespace mid
{

//______________________________________________________________________________
TrackerDevice::TrackerDevice() : FairMQDevice(), mTracker()
{
  /// Default constructor
  // register a handler for data arriving on "data-in" channel
  OnData("data-in", &TrackerDevice::handleData);
}

//______________________________________________________________________________
bool TrackerDevice::handleData(FairMQMessagePtr& msg, int /*index*/)
{
  /// Main function: runs on a data containing the clusters
  /// and builds the tracks

  LOG(INFO) << "Received data of size: " << msg->GetSize();

  if (msg->GetSize() == 0) {
    LOG(INFO) << "Empty message: skip";
    return true;
  }

  std::vector<Cluster2D> clusters;
  Deserialize<Deserializer<Cluster2D>>(*msg, clusters);

  mTracker.process(clusters);

  if (mTracker.getNTracks() > 0) {
    FairMQMessagePtr msgOut(NewMessage());
    // Serialize<BoostSerializer<Track>>(*msgOut, mTracks);
    Serialize<Serializer<Track>>(*msgOut, mTracker.getTracks(), mTracker.getNTracks());

    // Send out the output message
    if (Send(msgOut, "data-out") < 0) {
      LOG(ERROR) << "problem sending message";
      return false;
    }
  }

  return true;
}

} // namespace mid
} // namespace o2
