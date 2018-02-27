// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Clustering/src/ClusterizerDevice.cxx
/// \brief  Implementation of the cluster reconstruction device for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   24 October 2016
#include "ClusterizerDevice.h"

#include <vector>
#include "FairMQLogger.h"
#include "MIDBase/Serializer.h"
#include "DataFormatsMID/StripPattern.h"

#include "MIDBase/LegacyUtility.h" // FOR TESTING

namespace o2
{
namespace mid
{

//______________________________________________________________________________
ClusterizerDevice::ClusterizerDevice() : mClusterizer()
{
  /// Default constructor
  // register a handler for data arriving on "data-in" channel
  OnData("data-in", &ClusterizerDevice::handleData);
}

//______________________________________________________________________________
bool ClusterizerDevice::handleData(FairMQMessagePtr& msg, int /*index*/)
{
  /// Main function: runs on a data containing the digits
  /// and builds the clusters

  LOG(INFO) << "Received data of size: " << msg->GetSize();

  // FIXME: This will change when we will have the final format
  // The HLT adds an header. After 100 bytes, we have the number of digits
  if (msg->GetSize() < 100) {
    LOG(INFO) << "Empty message: skip";
    return true;
  }
  uint8_t* data = reinterpret_cast<uint8_t*>(msg->GetData());
  uint32_t* digitsData = reinterpret_cast<uint32_t*>(data + 100);
  uint32_t nDigits = digitsData[0];
  uint32_t offset(1);
  // Loop on digits
  std::vector<uint32_t> digits;
  for (int idig = 0; idig < nDigits; idig++) {
    uint32_t uniqueID = digitsData[offset++];
    digits.push_back(uniqueID);
    ++offset; // skip idx and adc
  }
  std::vector<ColumnData> patterns = LegacyUtility::digitsToPattern(digits);
  // end of the block that will require a change

  mClusterizer.process(patterns);

  if (mClusterizer.getNClusters() > 0) {
    // Store clusters
    FairMQMessagePtr msgOut(NewMessage());
    // Serialize<BoostSerializer<Track>>(*msgOut, mTracks);
    Serialize<Serializer<Cluster2D>>(*msgOut, mClusterizer.getClusters(), mClusterizer.getNClusters());

    // Send out the output message
    if (Send(msgOut, "data-out") < 0) {
      LOG(ERROR) << "problem sending message";
      return false;
    }
  }
  return true;
}

//______________________________________________________________________________
void ClusterizerDevice::InitTask()
{
  /// Initializes the task
  if (!mClusterizer.init()) {
    LOG(ERROR) << "Initialization of MID clusterizer device failed";
  }
}
} // namespace mid
} // namespace o2
