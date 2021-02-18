// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TOFCalibration/TOFFEElightReader.h>
#include "Framework/Logger.h"
#include "TSystem.h"
#include <fstream>

using namespace o2::tof;

void TOFFEElightReader::loadFEElightConfig(const char* fileName)
{
  // load FEElight config

  char* expandedFileName = gSystem->ExpandPathName(fileName);
  std::ifstream is;
  is.open(expandedFileName, std::ios::binary);
  is.read(reinterpret_cast<char*>(&mFEElightConfig), sizeof(mFEElightConfig));
  is.close();
}

//_______________________________________________________________

int TOFFEElightReader::parseFEElightConfig()
{
  // parse FEElight config
  // loops over all FEE channels, checks whether they are enabled
  // and sets channel enabled

  mFEElightInfo.resetAll();

  int nEnabled = 0, index;
  TOFFEEchannelConfig* channelConfig = nullptr;
  for (int crateId = 0; crateId < Geo::kNCrate; crateId++) {
    for (int trmId = 0; trmId < Geo::kNTRM; trmId++) {
      for (int chainId = 0; chainId < Geo::kNChain; chainId++) {
        for (int tdcId = 0; tdcId < Geo::kNTdc; tdcId++) {
          for (int channelId = 0; channelId < Geo::kNCh; channelId++) {
            channelConfig = mFEElightConfig.getChannelConfig(crateId, trmId, chainId, tdcId, channelId);
            LOG(INFO) << "Processing electronic channel with indices: crate = " << crateId << ", trm = " << trmId << ", chain = "
                      << chainId << ", tdc = " << tdcId << ", tdcChannel = " << channelId << " -> " << channelConfig;
            if (channelConfig) {
              if (!channelConfig->isEnabled()) {
                continue;
              }
              // get index DO from crate, trm, chain, tdc, tdcchannel
              index = Geo::getCHFromECH(Geo::getECHFromIndexes(crateId, trmId, chainId, tdcId, channelId));
              if (index == -1) {
                continue;
              }
              nEnabled++;
              mFEElightInfo.mChannelEnabled[index] = channelConfig->isEnabled();
              mFEElightInfo.mMatchingWindow[index] = channelConfig->mMatchingWindow;
              mFEElightInfo.mLatencyWindow[index] = channelConfig->mLatencyWindow;
            }
          }
        }
      }
    }
  }

  TOFFEEtriggerConfig* triggerConfig = nullptr;
  for (Int_t iddl = 0; iddl < TOFFEElightConfig::NTRIGGERMAPS; iddl++) {
    triggerConfig = mFEElightConfig.getTriggerConfig(iddl);
    LOG(INFO) << "Processing trigger config " << iddl << ": " << triggerConfig;
    if (triggerConfig) {
      mFEElightInfo.mTriggerMask[iddl] = triggerConfig->mStatusMap;
    }
  }

  return nEnabled;
}
