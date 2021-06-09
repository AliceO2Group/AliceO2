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
  mFileLoadBuff.reset(new char[sizeof(o2::tof::TOFFEElightConfig)]);
  is.read(mFileLoadBuff.get(), sizeof(o2::tof::TOFFEElightConfig));
  is.close();
  mFEElightConfig = reinterpret_cast<const TOFFEElightConfig*>(mFileLoadBuff.get());
}

//_______________________________________________________________
void TOFFEElightReader::loadFEElightConfig(gsl::span<const char> configBuf)
{
  // load FEElight config from buffer

  if (configBuf.size() != sizeof(o2::tof::TOFFEElightConfig)) {
    LOG(FATAL) << "Incoming message with TOFFEE configuration does not match expected size: " << configBuf.size() << " received, " << sizeof(*mFEElightConfig) << " expected";
  }
  mFEElightConfig = reinterpret_cast<const TOFFEElightConfig*>(configBuf.data());
}

//_______________________________________________________________

int TOFFEElightReader::parseFEElightConfig(bool verbose)
{
  // parse FEElight config
  // loops over all FEE channels, checks whether they are enabled
  // and sets channel enabled

  mFEElightInfo.resetAll();

  int version = mFEElightConfig->mVersion;
  int runNumber = mFEElightConfig->mRunNumber;
  int runType = mFEElightConfig->mRunType;

  mFEElightInfo.mVersion = version;
  mFEElightInfo.mRunNumber = runNumber;
  mFEElightInfo.mRunType = runType;

  int nEnabled = 0, index;
  const TOFFEEchannelConfig* channelConfig = nullptr;
  for (int crateId = 0; crateId < Geo::kNCrate; crateId++) {
    for (int trmId = 0; trmId < Geo::kNTRM - 2; trmId++) { // in O2, the number of TRMs is 12, but in the FEE world it is 10
      for (int chainId = 0; chainId < Geo::kNChain; chainId++) {
        for (int tdcId = 0; tdcId < Geo::kNTdc; tdcId++) {
          for (int channelId = 0; channelId < Geo::kNCh; channelId++) {
            channelConfig = mFEElightConfig->getChannelConfig(crateId, trmId, chainId, tdcId, channelId);
            if (verbose) {
              LOG(INFO) << "Processing electronic channel with indices: crate = " << crateId << ", trm = " << trmId << ", chain = "
                        << chainId << ", tdc = " << tdcId << ", tdcChannel = " << channelId << " -> " << channelConfig;
            }
            if (channelConfig) {
              if (!channelConfig->isEnabled()) {
                continue;
              }
              // get index DO from crate, trm, chain, tdc, tdcchannel
              index = Geo::getCHFromECH(Geo::getECHFromIndexes(crateId, trmId + 3, chainId, tdcId, channelId)); // in O2, the TRM index is shifted by 3 because it corresponds to the VME slot
              if (index == -1) {
                continue;
              }
              nEnabled++;
              LOG(INFO) << "Enabling channel " << index;
              mFEElightInfo.mChannelEnabled[index] = channelConfig->isEnabled();
              mFEElightInfo.mMatchingWindow[index] = channelConfig->mMatchingWindow;
              mFEElightInfo.mLatencyWindow[index] = channelConfig->mLatencyWindow;
            }
          }
        }
      }
    }
  }

  const TOFFEEtriggerConfig* triggerConfig = nullptr;
  for (Int_t iddl = 0; iddl < TOFFEElightConfig::NTRIGGERMAPS; iddl++) {
    triggerConfig = mFEElightConfig->getTriggerConfig(iddl);
    if (verbose) {
      LOG(INFO) << "Processing trigger config " << iddl << ": " << triggerConfig;
    }
    if (triggerConfig) {
      mFEElightInfo.mTriggerMask[iddl] = triggerConfig->mStatusMap;
    }
  }

  return nEnabled;
}
