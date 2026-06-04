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

#include <TOFCalibration/TOFFEElightReader.h>
#include "Framework/Logger.h"
#include <TString.h>
#include <TSystem.h>
#include <fstream>

using namespace o2::tof;

void TOFFEElightReader::loadFEElightConfig(const char* fileName)
{
  // load FEElight config

  TString expandedFileName = fileName;
  gSystem->ExpandPathName(expandedFileName);
  std::ifstream is;
  is.open(expandedFileName.Data(), std::ios::binary);
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
    LOG(fatal) << "Incoming message with TOFFEE configuration does not match expected size: " << configBuf.size() << " received, " << sizeof(*mFEElightConfig) << " expected";
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
              LOG(info) << "Processing electronic channel with indices: crate = " << crateId << ", trm = " << trmId << ", chain = "
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
              if (verbose) {
                LOG(info) << "Enabling channel " << index;
              }
              mFEElightInfo.mChannelEnabled[index] = channelConfig->isEnabled();
              mFEElightInfo.mMatchingWindow[index] = channelConfig->mMatchingWindow;
              mFEElightInfo.mLatencyWindow[index] = channelConfig->mLatencyWindow;
            }
          }
        }
      }
    }
  }

  const int istripInPlate[Geo::NSECTORS] = {Geo::NSTRIPC, Geo::NSTRIPB, Geo::NSTRIPA, Geo::NSTRIPB, Geo::NSTRIPC};
  const int channelInSector = Geo::NPADS * Geo::NSTRIPXSECTOR;
  for (int isector = 0; isector < Geo::NSECTORS; isector++) {
    int nstripInPrevPlates = 0;
    for (int iplate = 0; iplate < Geo::NPLATES; iplate++) {
      unsigned int mask = mFEElightConfig->getHVConfig(isector, iplate);
      for (int istrip = 0; istrip < istripInPlate[iplate]; istrip++) {
        bool isActive = mask & 1; // check first bit/current_strip
        mask /= 2;                // move to the next bit/strip

        if (!isActive) { // switch off all channels in this strip
          int index0 = isector * channelInSector + (nstripInPrevPlates + istrip) * Geo::NPADS;
          int indexF = index0 + Geo::NPADS;
          for (int index = index0; index < indexF; index++) {
            mFEElightInfo.mChannelEnabled[index] = 0;
          }
        }
      }
      nstripInPrevPlates += istripInPlate[iplate];
    }
  }

  const TOFFEEtriggerConfig* triggerConfig = nullptr;
  for (Int_t iddl = 0; iddl < TOFFEElightConfig::NTRIGGERMAPS; iddl++) {
    triggerConfig = mFEElightConfig->getTriggerConfig(iddl);
    if (verbose) {
      LOG(info) << "Processing trigger config " << iddl << ": " << triggerConfig;
    }
    if (triggerConfig) {
      mFEElightInfo.mTriggerMask[iddl] = triggerConfig->mStatusMap;
    }
  }

  return nEnabled;
}
