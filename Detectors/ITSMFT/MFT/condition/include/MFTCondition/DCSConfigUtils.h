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

#ifndef O2_MFT_DNSUTILS_H
#define O2_MFT_DNSUTILS_H

/// @file   DCSConfigUtils.h
/// @brief  MFT Processor for DCS Config

#include <TString.h>
#include <unordered_map>
#include <iostream>
#include "MFTCondition/DCSConfigInfo.h"

namespace o2
{
namespace mft
{

class DCSConfigUtils
{

 public:
  void init(const DCSConfigInfo& info)
  {
    mData = info.getData();
    mAdd = info.getAdd();
    mType = info.getType();
    mVersion = info.getVersion();
    connectNameAdd();
  }

  void clear()
  {
    mData = 0;
    mAdd = 0;
    mType = 0;
    mVersion = "";
  }

  const int& getData() const
  {
    return mData;
  }
  const int& getAdd() const
  {
    return mAdd;
  }
  const int& getType() const
  {
    return mType;
  }
  const std::string& getVersion() const
  {
    return mVersion;
  }
  const std::string& getName() const
  {
    if (mType == 0) {
      return mMapAddNameRU.find(mAdd)->second;
    } else {
      return mMapAddNameALPIDE.find(mAdd)->second;
    }
  }

  const std::string& getTypeStr() const
  {
    if (mType == 0 || mType == 1) {
      return mTypeNameList[mType];
    } else {
      return mTypeNameList[2];
    }
  }

 private:
  int mData;
  int mAdd;
  int mType;

  std::string mVersion;

  std::string mTypeNameList[3] = {"RU", "ALPIDE", "UNKNOWN"};

  std::unordered_map<int, std::string> mMapAddNameRU;
  std::unordered_map<int, std::string> mMapAddNameALPIDE;

  void connectNameAdd()
  {
    mMapAddNameRU.clear();
    mMapAddNameALPIDE.clear();

    mMapAddNameRU[1046] = "MANCHESTER";
    mMapAddNameRU[4096] = "ENABLE";
    mMapAddNameRU[4097] = "TRIGGER_PERIOD";
    mMapAddNameRU[4098] = "PULSE_nTRIGGER";
    mMapAddNameRU[4099] = "TRIGGER_MIN_DISTANCE";
    mMapAddNameRU[4101] = "OPCODE_GATING";
    mMapAddNameRU[4102] = "TRIGGER_DELAY";
    mMapAddNameRU[4103] = "ENABLE_PACKER_0";
    mMapAddNameRU[4104] = "ENABLE_PACKER_1";
    mMapAddNameRU[4105] = "ENABLE_PACKER_2";
    mMapAddNameRU[4106] = "TRIG_SOURCE";
    mMapAddNameRU[5376] = "TIMEOUT_TO_START";
    mMapAddNameRU[5377] = "TIMEOUT_TO_STOP";
    mMapAddNameRU[5378] = "TIMEOUT_IN_IDLE";
    mMapAddNameRU[5631] = "GBT_LOAD_BALANCING";

    mMapAddNameALPIDE[1] = "Mode Control Register";
    mMapAddNameALPIDE[4] = "FROMU Configration Register 1";
    mMapAddNameALPIDE[5] = "FROMU Configration Register 2";
    mMapAddNameALPIDE[6] = "FROMU Configration Register 3";
    mMapAddNameALPIDE[7] = "FROMU Pulsing Register 1";
    mMapAddNameALPIDE[8] = "FROMU Pulsing Register 2";
    mMapAddNameALPIDE[16] = "CMU&DMU Configration Register";
    mMapAddNameALPIDE[20] = "DTU Configration Register";
    mMapAddNameALPIDE[21] = "DTU DACs Register";
    mMapAddNameALPIDE[24] = "DTU Test Register 1";
    mMapAddNameALPIDE[25] = "DTU Test Register 2";
    mMapAddNameALPIDE[26] = "DTU Test Register 3";
    mMapAddNameALPIDE[1539] = "VCASP";
    mMapAddNameALPIDE[1544] = "VCLIP";
    mMapAddNameALPIDE[1549] = "IBIAS";
    mMapAddNameALPIDE[255] = "VPULSEH";
    mMapAddNameALPIDE[0] = "VPULSEL";
    mMapAddNameALPIDE[1538] = "VRESETD";
    mMapAddNameALPIDE[1548] = "IDB";
    mMapAddNameALPIDE[65535] = "AUTO_ROF, NOISE_MASK, MASK_LEV, MC_HIT, MC_ID";
  }

  ClassDefNV(DCSConfigUtils, 1);

}; // end class
} // namespace mft
} // namespace o2

#endif
