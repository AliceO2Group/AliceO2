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

#ifndef O2_MFT_DATAPROCESSOR_H
#define O2_MFT_DATAPROCESSOR_H

/// @file   DCSMFTDataProcessorSpec.h
/// @brief  MFT Processor for DCS Data Points

#include <TString.h>
#include <unordered_map>
#include <iostream>

namespace o2
{
namespace mft
{

class DCSNameResolver
{

 public:
  void init()
  {

    mDictAlias2Full.clear();
    mDictFull2Alias.clear();

    for (int iH = 0; iH < 2; ++iH) {
      for (int iD = 0; iD < 5; ++iD) {
        for (int iF = 0; iF < 2; ++iF) {
          for (int iZ = 0; iZ < 4; ++iZ) {

            mDictFull2Alias.emplace(Form("mft_main:MFT_PSU_Zone/H%dD%dF%dZ%d.Monitoring.Current.Analog", iH, iD, iF, iZ), Form("MFT_PSU_ZONE/H%d/D%d/F%d/Z%d/Current/Analog", iH, iD, iF, iZ));
            mDictFull2Alias.emplace(Form("mft_main:MFT_PSU_Zone/H%dD%dF%dZ%d.Monitoring.Current.BackBias", iH, iD, iF, iZ), Form("MFT_PSU_ZONE/H%d/D%d/F%d/Z%d/Current/BackBias", iH, iD, iF, iZ));
            mDictFull2Alias.emplace(Form("mft_main:MFT_PSU_Zone/H%dD%dF%dZ%d.Monitoring.Current.Digital", iH, iD, iF, iZ), Form("MFT_PSU_ZONE/H%d/D%d/F%d/Z%d/Current/Digital", iH, iD, iF, iZ));
            mDictFull2Alias.emplace(Form("mft_main:MFT_PSU_Zone/H%dD%dF%dZ%d.Monitoring.Voltage.BackBias", iH, iD, iF, iZ), Form("MFT_PSU_ZONE/H%d/D%d/F%d/Z%d/Voltage/BackBias", iH, iD, iF, iZ));

            mDictAlias2Full.emplace(Form("MFT_PSU_ZONE/H%d/D%d/F%d/Z%d/Current/Analog", iH, iD, iF, iZ), Form("mft_main:MFT_PSU_Zone/H%dD%dF%dZ%d.Monitoring.Current.Analog", iH, iD, iF, iZ));
            mDictAlias2Full.emplace(Form("MFT_PSU_ZONE/H%d/D%d/F%d/Z%d/Current/BackBias", iH, iD, iF, iZ), Form("mft_main:MFT_PSU_Zone/H%dD%dF%dZ%d.Monitoring.Current.BackBias", iH, iD, iF, iZ));
            mDictAlias2Full.emplace(Form("MFT_PSU_ZONE/H%d/D%d/F%d/Z%d/Current/Digital", iH, iD, iF, iZ), Form("mft_main:MFT_PSU_Zone/H%dD%dF%dZ%d.Monitoring.Current.Digital", iH, iD, iF, iZ));
            mDictAlias2Full.emplace(Form("MFT_PSU_ZONE/H%d/D%d/F%d/Z%d/Voltage/BackBias", iH, iD, iF, iZ), Form("mft_main:MFT_PSU_Zone/H%dD%dF%dZ%d.Monitoring.Voltage.BackBias", iH, iD, iF, iZ));
          }
        }
      }
    }

    mDictAlias2Full.emplace("MFT_RU_LV/H0/D0/F0/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel000.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D0/F0/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel001.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D0/F0/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel002.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D0/F0/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel003.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D1/F0/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel004.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D1/F0/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel005.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D1/F0/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel006.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D1/F0/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel007.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D2/F0/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel008.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D2/F0/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel009.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D2/F0/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel010.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D2/F0/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel011.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D3/F0/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel000.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D3/F0/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel001.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D3/F0/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel002.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D3/F0/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel003.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D4/F0/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel004.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D4/F0/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel005.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D4/F0/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel006.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D4/F0/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel007.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D0/F1/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel000.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D0/F1/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel001.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D0/F1/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel002.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D0/F1/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel003.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D1/F1/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel004.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D1/F1/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel005.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D1/F1/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel006.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D1/F1/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel007.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D2/F1/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel008.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D2/F1/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel009.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D2/F1/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel010.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D2/F1/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel011.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D3/F1/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel000.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D3/F1/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel001.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D3/F1/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel002.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D3/F1/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel003.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D4/F1/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel004.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D4/F1/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel005.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D4/F1/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel006.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H0/D4/F1/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel007.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D0/F0/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel000.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D0/F0/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel001.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D0/F0/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel002.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D0/F0/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel003.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D1/F0/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel004.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D1/F0/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel005.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D1/F0/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel006.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D1/F0/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel007.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D2/F0/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel008.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D2/F0/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel009.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D2/F0/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel010.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D2/F0/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel011.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D3/F0/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel000.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D3/F0/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel001.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D3/F0/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel002.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D3/F0/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel003.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D4/F0/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel004.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D4/F0/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel005.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D4/F0/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel006.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D4/F0/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel007.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D0/F1/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel000.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D0/F1/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel001.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D0/F1/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel002.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D0/F1/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel003.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D1/F1/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel004.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D1/F1/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel005.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D1/F1/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel006.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D1/F1/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel007.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D2/F1/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel008.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D2/F1/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel009.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D2/F1/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel010.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D2/F1/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel011.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D3/F1/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel000.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D3/F1/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel001.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D3/F1/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel002.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D3/F1/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel003.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D4/F1/Z0/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel004.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D4/F1/Z1/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel005.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D4/F1/Z2/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel006.actual.iMon");
    mDictAlias2Full.emplace("MFT_RU_LV/H1/D4/F1/Z3/iMon", "mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel007.actual.iMon");

    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel000.actual.iMon", "MFT_RU_LV/H0/D0/F0/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel001.actual.iMon", "MFT_RU_LV/H0/D0/F0/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel002.actual.iMon", "MFT_RU_LV/H0/D0/F0/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel003.actual.iMon", "MFT_RU_LV/H0/D0/F0/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel004.actual.iMon", "MFT_RU_LV/H0/D1/F0/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel005.actual.iMon", "MFT_RU_LV/H0/D1/F0/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel006.actual.iMon", "MFT_RU_LV/H0/D1/F0/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel007.actual.iMon", "MFT_RU_LV/H0/D1/F0/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel008.actual.iMon", "MFT_RU_LV/H0/D2/F0/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel009.actual.iMon", "MFT_RU_LV/H0/D2/F0/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel010.actual.iMon", "MFT_RU_LV/H0/D2/F0/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel011.actual.iMon", "MFT_RU_LV/H0/D2/F0/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel000.actual.iMon", "MFT_RU_LV/H0/D3/F0/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel001.actual.iMon", "MFT_RU_LV/H0/D3/F0/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel002.actual.iMon", "MFT_RU_LV/H0/D3/F0/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel003.actual.iMon", "MFT_RU_LV/H0/D3/F0/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel004.actual.iMon", "MFT_RU_LV/H0/D4/F0/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel005.actual.iMon", "MFT_RU_LV/H0/D4/F0/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel006.actual.iMon", "MFT_RU_LV/H0/D4/F0/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel007.actual.iMon", "MFT_RU_LV/H0/D4/F0/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel000.actual.iMon", "MFT_RU_LV/H0/D0/F1/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel001.actual.iMon", "MFT_RU_LV/H0/D0/F1/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel002.actual.iMon", "MFT_RU_LV/H0/D0/F1/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel003.actual.iMon", "MFT_RU_LV/H0/D0/F1/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel004.actual.iMon", "MFT_RU_LV/H0/D1/F1/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel005.actual.iMon", "MFT_RU_LV/H0/D1/F1/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel006.actual.iMon", "MFT_RU_LV/H0/D1/F1/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel007.actual.iMon", "MFT_RU_LV/H0/D1/F1/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel008.actual.iMon", "MFT_RU_LV/H0/D2/F1/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel009.actual.iMon", "MFT_RU_LV/H0/D2/F1/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel010.actual.iMon", "MFT_RU_LV/H0/D2/F1/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel011.actual.iMon", "MFT_RU_LV/H0/D2/F1/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel000.actual.iMon", "MFT_RU_LV/H0/D3/F1/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel001.actual.iMon", "MFT_RU_LV/H0/D3/F1/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel002.actual.iMon", "MFT_RU_LV/H0/D3/F1/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel003.actual.iMon", "MFT_RU_LV/H0/D3/F1/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel004.actual.iMon", "MFT_RU_LV/H0/D4/F1/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel005.actual.iMon", "MFT_RU_LV/H0/D4/F1/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel006.actual.iMon", "MFT_RU_LV/H0/D4/F1/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel007.actual.iMon", "MFT_RU_LV/H0/D4/F1/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel000.actual.iMon", "MFT_RU_LV/H1/D0/F0/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel001.actual.iMon", "MFT_RU_LV/H1/D0/F0/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel002.actual.iMon", "MFT_RU_LV/H1/D0/F0/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel003.actual.iMon", "MFT_RU_LV/H1/D0/F0/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel004.actual.iMon", "MFT_RU_LV/H1/D1/F0/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel005.actual.iMon", "MFT_RU_LV/H1/D1/F0/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel006.actual.iMon", "MFT_RU_LV/H1/D1/F0/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel007.actual.iMon", "MFT_RU_LV/H1/D1/F0/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel008.actual.iMon", "MFT_RU_LV/H1/D2/F0/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel009.actual.iMon", "MFT_RU_LV/H1/D2/F0/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel010.actual.iMon", "MFT_RU_LV/H1/D2/F0/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel011.actual.iMon", "MFT_RU_LV/H1/D2/F0/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel000.actual.iMon", "MFT_RU_LV/H1/D3/F0/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel001.actual.iMon", "MFT_RU_LV/H1/D3/F0/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel002.actual.iMon", "MFT_RU_LV/H1/D3/F0/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel003.actual.iMon", "MFT_RU_LV/H1/D3/F0/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel004.actual.iMon", "MFT_RU_LV/H1/D4/F0/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel005.actual.iMon", "MFT_RU_LV/H1/D4/F0/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel006.actual.iMon", "MFT_RU_LV/H1/D4/F0/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel007.actual.iMon", "MFT_RU_LV/H1/D4/F0/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel000.actual.iMon", "MFT_RU_LV/H1/D0/F1/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel001.actual.iMon", "MFT_RU_LV/H1/D0/F1/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel002.actual.iMon", "MFT_RU_LV/H1/D0/F1/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel003.actual.iMon", "MFT_RU_LV/H1/D0/F1/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel004.actual.iMon", "MFT_RU_LV/H1/D1/F1/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel005.actual.iMon", "MFT_RU_LV/H1/D1/F1/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel006.actual.iMon", "MFT_RU_LV/H1/D1/F1/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel007.actual.iMon", "MFT_RU_LV/H1/D1/F1/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel008.actual.iMon", "MFT_RU_LV/H1/D2/F1/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel009.actual.iMon", "MFT_RU_LV/H1/D2/F1/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel010.actual.iMon", "MFT_RU_LV/H1/D2/F1/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel011.actual.iMon", "MFT_RU_LV/H1/D2/F1/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel000.actual.iMon", "MFT_RU_LV/H1/D3/F1/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel001.actual.iMon", "MFT_RU_LV/H1/D3/F1/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel002.actual.iMon", "MFT_RU_LV/H1/D3/F1/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel003.actual.iMon", "MFT_RU_LV/H1/D3/F1/Z3/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel004.actual.iMon", "MFT_RU_LV/H1/D4/F1/Z0/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel005.actual.iMon", "MFT_RU_LV/H1/D4/F1/Z1/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel006.actual.iMon", "MFT_RU_LV/H1/D4/F1/Z2/iMon");
    mDictFull2Alias.emplace("mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel007.actual.iMon", "MFT_RU_LV/H1/D4/F1/Z3/iMon");
  }

  std::string& getFullName(const std::string& alias)
  {
    return mDictAlias2Full[alias];
  }

  std::string& getAlias(const std::string& full)
  {
    return mDictFull2Alias[full];
  }

 private:
  std::unordered_map<std::string, std::string> mDictAlias2Full;
  std::unordered_map<std::string, std::string> mDictFull2Alias;

  ClassDefNV(DCSNameResolver, 1);

}; // end class
} // namespace mft
} // namespace o2

#endif
