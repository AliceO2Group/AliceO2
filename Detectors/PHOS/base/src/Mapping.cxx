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

/// \file Mapping.cxx
/// \author Dmitri Peresunko

#include <fstream>
#include "TSystem.h"
#include "FairLogger.h"
#include "PHOSBase/Mapping.h"
#include "PHOSBase/Geometry.h"

using namespace o2::phos;
Mapping* Mapping::sMapping = nullptr;
//_______________________________________________________
Mapping::Mapping(std::basic_string_view<char> path) : mPath(path),
                                                      mInitialized(false)
{
}
//_______________________________________________________
Mapping* Mapping::Instance()
{
  if (sMapping) {
    return sMapping;
  } else {
    sMapping = new Mapping();
    sMapping->setMapping();
    return sMapping;
  }
}
//_______________________________________________________
Mapping* Mapping::Instance(std::basic_string_view<char> path)
{
  if (sMapping) {
    if (sMapping->mPath == path) {
      return sMapping;
    } else {
      delete sMapping;
    }
  }
  sMapping = new Mapping(path);
  sMapping->setMapping();
  return sMapping;
}
//_______________________________________________________
Mapping::ErrorStatus Mapping::hwToAbsId(short ddl, short hwAddr, short& absId, CaloFlag& caloFlag) const
{

  if (!mInitialized) {
    LOG(error) << "Mapping not initialized";
    return kNotInitialized;
  }

  if (ddl < 0 || ddl > 14) {
    return kWrongDDL;
  }

  if ((hwAddr >= 112 && hwAddr < 128) || (hwAddr >= 2159 && hwAddr < 2176)) { // TRU flags
    caloFlag = kTRU;
    absId = -1;
    return kOK;
  }
  if (hwAddr < 0 || hwAddr >= NMaxHWAddress) {
    return kWrongHWAddress;
  }

  // transform
  absId = mAbsId[ddl][hwAddr];
  caloFlag = mCaloFlag[ddl][hwAddr];

  if (caloFlag == 2) {
    absId += NCHANNELS;
  }
  if (caloFlag < 2 && (absId > NCHANNELS || absId <= 1792)) {
    absId = 0;
    return kWrongHWAddress;
  }
  return kOK;
}
//_______________________________________________________
Mapping::ErrorStatus Mapping::absIdTohw(short absId, short caloFlag, short& ddl, short& hwAddr) const
{

  if (caloFlag < 0 || caloFlag > 2) {
    ddl = 0;
    hwAddr = 0;
    return kWrongCaloFlag;
  }
  if (caloFlag < 2) {
    if (absId <= 1792 || absId > NCHANNELS) {
      ddl = 0;
      hwAddr = 0;
      return kWrongAbsId;
    }
  } else { // TRU: absId goes after readout ones
    absId -= NCHANNELS;
    if (absId < 1 || absId > NTRUReadoutChannels) {
      ddl = 0;
      hwAddr = 0;
      return kWrongAbsId;
    }
  }

  if (!mInitialized) {
    LOG(error) << "Mapping not initialized";
    return kNotInitialized;
  }

  ddl = mAbsToHW[absId - 1][caloFlag][0];
  hwAddr = mAbsToHW[absId - 1][caloFlag][1];
  return kOK;
}
//_______________________________________________________
Mapping::ErrorStatus Mapping::setMapping()
{
  // Read mapping from data files a-la Run2

  std::string p;
  if (mPath.empty()) { // use default path
    p = gSystem->Getenv("O2_ROOT");
    p += "/share/Detectors/PHOS/files";
  } else {
    p = mPath.data();
  }

  for (short m = 0; m < 4; m++) {   // modules
    for (short i = 0; i < 4; i++) { // RCU
      if (m == 0 && (i < 2)) {
        continue; // half of module: only RCU 2,3
      }

      short numberOfChannels = 0;
      short maxHWAddress = 0;
      std::string fname = fmt::format("{:s}/Mod{:d}RCU{:d}.data", p, m, i);
      std::ifstream fIn(fname);
      if (!fIn.is_open()) {
        LOG(fatal) << "Missing mapping file " << p << "/Mod" << m << "RCU" << i << ".data";
        return kNotInitialized;
      }
      if (!(fIn >> numberOfChannels)) {
        LOG(fatal) << "Syntax of mapping file " << p << "/Mod" << m << "RCU" << i << ".data is wrong: no numberOfChannels";
        return kNotInitialized;
      }
      if (numberOfChannels != NHWPERDDL) {
        LOG(fatal) << "Unexpected number of channels: " << numberOfChannels << " expecting " << NHWPERDDL << " file " << p << "/Mod" << m << "RCU" << i << ".data is wrong: no numberOfChannels";
        return kNotInitialized;
      }
      if (!(fIn >> maxHWAddress)) {
        LOG(fatal) << "Syntax of mapping file " << p << "/Mod" << m << "RCU" << i << ".data is wrong: no maxHWAddress";
        return kNotInitialized;
      }
      if (maxHWAddress > NMaxHWAddress) {
        LOG(fatal) << "Maximal HW address in file " << maxHWAddress << "larger than array size " << NMaxHWAddress << "for /Mod" << m << "RCU" << i << ".data is wrong: no maxHWAddress";
        return kNotInitialized;
      }
      for (short ich = 0; ich < numberOfChannels; ich++) { // 1792 = 2*896 channels connected to each RCU
        int hwAddress;
        if (!(fIn >> hwAddress)) {
          LOG(fatal) << "Syntax of mapping file " << p << "/Mod" << m << "RCU" << i << ".data is wrong: no HWadd for ch " << ich;
          return kNotInitialized;
        }
        if (hwAddress > maxHWAddress) {
          LOG(fatal) << "Hardware (ALTRO) adress (" << hwAddress << ") outside the range (0 -> " << maxHWAddress << ") !";
          return kNotInitialized;
        }
        int row, col, caloFlag;
        if (!(fIn >> row >> col >> caloFlag)) {
          LOG(fatal) << "Syntax of mapping file " << p << "/Mod" << m << "RCU" << i << ".data is wrong:  no (raw col caloFlag)";
          return kNotInitialized;
        }

        if (caloFlag < 0 || caloFlag > 2) {
          LOG(fatal) << "Wrong CaloFlag value found (" << caloFlag << "). Should be 0, 1, 2 !";
          return kNotInitialized;
        }

        // convert ddl, col,raw caloFlag to AbsId
        //  Converts the absolute numbering into the following array
        //   relid[0] = PHOS Module number
        //   relid[1] = Row number inside a PHOS module (Phi coordinate)
        //   relid[2] = Column number inside a PHOS module (Z coordinate)
        short ddl = 4 * m + i - 2;
        if (ddl < 0 || ddl >= NDDL) {
          LOG(fatal) << "Wrong ddl address found (" << ddl << "). Module= " << m << " RCU =" << i;
          return kNotInitialized;
        }

        short absId;
        if (caloFlag < 2) { // readout channels
          char relid[3] = {static_cast<char>(m + 1), static_cast<char>(row + 1), static_cast<char>(col + 1)};
          Geometry::relToAbsNumbering(relid, absId);
        } else { // TRU channels: internal storage of TRU channesl absId-NCHANNELS
          if (isTRUReadoutchannel(hwAddress)) {
            if (hwAddress < 2048) { // branch 28<=z<56
              absId = 1 + ddl * 2 * NTRUBranchReadoutChannels + hwAddress;
            } else { // branch 0<=z<28
              absId = 1 + (ddl * 2 + 1) * NTRUBranchReadoutChannels + hwAddress - 2048;
            }
          } else { // TRU flag channels, no absId
            continue;
          }
        }

        mAbsId[ddl][hwAddress] = absId;
        mCaloFlag[ddl][hwAddress] = (CaloFlag)caloFlag;
        mAbsToHW[absId - 1][caloFlag][0] = ddl;
        mAbsToHW[absId - 1][caloFlag][1] = hwAddress;
      }
      fIn.close();
    } // RCU
  }   // module
  mInitialized = true;
  return kOK;
}
