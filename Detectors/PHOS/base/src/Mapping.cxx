// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

Mapping::Mapping(std::basic_string_view<char> path) : mPath(path),
                                                      mInitialized(false)
{
}
//_______________________________________________________
Mapping::ErrorStatus Mapping::hwToAbsId(short ddl, short hwAddr, short& absId, CaloFlag& caloFlag)
{

  if (!mInitialized) {
    LOG(ERROR) << "Mapping not initialized";
    return kNotInitialized;
  }

  if (ddl < 0 || ddl > 14) {
    return kWrongDDL;
  }
  if (hwAddr < 0 || hwAddr >= NMaxHWAddress) {
    return kWrongHWAddress;
  }

  //transform
  absId = mAbsId[ddl][hwAddr];
  caloFlag = mCaloFlag[ddl][hwAddr];

  if (absId > NCHANNELS) {
    absId = 0;
    return kWrongHWAddress;
  }
  return kOK;
}
//_______________________________________________________
Mapping::ErrorStatus Mapping::absIdTohw(short absId, short caloFlag, short& ddl, short& hwAddr)
{

  if (absId < 0 || absId > NCHANNELS) {
    ddl = 0;
    hwAddr = 0;
    return kWrongAbsId;
  }
  if (caloFlag < 0 || caloFlag > 2) {
    ddl = 0;
    hwAddr = 0;
    return kWrongCaloFlag;
  }

  if (!mInitialized) {
    LOG(ERROR) << "Mapping not initialized";
    return kNotInitialized;
  }

  ddl = mAbsToHW[absId][caloFlag][0];
  hwAddr = mAbsToHW[absId][caloFlag][1];
  return kOK;
}
//_______________________________________________________
Mapping::ErrorStatus Mapping::setMapping()
{
  //Read mapping from data files a-la Run2

  o2::phos::Geometry* geom = o2::phos::Geometry::GetInstance();

  std::string p;
  if (mPath.empty()) { //use default path
    p = gSystem->Getenv("O2_ROOT");
    p += "/share/Detectors/PHOS/files";
  } else {
    p = mPath.data();
  }

  for (short m = 0; m < 4; m++) {   //modules
    for (short i = 0; i < 4; i++) { //RCU
      if (m == 0 && (i < 2)) {
        continue; //half of module: only RCU 2,3
      }

      short numberOfChannels = 0;
      short maxHWAddress = 0;
      char fname[255];
      snprintf(fname, 255, "%s/Mod%dRCU%d.data", p.data(), m, i);
      std::ifstream* fIn = new std::ifstream(fname);
      if (!*fIn) {
        LOG(FATAL) << "Missing mapping file " << p << "/Mod" << m << "RCU" << i << ".data";
        return kNotInitialized;
      }
      if (!(*fIn >> numberOfChannels)) {
        LOG(FATAL) << "Syntax of mapping file " << p << "/Mod" << m << "RCU" << i << ".data is wrong: no numberOfChannels";
        return kNotInitialized;
      }
      if (numberOfChannels != NHWPERDDL) {
        LOG(FATAL) << "Unexpected number of channels: " << numberOfChannels << " expecting " << NHWPERDDL << " file " << p << "/Mod" << m << "RCU" << i << ".data is wrong: no numberOfChannels";
        return kNotInitialized;
      }
      if (!(*fIn >> maxHWAddress)) {
        LOG(FATAL) << "Syntax of mapping file " << p << "/Mod" << m << "RCU" << i << ".data is wrong: no maxHWAddress";
        return kNotInitialized;
      }
      if (maxHWAddress > NMaxHWAddress) {
        LOG(FATAL) << "Maximal HW address in file " << maxHWAddress << "larger than array size " << NMaxHWAddress << "for /Mod" << m << "RCU" << i << ".data is wrong: no maxHWAddress";
        return kNotInitialized;
      }

      for (short ich = 0; ich < numberOfChannels; ich++) { // 1792 = 2*896 channels connected to each RCU
        int hwAddress;
        if (!(*fIn >> hwAddress)) {
          LOG(FATAL) << "Syntax of mapping file " << p << "/Mod" << m << "RCU" << i << ".data is wrong: no HWadd for ch " << ich;
          return kNotInitialized;
        }
        if (hwAddress > maxHWAddress) {
          LOG(FATAL) << "Hardware (ALTRO) adress (" << hwAddress << ") outside the range (0 -> " << maxHWAddress << ") !";
          return kNotInitialized;
        }
        int row, col, caloFlag;
        if (!(*fIn >> row >> col >> caloFlag)) {
          LOG(FATAL) << "Syntax of mapping file " << p << "/Mod" << m << "RCU" << i << ".data is wrong:  no (raw col caloFlag)";
          return kNotInitialized;
        }

        if (caloFlag < 0 || caloFlag > 2) {
          LOG(FATAL) << "Wrong CaloFlag value found (" << caloFlag << "). Should be 0, 1, 2 !";
          return kNotInitialized;
        }

        if (caloFlag == 2) { //TODO!!!! TRU mapping not known yet
          continue;
        }

        //convert ddl, col,raw caloFlag to AbsId
        // Converts the absolute numbering into the following array
        //  relid[0] = PHOS Module number
        //  relid[1] = Row number inside a PHOS module (Phi coordinate)
        //  relid[2] = Column number inside a PHOS module (Z coordinate)
        short ddl = 4 * m + i - 2;

        char relid[3] = {(char)m, (char)row, (char)col};
        short absId;
        geom->relToAbsNumbering(relid, absId);

        if (ddl < 0 || ddl >= NDDL) {
          LOG(FATAL) << "Wrong ddl address found (" << ddl << "). Module= " << m << " RCU =" << i;
          return kNotInitialized;
        }

        mAbsId[ddl][hwAddress] = absId;
        mCaloFlag[ddl][hwAddress] = (CaloFlag)caloFlag;

        mAbsToHW[absId][caloFlag][0] = ddl;
        mAbsToHW[absId][caloFlag][1] = hwAddress;
      }
      fIn->close();
    } //RCU
  }   // module
  mInitialized = true;
  return kOK;
}
