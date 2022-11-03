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

#include "EMCALCalib/TriggerSTUDCS.h"

#include <fairlogger/Logger.h>

#include <bitset>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace o2::emcal;

TriggerSTUDCS::TriggerSTUDCS(std::array<int, 3> Gammahigh, std::array<int, 3> Jethigh,
                             std::array<int, 3> Gammalow, std::array<int, 3> Jetlow,
                             int rawData, int region, int fw, int patchSize, int median,
                             std::array<int, 4> phosScale) : mGammaHigh(Gammahigh),
                                                             mJetHigh(Jethigh),
                                                             mGammaLow(Gammalow),
                                                             mJetLow(Jetlow),
                                                             mGetRawData(rawData),
                                                             mRegion(region),
                                                             mFw(fw),
                                                             mPatchSize(patchSize),
                                                             mMedian(median),
                                                             mPHOSScale(phosScale)
{
}

bool TriggerSTUDCS::operator==(const TriggerSTUDCS& other) const
{
  return (mGetRawData == other.mGetRawData) && (mRegion == other.mRegion) &&
         (mFw == other.mFw) && (mPatchSize == other.mPatchSize) && (mMedian == other.mMedian) &&
         (mPHOSScale == other.mPHOSScale) && (mGammaHigh == other.mGammaHigh) &&
         (mJetHigh == other.mJetHigh) && (mGammaLow == other.mGammaLow) && (mJetLow == other.mJetLow);
}

void TriggerSTUDCS::PrintStream(std::ostream& stream) const
{
  stream << "PatchSize: " << mPatchSize
         << ", GetRawData: " << mGetRawData
         << ", Region: 0x" << std::hex << mRegion << std::dec << " (b'" << std::bitset<sizeof(mRegion) * 8>(mRegion) << ")"
         << ", Median: " << mMedian
         << ", Firmware: 0x" << std::hex << mFw << std::dec << std::endl;
  stream << "Gamma High: (" << mGammaHigh[0] << ", " << mGammaHigh[1] << ", " << mGammaHigh[2] << ")" << std::endl;
  stream << "Gamma Low:  (" << mGammaLow[0] << ", " << mGammaLow[1] << ", " << mGammaLow[2] << ")" << std::endl;
  stream << "Jet High:   (" << mJetHigh[0] << ", " << mJetHigh[1] << ", " << mJetHigh[2] << ")" << std::endl;
  stream << "Jet Low:    (" << mJetLow[0] << ", " << mJetLow[1] << ", " << mJetLow[2] << ")" << std::endl;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const TriggerSTUDCS& stu)
{
  stu.PrintStream(stream);
  return stream;
}

std::string TriggerSTUDCS::toJSON() const
{
  std::stringstream jsonstring;
  jsonstring << "{"
             << "\"mGammaHigh\":[[" << mGammaHigh[0] << "," << mGammaHigh[1] << "," << mGammaHigh[2] << "],[" << mGammaLow[0] << "," << mGammaLow[1] << "," << mGammaLow[2] << "]],"
             << "\"mJetHigh\":[[" << mJetHigh[0] << "," << mJetHigh[1] << "," << mJetHigh[2] << "],[" << mJetLow[0] << "," << mJetLow[1] << "," << mJetLow[2] << "]],"
             << "\"mRawData\":" << mGetRawData << ","
             << "\"mRegion\":" << mRegion << ","
             << "\"mFirmware\":" << mFw << ","
             << "\"mMedian\":" << mMedian << ","
             << "\"mPHOSScale\":[" << mPHOSScale[0] << "," << mPHOSScale[1] << "," << mPHOSScale[2] << "," << mPHOSScale[3] << "]"
             << "}";

  return jsonstring.str();
}
