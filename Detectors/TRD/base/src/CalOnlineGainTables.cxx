// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for online gain tables.                            //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  Most things were stored in AliTRDOnlineGainTable{,ROC,MCM}.h             //
///////////////////////////////////////////////////////////////////////////////

#include "TRDBase/CalOnlineGainTables.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/FeeParam.h"
#include "DataFormatsTRD/Constants.h"

using namespace o2::trd;
using namespace o2::trd::constants;
using namespace std;

float o2::trd::CalOnlineGainTables::UnDef = -999.0;

int CalOnlineGainTables::getArrayOffset(int det, int row, int col) const
{
  FeeParam* mFeeParam = FeeParam::instance();
  int rob = mFeeParam->getROBfromPad(row, col);
  int mcm = mFeeParam->getMCMfromPad(row, col);
  int detoffset = det * 128; //TODO find this constant from somewhere else max rob=8, max mcm=16, so 7x16+15=127
  int mcmoffset = rob * NMCMROB + mcm;
  return detoffset + mcmoffset;
}

int CalOnlineGainTables::getChannel(int col) const
{
  return 19 - (col % 18); //TODO find both of these constants from somewhere else.
}

int CalOnlineGainTables::getArrayOffsetrm(int det, int rob, int mcm) const
{
  return det * 128 + rob * NMCMROB + mcm;
}

float CalOnlineGainTables::getGainCorrectionFactor(int det, int row, int col) const
{
  //notes for my sanity ...
  //det gives us the [0-540]
  //row and col give us the [0-128] 16*rob+mcm
  int arrayoffset = getArrayOffset(det, row, col);
  int channel = getChannel(col);
  float GainCorrectionFactor = 0.0;
  if (mGainTable[arrayoffset].mAdcdac == 0) {
    if (mGainTable[arrayoffset].mFGFN[channel] < 0) {
      GainCorrectionFactor = -1.0;
    } else if (mGainTable[arrayoffset].mFGFN[channel] > 511) {
      GainCorrectionFactor = CalOnlineGainTables::UnDef;
    } else {
      GainCorrectionFactor = (mGainTable[arrayoffset].mFGFN[channel] / 2048.) + 0.875;
    }
  } else {
    float ADCCorrection = (1. / (1. + ((float)mGainTable[arrayoffset].mAdcdac / 31.) * 0.4 / 1.05));
    GainCorrectionFactor = ADCCorrection * (((mGainTable[arrayoffset].mFGFN[channel]) / 2048.) + 0.875);
  }
  return GainCorrectionFactor;
}

short CalOnlineGainTables::getAdcdacrm(int det, int rob, int mcm) const
{
  return mGainTable[getArrayOffsetrm(det, rob, mcm)].mAdcdac;
}

short CalOnlineGainTables::getAdcdac(int det, int row, int col) const
{
  int arrayoffset = getArrayOffset(det, row, col);
  return mGainTable[arrayoffset].mAdcdac;
};

float CalOnlineGainTables::getMCMGainrm(int det, int rob, int mcm) const
{
  return mGainTable[getArrayOffsetrm(det, rob, mcm)].mMCMGain;
}

float CalOnlineGainTables::getMCMGain(int det, int row, int col) const
{
  int arrayoffset = getArrayOffset(det, row, col);
  return mGainTable[arrayoffset].mMCMGain;
}

short CalOnlineGainTables::getFGANrm(int det, int rob, int mcm, int channel) const
{
  return mGainTable[getArrayOffsetrm(det, rob, mcm)].mFGAN[channel];
}

short CalOnlineGainTables::getFGAN(int det, int row, int col) const
{
  int arrayoffset = getArrayOffset(det, row, col);
  int channel = getChannel(col);
  return mGainTable[arrayoffset].mFGAN[channel];
}

short CalOnlineGainTables::getFGFNrm(int det, int rob, int mcm, int channel) const
{
  return mGainTable[getArrayOffsetrm(det, rob, mcm)].mFGFN[channel];
}

short CalOnlineGainTables::getFGFN(int det, int row, int col) const
{
  int arrayoffset = getArrayOffset(det, row, col);
  int channel = getChannel(col);
  return mGainTable[arrayoffset].mFGFN[channel];
}

void CalOnlineGainTables::setAdcdacrm(int det, int rob, int mcm, short adcdac)
{
  mGainTable[getArrayOffsetrm(det, rob, mcm)].mAdcdac = adcdac;
}

void CalOnlineGainTables::setAdcdac(int det, int row, int col, short adcdac)
{
  int arrayoffset = getArrayOffset(det, row, col);
  mGainTable[arrayoffset].mAdcdac = adcdac;
}

void CalOnlineGainTables::setMCMGainrm(int det, int rob, int mcm, float gain)
{
  mGainTable[getArrayOffsetrm(det, rob, mcm)].mMCMGain = gain;
}

void CalOnlineGainTables::setMCMGain(int det, int row, int col, float gain)
{
  int arrayoffset = getArrayOffset(det, row, col);
  mGainTable[arrayoffset].mMCMGain = gain;
}

void CalOnlineGainTables::setFGANrm(int det, int rob, int mcm, int channel, short gain)
{
  mGainTable[getArrayOffsetrm(det, rob, mcm)].mFGAN[channel] = gain;
}

void CalOnlineGainTables::setFGAN(int det, int row, int col, short gain)
{
  int arrayoffset = getArrayOffset(det, row, col);
  int channel = getChannel(col);
  mGainTable[arrayoffset].mFGAN[channel] = gain;
}

void CalOnlineGainTables::setFGFNrm(int det, int rob, int mcm, int channel, short gain)
{
  mGainTable[getArrayOffsetrm(det, rob, mcm)].mFGFN[channel] = gain;
}

void CalOnlineGainTables::setFGFN(int det, int row, int col, short gain)
{
  int arrayoffset = getArrayOffset(det, row, col);
  int channel = getChannel(col);
  mGainTable[arrayoffset].mFGFN[channel] = gain;
}
