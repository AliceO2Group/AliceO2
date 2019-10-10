// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_CALONLINEGAINTABLES_H
#define O2_TRD_CALONLINEGAINTABLES_H

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for online gain tables.                            //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  Most things were stored in AliTRDOnlineGainTable{,ROC,MCM}.h             //
///////////////////////////////////////////////////////////////////////////////

#include <array>
#include "TRDBase/TRDGeometry.h"
class FeeParam;

namespace o2
{
namespace trd
{
class CalOnlineGainTables
{
 public:
  CalOnlineGainTables() = default;
  ~CalOnlineGainTables() = default;
  // get and set the various values stored internally in the gain tables.
  float getGainCorrectionFactor(int det, int row, int col) const;
  float getGainCorrectionFactor(int sector, int stack, int layer, int row, int col) const { return getGainCorrectionFactor(TRDGeometry::getDetector(sector, stack, layer), row, col); };
  short getAdcdac(int det, int row, int col) const;
  short getAdcdac(int sector, int stack, int layer, int row, int col) const { return getAdcdac(TRDGeometry::getDetector(sector, stack, layer), row, col); };
  float getMCMGain(int det, int row, int col) const;
  float getMCMGain(int sector, int stack, int layer, int row, int col) const { return getMCMGain(TRDGeometry::getDetector(sector, stack, layer), row, col); };
  short getFGAN(int det, int row, int col) const;
  short getFGAN(int sector, int stack, int layer, int row, int col) const { return getFGAN(TRDGeometry::getDetector(sector, stack, layer), row, col); };
  short getFGFN(int det, int row, int col) const;
  short getFGFN(int sector, int stack, int layer, int row, int col) const { return getFGFN(TRDGeometry::getDetector(sector, stack, layer), row, col); };
  void setGainCorrectionFactor(int det, int row, int col, float gain);
  void setGainCorrectionFactor(int sector, int stack, int layer, int row, int col, float gain) { setGainCorrectionFactor(TRDGeometry::getDetector(sector, stack, layer), row, col, gain); };
  void setAdcdac(int det, int row, int col, short gain);
  void setAdcdac(int sector, int stack, int layer, int row, int col, short gain) { setAdcdac(TRDGeometry::getDetector(sector, stack, layer), row, col, gain); };
  void setMCMGain(int det, int row, int col, float gain);
  void setMCMGain(int sector, int stack, int layer, int row, int col, float gain) { setMCMGain(TRDGeometry::getDetector(sector, stack, layer), row, col, gain); };
  void setFGAN(int det, int row, int col, short gain);
  void setFGAN(int sector, int stack, int layer, int row, int col, short gain) { setFGAN(TRDGeometry::getDetector(sector, stack, layer), row, col, gain); };
  void setFGFN(int det, int row, int col, short gain);
  void setFGFN(int sector, int stack, int layer, int row, int col, short gain) { setFGFN(TRDGeometry::getDetector(sector, stack, layer), row, col, gain); };

  // these 4 are used primarily to reading in from run2, might have other uses.
  void setAdcdac(int arrayoffset, short adc) { mGainTable[arrayoffset].mAdcdac = adc; };
  void setMCMGain(int arrayoffset, float gain) { mGainTable[arrayoffset].mMCMGain = gain; };
  void setFGAN(int arrayoffset, int channel, short gain) { mGainTable[arrayoffset].mFGAN[channel] = gain; };
  void setFGFN(int arrayoffset, int channel, short gain) { mGainTable[arrayoffset].mFGFN[channel] = gain; };

  // two methods to localise the algorithms replacing many copies of the calculations.
  int getArrayOffset(int det, int row, int col) const;
  int getChannel(int col) const;

  static float UnDef;
  class MCMGain
  {
   public:
    short mAdcdac;                 // Reference voltage of the ADCs  U_Ref =  (1.05V + (fAdcdac/31)*0.4V
    std::array<short, 21> mFGFN{}; // Gain Correction Filter Factor
    std::array<short, 21> mFGAN{}; // Gain Correction Filter Additive
    float mMCMGain;
  };

  std::array<MCMGain, 540 * 128> mGainTable;
};
} // namespace trd
} // namespace o2
#endif
