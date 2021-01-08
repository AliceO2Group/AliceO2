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
//
//  For my own sanity ... What happened in Run2 is not relevant for the new class.
//  What is relevant is how the data was transfered across....
//  OCDB2CCDB takes the old AliTRDOnlineGainTable* and produces a single
//  array of 540x128. So Det*128+ROB, and each of those has 21 values 1 for each adc.
//
//  Notes:
//  sector, stack, layer === via Geometry Class === Detector. [0,539]
//  rob (read out board) *16 + MCM  == [0,127]   max number of MCM is 16.
///////////////////////////////////////////////////////////////////////////////

#include <array>
#include "TRDBase/Geometry.h"
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
  float getGainCorrectionFactorrm(int det, int rob, int mcm) const;
  float getGainCorrectionFactor(int det, int row, int col) const;
  float getGainCorrectionFactor(int sector, int stack, int layer, int row, int col) const { return getGainCorrectionFactor(Geometry::getDetector(sector, stack, layer), row, col); };
  short getAdcdacrm(int det, int rob, int mcm) const;
  short getAdcdac(int det, int row, int col) const;
  short getAdcdac(int sector, int stack, int layer, int row, int col) const { return getAdcdac(Geometry::getDetector(sector, stack, layer), row, col); };
  float getMCMGainrm(int det, int rob, int mcm) const;
  float getMCMGain(int det, int row, int col) const;
  float getMCMGain(int sector, int stack, int layer, int row, int col) const { return getMCMGain(Geometry::getDetector(sector, stack, layer), row, col); };
  short getFGANrm(int det, int rob, int mcm, int channel) const;
  short getFGAN(int det, int row, int col) const;
  short getFGAN(int sector, int stack, int layer, int row, int col) const { return getFGAN(Geometry::getDetector(sector, stack, layer), row, col); };
  short getFGFNrm(int det, int rob, int mcm, int channel) const;
  short getFGFN(int det, int row, int col) const;
  short getFGFN(int sector, int stack, int layer, int row, int col) const { return getFGFN(Geometry::getDetector(sector, stack, layer), row, col); };
  void setGainCorrectionFactorrm(int det, int rob, int mcm, float gain);
  void setGainCorrectionFactor(int det, int row, int col, float gain);
  void setGainCorrectionFactor(int sector, int stack, int layer, int row, int col, float gain) { setGainCorrectionFactor(Geometry::getDetector(sector, stack, layer), row, col, gain); };
  void setAdcdacrm(int det, int rob, int mcm, short gain);
  void setAdcdac(int det, int row, int col, short gain);
  void setAdcdac(int sector, int stack, int layer, int row, int col, short gain) { setAdcdac(Geometry::getDetector(sector, stack, layer), row, col, gain); };
  void setMCMGainrm(int det, int rob, int mcm, float gain);
  void setMCMGain(int det, int row, int col, float gain);
  void setMCMGain(int sector, int stack, int layer, int row, int col, float gain) { setMCMGain(Geometry::getDetector(sector, stack, layer), row, col, gain); };
  void setFGANrm(int det, int rob, int mcm, int channel, short gain);
  void setFGAN(int det, int row, int col, short gain);
  void setFGAN(int sector, int stack, int layer, int row, int col, short gain) { setFGAN(Geometry::getDetector(sector, stack, layer), row, col, gain); };
  void setFGFNrm(int det, int rob, int mcm, int channel, short gain);
  void setFGFN(int det, int row, int col, short gain);
  void setFGFN(int sector, int stack, int layer, int row, int col, short gain) { setFGFN(Geometry::getDetector(sector, stack, layer), row, col, gain); };

  // these 4 are used primarily to reading in from run2 ocdb, might have wider uses.
  void setAdcdac(int arrayoffset, short adc) { mGainTable[arrayoffset].mAdcdac = adc; };
  void setMCMGain(int arrayoffset, float gain) { mGainTable[arrayoffset].mMCMGain = gain; };
  void setFGAN(int arrayoffset, int channel, short gain) { mGainTable[arrayoffset].mFGAN[channel] = gain; };
  void setFGFN(int arrayoffset, int channel, short gain) { mGainTable[arrayoffset].mFGFN[channel] = gain; };

  // two methods to localise the algorithms replacing many copies of the calculations.
  int getArrayOffset(int det, int row, int col) const;
  int getArrayOffsetrm(int det, int row, int col) const;
  int getChannel(int col) const;

  static float UnDef;
  class MCMGain
  {
   public:
    short mAdcdac{0};              // Reference voltage of the ADCs  U_Ref =  (1.05V + (fAdcdac/31)*0.4V
    std::array<short, 21> mFGFN{}; // Gain Correction Filter Factor
    std::array<short, 21> mFGAN{}; // Gain Correction Filter Additive
    float mMCMGain{0};
  };

  std::array<MCMGain, 540 * 128> mGainTable;
};
} // namespace trd
} // namespace o2
#endif
