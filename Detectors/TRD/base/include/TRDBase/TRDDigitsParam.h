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
//  Class containing parameters for digits                                   //
//  2019 - Ported from AliRoot to O2 (Jorge Lopez)                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef O2_TRDDIGITSPARAM_H
#define O2_TRDDIGITSPARAM_H

#include <array>

namespace o2
{
namespace trd
{
class TRDDigitsParam
{

 public:
  TRDDigitsParam() = default;
  ~TRDDigitsParam() = default;

  void setNTimeBins(int det, int ntb) { mNTimeBins[det] = ntb; }
  void setPretriggerPhase(int det, unsigned int phase) { mPretriggerPhase[det] = phase; }
  void setADCbaseline(int det, int baseline) { mADCbaseline[det] = baseline; }

  void setNTimeBinsAll(int ntb)
  {
    for (auto& el : mNTimeBins) {
      el = ntb;
    }
  }
  void setPretriggerPhaseAll(unsigned int phase)
  {
    for (auto& el : mPretriggerPhase) {
      el = phase;
    }
  }
  void setADCbaselineAll(int baseline)
  {
    for (auto& el : mADCbaseline) {
      el = baseline;
    }
  }

  int getNTimeBins(int det) const { return mNTimeBins[det]; }
  unsigned int getPretriggerPhase(int det) const { return mPretriggerPhase[det]; }
  int getADCbaseline(int det) const { return mADCbaseline[det]; }

 protected:
  std::array<int, 540> mNTimeBins;       //  Number of time bins for each detector
  std::array<int, 540> mPretriggerPhase; //  Pretrigger phase for each detector
  std::array<int, 540> mADCbaseline;     //  ADC baseline for each detector, given in ADC channels
};
} // namespace trd
} // namespace o2
#endif
