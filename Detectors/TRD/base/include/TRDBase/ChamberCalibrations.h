// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_CHAMBERCALIBRATIONS_H
#define O2_TRD_CHAMBERCALIBRATIONS_H

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for parameters which are saved frequently(/run)    //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  Most things were stored in AliTRDcalROC,AliTRDcalPad, AliTRDcalDet       //
///////////////////////////////////////////////////////////////////////////////

#include "TRDBase/TRDSimParam.h"
//
class TRDGeometry;

namespace o2
{
namespace trd
{
class ChamberCalibrations
{
 public:
  ChamberCalibrations() = default;
  ~ChamberCalibrations() = default;
  //
  float getVDrift(int p, int c, int s) const { return mVDrift[o2::trd::TRDGeometry::getDetector(p, c, s)]; };
  float getVDrift(int roc) const { return mVDrift[roc]; };
  float getGainFactor(int p, int c, int s) const { return mGainFactor[TRDGeometry::getDetector(p, c, s)]; };
  float getGainFactor(int roc) const { return mGainFactor[roc]; };
  float getT0(int p, int c, int s) const { return mT0[TRDGeometry::getDetector(p, c, s)]; };
  float getT0(int roc) const { return mT0[roc]; };
  float getExB(int p, int c, int s) const { return mExB[TRDGeometry::getDetector(p, c, s)]; };
  float getExB(int roc) const { return mExB[roc]; };
  void setVDrift(int p, int c, int s, float vdrift) { mVDrift[o2::trd::TRDGeometry::getDetector(p, c, s)] = vdrift; };
  void setVDrift(int roc, float vdrift) { mVDrift[roc] = vdrift; };
  void setGainFactor(int p, int c, int s, float gainfactor) { mGainFactor[TRDGeometry::getDetector(p, c, s)] = gainfactor; };
  void setGainFactor(int roc, float gainfactor) { mGainFactor[roc] = gainfactor; };
  void setT0(int p, int c, int s, float t0) { mT0[TRDGeometry::getDetector(p, c, s)] = t0; };
  void setT0(int roc, float t0) { mT0[roc] = t0; };
  void setExB(int p, int c, int s, float exb) { mExB[TRDGeometry::getDetector(p, c, s)] = exb; };
  void setExB(int roc, float exb) { mExB[roc] = exb; };
  //bulk gets ?
  bool init(int run2run = 0);

 protected:
  std::array<float, TRDSimParam::kNdet> mVDrift{};     // mean drift velocity per chamber.
  std::array<float, TRDSimParam::kNdet> mGainFactor{}; // mean gas gain per chamber
  std::array<float, TRDSimParam::kNdet> mT0{};         // Min timeoffset in the chamber
  std::array<float, TRDSimParam::kNdet> mExB{};        //
};
} // namespace trd
} // namespace o2
#endif
