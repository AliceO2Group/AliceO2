// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_VDRIFTEXB_H
#define O2_TRD_VDRIFTEXB_H

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for parameters which are saved frequently(/run)    //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  Most things were stored in AliTRDcalROC,AliTRDcalPad, AliTRDcalDet       //
///////////////////////////////////////////////////////////////////////////////

#include "TRDBase/SimParam.h"
#include "DataFormatsTRD/Constants.h"
#include "TRDBase/Geometry.h"
//
class Geometry;

namespace o2
{
namespace trd
{
class CalVdriftExB
{
 public:
  CalVdriftExB() = default;
  ~CalVdriftExB() = default;
  //
  float getVDrift(int p, int c, int s) const { return mVDrift[o2::trd::Geometry::getDetector(p, c, s)]; };
  float getVDrift(int roc) const { return mVDrift[roc]; };
  float getExB(int p, int c, int s) const { return mExB[Geometry::getDetector(p, c, s)]; };
  float getExB(int roc) const { return mExB[roc]; };
  void setVDrift(int p, int c, int s, float vdrift) { mVDrift[o2::trd::Geometry::getDetector(p, c, s)] = vdrift; };
  void setVDrift(int roc, float vdrift) { mVDrift[roc] = vdrift; };
  void setExB(int p, int c, int s, float exb) { mExB[Geometry::getDetector(p, c, s)] = exb; };
  void setExB(int roc, float exb) { mExB[roc] = exb; };
  //bulk gets ?
  bool init(int run2run = 0);

 protected:
  std::array<float, constants::MAXCHAMBER> mVDrift{};     // mean drift velocity per chamber.
  std::array<float, constants::MAXCHAMBER> mGainFactor{}; // mean gas gain per chamber
  std::array<float, constants::MAXCHAMBER> mT0{};         // Min timeoffset in the chamber
  std::array<float, constants::MAXCHAMBER> mExB{};        //
};
} // namespace trd
} // namespace o2
#endif
