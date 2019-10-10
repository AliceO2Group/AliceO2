// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_CHAMBERNOISE_H
#define O2_TRD_CHAMBERNOISE_H

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for ChamberNoise                                   //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  Originally stored in AliTRDCalDet and instantiated at DetNoise in ocdb   //
///////////////////////////////////////////////////////////////////////////////

#include "TRDBase/TRDSimParam.h"
//
class TRDGeometry;

namespace o2
{
namespace trd
{
class ChamberNoise
{
 public:
  ChamberNoise() = default;
  ~ChamberNoise() = default;
  //
  float getNoise(int p, int c, int s) const { return mNoise[o2::trd::TRDGeometry::getDetector(p, c, s)]; };
  float getNoise(int det) const { return mNoise[det]; };
  void setNoise(int p, int c, int s, float noise) { mNoise[o2::trd::TRDGeometry::getDetector(p, c, s)] = noise; };
  void setNoise(int det, float noise) { mNoise[det] = noise; };
  //bulk gets ?
 protected:
  std::array<float, TRDSimParam::kNdet> mNoise{};
};
} // namespace trd
} // namespace o2
#endif
