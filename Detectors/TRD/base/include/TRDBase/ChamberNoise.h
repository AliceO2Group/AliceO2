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

//
class TRDGeometry;

namespace o2
{
namespace trd
{
class ChamberNoise
{
 public:
  enum { kNplan = 6, kNcham = 5, kNsect = 18, kNdet = 540 };
  ChamberNoise() = default;
  ~ChamberNoise() = default;
  //
  float getNoise(int p, int c, int s) const { return mNoise[o2::trd::TRDGeometry::getDetector(p,c,s)];};
  void setNoise(int p, int c, int s, float noise) { mNoise[o2::trd::TRDGeometry::getDetector(p,c,s)]=noise;};
 //bulk gets ? 
 protected:
  std::array<float, kNdet> mNoise{};
};
} // namespace trd
} // namespace o2
#endif
