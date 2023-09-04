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

/// @file   AlignableDetectorTRD.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TRD detector wrapper

#ifndef ALIGNABLEDETECTORTRD_H
#define ALIGNABLEDETECTORTRD_H

#include "Align/AlignableDetector.h"
#include "TRDBase/RecoParam.h"

namespace o2
{
namespace align
{

class AlignableDetectorTRD final : public AlignableDetector
{
 public:
  //
  enum { CalibNRCCorrDzDtgl, // correction parameter for NonRC tracklets
         CalibDVT,           // global correction to Vdrift*t
         NCalibParams };     // calibration parameters
  //
  AlignableDetectorTRD() = default; // RS FIXME do we need default c-tor?
  AlignableDetectorTRD(Controller* ctr);
  ~AlignableDetectorTRD() final = default;
  void defineVolumes() final;
  void Print(const Option_t* opt = "") const final;
  const char* getCalibDOFName(int i) const final;
  //
  void writePedeInfo(FILE* parOut, const Option_t* opt = "") const final;
  void writeLabeledPedeResults(FILE* parOut) const final;
  //
  void setNonRCCorrDzDtgl(double v = 0.) { mNonRCCorrDzDtgl = v; }
  double getNonRCCorrDzDtgl() const { return mNonRCCorrDzDtgl; }
  double getNonRCCorrDzDtglWithCal() const { return getNonRCCorrDzDtgl() + getParVal(CalibNRCCorrDzDtgl); }
  //
  void setCorrDVT(double v = 0) { mCorrDVT = 0; }
  double getCorrDVT() const { return mCorrDVT; }
  double getCorrDVTWithCal() const { return getCorrDVT() + getParVal(CalibDVT); }
  //
  double getCalibDOFVal(int id) const final;
  double getCalibDOFValWithCal(int id) const final;
  //
  const double* getExtraErrRC() const { return mExtraErrRC; }
  void setExtraErrRC(double y = 0.2, double z = 1.0)
  {
    mExtraErrRC[0] = y;
    mExtraErrRC[1] = z;
  }

  int processPoints(GIndex gid, int npntCut, bool inv) final;

 protected:
  o2::trd::RecoParam mRecoParam;    // parameters required for TRD reconstruction
  double mNonRCCorrDzDtgl = 0.;     // correction in Z for non-crossing tracklets
  double mCorrDVT = 0.;             // correction to Vdrift*t
  double mExtraErrRC[2] = {0., 0.}; // extra errors for RC tracklets
  //
  static const char* CalibDOFName[NCalibParams];
  //
  ClassDef(AlignableDetectorTRD, 1);
};
} // namespace align
} // namespace o2
#endif
