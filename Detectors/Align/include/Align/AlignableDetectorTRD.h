// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace align
{

class AlignableDetectorTRD : public AlignableDetector
{
 public:
  //
  enum { kCalibNRCCorrDzDtgl, // correction parameter for NonRC tracklets
         kCalibDVT,           // global correction to Vdrift*t
         kNCalibParams };     // calibration parameters
  //
  AlignableDetectorTRD(const char* title = "");
  virtual ~AlignableDetectorTRD();
  //
  virtual void defineVolumes();
  virtual void Print(const Option_t* opt = "") const;
  //
  bool AcceptTrack(const AliESDtrack* trc, int trtype) const;
  //
  virtual const char* getCalibDOFName(int i) const;
  //
  virtual void writePedeInfo(FILE* parOut, const Option_t* opt = "") const;
  //
  void SetNonRCCorrDzDtgl(double v = 0) { fNonRCCorrDzDtgl = v; }
  double GetNonRCCorrDzDtgl() const { return fNonRCCorrDzDtgl; }
  double GetNonRCCorrDzDtglWithCal() const { return GetNonRCCorrDzDtgl() + getParVal(kCalibNRCCorrDzDtgl); }
  //
  void SetCorrDVT(double v = 0) { fCorrDVT = 0; }
  double GetCorrDVT() const { return fCorrDVT; }
  double GetCorrDVTWithCal() const { return GetCorrDVT() + getParVal(kCalibDVT); }
  //
  virtual double getCalibDOFVal(int id) const;
  virtual double getCalibDOFValWithCal(int id) const;
  //
  const double* GetExtraErrRC() const { return fExtraErrRC; }
  void SetExtraErrRC(double y = 0.2, double z = 1.0)
  {
    fExtraErrRC[0] = y;
    fExtraErrRC[1] = z;
  }
  //
 protected:
  //
  // -------- dummies --------
  AlignableDetectorTRD(const AlignableDetectorTRD&);
  AlignableDetectorTRD& operator=(const AlignableDetectorTRD&);
  //
 protected:
  //
  double fNonRCCorrDzDtgl; // correction in Z for non-crossing tracklets
  double fCorrDVT;         // correction to Vdrift*t
  double fExtraErrRC[2];   // extra errors for RC tracklets
  //
  static const char* fgkCalibDOFName[kNCalibParams];
  //
  ClassDef(AlignableDetectorTRD, 1);
};
} // namespace align
} // namespace o2
#endif
