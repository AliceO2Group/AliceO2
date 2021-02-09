// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgDetTRD.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TRD detector wrapper

#ifndef ALIALGDETTRD_H
#define ALIALGDETTRD_H

#include "Align/AliAlgDet.h"

namespace o2
{
namespace align
{

class AliAlgDetTRD : public AliAlgDet
{
 public:
  //
  enum { kCalibNRCCorrDzDtgl, // correction parameter for NonRC tracklets
         kCalibDVT,           // global correction to Vdrift*t
         kNCalibParams };     // calibration parameters
  //
  AliAlgDetTRD(const char* title = "");
  virtual ~AliAlgDetTRD();
  //
  virtual void DefineVolumes();
  virtual void Print(const Option_t* opt = "") const;
  //
  Bool_t AcceptTrack(const AliESDtrack* trc, Int_t trtype) const;
  //
  virtual const char* GetCalibDOFName(int i) const;
  //
  virtual void WritePedeInfo(FILE* parOut, const Option_t* opt = "") const;
  //
  void SetNonRCCorrDzDtgl(double v = 0) { fNonRCCorrDzDtgl = v; }
  Double_t GetNonRCCorrDzDtgl() const { return fNonRCCorrDzDtgl; }
  Double_t GetNonRCCorrDzDtglWithCal() const { return GetNonRCCorrDzDtgl() + GetParVal(kCalibNRCCorrDzDtgl); }
  //
  void SetCorrDVT(double v = 0) { fCorrDVT = 0; }
  Double_t GetCorrDVT() const { return fCorrDVT; }
  Double_t GetCorrDVTWithCal() const { return GetCorrDVT() + GetParVal(kCalibDVT); }
  //
  virtual Double_t GetCalibDOFVal(int id) const;
  virtual Double_t GetCalibDOFValWithCal(int id) const;
  //
  const Double_t* GetExtraErrRC() const { return fExtraErrRC; }
  void SetExtraErrRC(double y = 0.2, double z = 1.0)
  {
    fExtraErrRC[0] = y;
    fExtraErrRC[1] = z;
  }
  //
 protected:
  //
  // -------- dummies --------
  AliAlgDetTRD(const AliAlgDetTRD&);
  AliAlgDetTRD& operator=(const AliAlgDetTRD&);
  //
 protected:
  //
  Double_t fNonRCCorrDzDtgl; // correction in Z for non-crossing tracklets
  Double_t fCorrDVT;         // correction to Vdrift*t
  Double_t fExtraErrRC[2];   // extra errors for RC tracklets
  //
  static const char* fgkCalibDOFName[kNCalibParams];
  //
  ClassDef(AliAlgDetTRD, 1);
};
} // namespace align
} // namespace o2
#endif
