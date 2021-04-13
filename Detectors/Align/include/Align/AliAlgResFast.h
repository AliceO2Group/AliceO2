// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgResFast.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Container for control fast residuals evaluated via derivatives

#ifndef ALIALGRESFAST_H
#define ALIALGRESFAST_H

#include <TObject.h>

namespace o2
{
namespace align
{

class AliAlgResFast : public TObject
{
 public:
  enum { kCosmicBit = BIT(14),
         kVertexBit = BIT(15) };
  //
  AliAlgResFast();
  virtual ~AliAlgResFast();
  //
  void SetNPoints(int n)
  {
    fNPoints = n;
    Resize(n);
  }
  void SetNMatSol(int n) { fNMatSol = n; }
  //
  void SetChi2(float v) { fChi2 = v; }
  float GetChi2() const { return fChi2; }
  //
  void SetChi2Ini(float v) { fChi2Ini = v; }
  float GetChi2Ini() const { return fChi2Ini; }
  //
  bool IsCosmic() const { return TestBit(kCosmicBit); }
  bool HasVertex() const { return TestBit(kVertexBit); }
  void SetCosmic(bool v = true) { SetBit(kCosmicBit, v); }
  void SetHasVertex(bool v = true) { SetBit(kVertexBit, v); }
  //
  int GetNPoints() const { return fNPoints; }
  int GetNMatSol() const { return fNMatSol; }
  int GetNBook() const { return fNBook; }
  float GetD0(int i) const { return fD0[i]; }
  float GetD1(int i) const { return fD1[i]; }
  float GetSig0(int i) const { return fSig0[i]; }
  float GetSig1(int i) const { return fSig1[i]; }
  int GetVolID(int i) const { return fVolID[i]; }
  int GetLabel(int i) const { return fLabel[i]; }
  //
  float* GetTrCor() const { return (float*)fTrCorr; }
  float* GetD0() const { return (float*)fD0; }
  float* GetD1() const { return (float*)fD1; }
  float* GetSig0() const { return (float*)fSig0; }
  float* GetSig1() const { return (float*)fSig1; }
  int* GetVolID() const { return (int*)fVolID; }
  int* GetLaber() const { return (int*)fLabel; }
  float* GetSolMat() const { return (float*)fSolMat; }
  float* GetMatErr() const { return (float*)fMatErr; }
  //
  void SetResSigMeas(int ip, int ord, float res, float sig);
  void SetMatCorr(int id, float res, float sig);
  void SetLabel(int ip, int lab, int vol);
  //
  void Resize(int n);
  virtual void Clear(const Option_t* opt = "");
  virtual void Print(const Option_t* opt = "") const;
  //
 protected:
  //
  // -------- dummies --------
  AliAlgResFast(const AliAlgResFast&);
  AliAlgResFast& operator=(const AliAlgResFast&);
  //
 protected:
  //
  int fNPoints;   // n meas points
  int fNMatSol;   // n local params - ExtTrPar corrections
  int fNBook;     //! booked lenfth
  float fChi2;    // chi2
  float fChi2Ini; // chi2 before local fit
  //
  float fTrCorr[5]; //  correction to ExternalTrackParam
  float* fD0;       //[fNPoints] 1st residual (track - meas)
  float* fD1;       //[fNPoints] 2ns residual (track - meas)
  float* fSig0;     //[fNPoints] ort. error 0
  float* fSig1;     //[fNPoints] ort. errir 1
  int* fVolID;      //[fNPoints] volume id (0 for vertex constraint)
  int* fLabel;      //[fNPoints] label of the volume
  //
  float* fSolMat; //[fNMatSol] // material corrections
  float* fMatErr; //[fNMatSol] // material corrections errors
  //
  ClassDef(AliAlgResFast, 1);
};
} // namespace align
} // namespace o2
#endif
