// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgRes.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Container for control residuals

#ifndef ALIALGRES_H
#define ALIALGRES_H

#include <TObject.h>
#include <TMath.h>

namespace o2
{
namespace align
{

class AliAlgTrack;

class AliAlgRes : public TObject
{
 public:
  enum { kCosmicBit = BIT(14),
         kVertexBit = BIT(15),
         kKalmanDoneBit = BIT(16) };
  //
  AliAlgRes();
  virtual ~AliAlgRes();
  //
  void SetRun(int r) { fRun = r; }
  void SetBz(float v) { fBz = v; }
  void SetTimeStamp(uint32_t v) { fTimeStamp = v; }
  void SetTrackID(uint32_t v) { fTrackID = v; }
  void SetNPoints(int n)
  {
    fNPoints = n;
    Resize(n);
  }
  //
  bool IsCosmic() const { return TestBit(kCosmicBit); }
  bool HasVertex() const { return TestBit(kVertexBit); }
  void SetCosmic(bool v = true) { SetBit(kCosmicBit, v); }
  void SetHasVertex(bool v = true) { SetBit(kVertexBit, v); }
  //
  bool GetKalmanDone() const { return TestBit(kKalmanDoneBit); }
  void SetKalmanDone(bool v = true) { SetBit(kKalmanDoneBit, v); }
  //
  int GetRun() const { return fRun; }
  float GetBz() const { return fBz; }
  uint32_t GetTimeStamp() const { return fTimeStamp; }
  uint32_t GetTrackID() const { return fTrackID; }
  int GetNPoints() const { return fNPoints; }
  int GetNBook() const { return fNBook; }
  float GetChi2() const { return fChi2; }
  float GetChi2Ini() const { return fChi2Ini; }
  float GetChi2K() const { return fChi2K; }
  float GetQ2Pt() const { return fQ2Pt; }
  float GetX(int i) const { return fX[i]; }
  float GetY(int i) const { return fY[i]; }
  float GetZ(int i) const { return fZ[i]; }
  float GetSnp(int i) const { return fSnp[i]; }
  float GetTgl(int i) const { return fTgl[i]; }
  float GetAlpha(int i) const { return fAlpha[i]; }
  float GetDY(int i) const { return fDY[i]; }
  float GetDZ(int i) const { return fDZ[i]; }
  float GetDYK(int i) const { return fDYK[i]; }
  float GetDZK(int i) const { return fDZK[i]; }
  //
  float GetSigY2K(int i) const { return fSigY2K[i]; }
  float GetSigYZK(int i) const { return fSigYZK[i]; }
  float GetSigZ2K(int i) const { return fSigZ2K[i]; }
  float GetSigmaYK(int i) const { return TMath::Sqrt(fSigY2K[i]); }
  float GetSigmaZK(int i) const { return TMath::Sqrt(fSigZ2K[i]); }
  //
  float GetSigY2(int i) const { return fSigY2[i]; }
  float GetSigYZ(int i) const { return fSigYZ[i]; }
  float GetSigZ2(int i) const { return fSigZ2[i]; }
  float GetSigmaY(int i) const { return TMath::Sqrt(fSigY2[i]); }
  float GetSigmaZ(int i) const { return TMath::Sqrt(fSigZ2[i]); }
  //
  float GetSigY2Tot(int i) const { return fSigY2K[i] + fSigY2[i]; }
  float GetSigYZTot(int i) const { return fSigYZK[i] + fSigYZ[i]; }
  float GetSigZ2Tot(int i) const { return fSigZ2K[i] + fSigZ2[i]; }
  float GetSigmaYTot(int i) const { return TMath::Sqrt(GetSigY2Tot(i)); }
  float GetSigmaZTot(int i) const { return TMath::Sqrt(GetSigZ2Tot(i)); }
  //
  int GetVolID(int i) const { return fVolID[i]; }
  //
  float GetXLab(int i) const;
  float GetYLab(int i) const;
  float GetZLab(int i) const;
  //
  bool FillTrack(AliAlgTrack* trc, bool doKalman = true);
  void Resize(int n);
  virtual void Clear(const Option_t* opt = "");
  virtual void Print(const Option_t* opt = "re") const;
  //
 protected:
  //
  // -------- dummies --------
  AliAlgRes(const AliAlgRes&);
  AliAlgRes& operator=(const AliAlgRes&);
  //
 protected:
  //
  int fRun;            // run
  float fBz;           // field
  uint32_t fTimeStamp; // event time
  uint32_t fTrackID;   // track ID
  int fNPoints;        // n meas points
  int fNBook;          //! booked lenfth
  float fChi2;         //  chi2 after solution
  float fChi2Ini;      //  chi2 before solution
  float fChi2K;        //  chi2 from kalman
  float fQ2Pt;         //  Q/Pt at reference point
  float* fX;           //[fNPoints] tracking X of cluster
  float* fY;           //[fNPoints] tracking Y of cluster
  float* fZ;           //[fNPoints] tracking Z of cluster
  float* fSnp;         //[fNPoints] track Snp
  float* fTgl;         //[fNPoints] track Tgl
  float* fAlpha;       //[fNPoints] track alpha
  float* fDY;          //[fNPoints] Y residual (track - meas)
  float* fDZ;          //[fNPoints] Z residual (track - meas)
  float* fDYK;         //[fNPoints] Y residual (track - meas) Kalman
  float* fDZK;         //[fNPoints] Z residual (track - meas) Kalman
  float* fSigY2;       //[fNPoints] Y err^2
  float* fSigYZ;       //[fNPoints] YZ err
  float* fSigZ2;       //[fNPoints] Z err^2
  float* fSigY2K;      //[fNPoints] Y err^2 of Kalman track smoothing
  float* fSigYZK;      //[fNPoints] YZ err  of Kalman track smoothing
  float* fSigZ2K;      //[fNPoints] Z err^2 of Kalman track smoothing
  int* fVolID;         //[fNPoints] volume id (0 for vertex constraint)
  int* fLabel;         //[fNPoints] label of the volume
  //
  ClassDef(AliAlgRes, 2);
};
} // namespace align
} // namespace o2
#endif
