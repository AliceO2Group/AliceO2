// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgMPRecord.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Millepede record in root format (can be converted to proper pede binary format.

#ifndef ALIALGMPRECORD_H
#define ALIALGMPRECORD_H

#include <TObject.h>

namespace o2
{
namespace align
{

class AliAlgTrack;

class AliAlgMPRecord : public TObject
{
 public:
  enum { kCosmicBit = BIT(14) };
  //
  AliAlgMPRecord();
  virtual ~AliAlgMPRecord();
  //
  int GetRun() const { return GetUniqueID(); }
  void SetRun(int r) { SetUniqueID(r); }
  uint32_t GetTimeStamp() const { return fTimeStamp; }
  void SetTimeStamp(uint32_t t) { fTimeStamp = t; }
  uint32_t GetTrackID() const { return fTrackID; }
  void SetTrackID(uint32_t t) { fTrackID = t; }
  bool IsCosmic() const { return TestBit(kCosmicBit); }
  void SetCosmic(bool v = true) { SetBit(kCosmicBit, v); }
  //
  int GetNVarGlo() const { return fNVarGlo; }
  void SetNVarGlo(int n) { fNVarGlo = n; }
  //
  int GetNResid() const { return fNResid; }
  int GetNVarLoc() const { return fNVarLoc; }
  //
  int GetNDLoc(int id) const { return fNDLoc[id]; }
  int GetNDGlo(int id) const { return fNDGlo[id]; }
  int GetVolID(int id) const { return fVolID ? fVolID[id] - 1 : -1; }
  float GetResid(int id) const { return fResid[id]; }
  float GetResErr(int id) const { return fResErr[id]; }
  //
  float GetChi2Ini() const { return fChi2Ini; }
  float GetQ2Pt() const { return fQ2Pt; }
  float GetTgl() const { return fTgl; }
  int GetNDLocTot() const { return fNDLocTot; }
  int GetNDGloTot() const { return fNDGloTot; }
  const float* GetArrGlo() const { return fDGlo; }
  const float* GetArrLoc() const { return fDLoc; }
  const int16_t* GetArrLabLoc() const { return fIDLoc; }
  const int* GetArrLabGlo() const { return fIDGlo; }
  //
  bool FillTrack(const AliAlgTrack* trc, const int* id2Lab = 0);
  void DummyRecord(float res, float err, float dGlo, int labGlo);
  //
  void Resize(int nresid, int nloc, int nglo);
  //
  virtual void Clear(const Option_t* opt = "");
  virtual void Print(const Option_t* opt = "") const;
  //
 protected:
  //
  // ------- dummies --------
  AliAlgMPRecord(const AliAlgMPRecord&);
  AliAlgMPRecord& operator=(const AliAlgMPRecord&);
  //
 protected:
  //
  uint32_t fTrackID;   // track in the event
  uint32_t fTimeStamp; // event time stamp
  int fNResid;         // number of residuals for the track (=2 npoints)
  int fNVarLoc;        // number of local variables for the track
  int fNVarGlo;        // number of global variables defined
  int fNDLocTot;       // total number of non-zero local derivatives
  int fNDGloTot;       // total number of non-zero global derivatives
  int fNMeas;          // number of measured points
  float fChi2Ini;      // chi2 of initial kalman fit
  float fQ2Pt;         // q/pt at ref point
  float fTgl;          // dip angle at ref point
  //
  int16_t* fNDLoc; //[fNResid] number of non-0 local derivatives per residual
  int* fNDGlo;     //[fNResid] number of non-0 global derivatives per residual
  int* fVolID;     //[fNResid] volume id + 1 (0 - not a volume)
  float* fResid;   //[fNResid] residuals
  float* fResErr;  //[fNResid] error associated to residual
  //
  int16_t* fIDLoc; //[fNDLocTot] ID of local variables for non-0 local derivatives
  int* fIDGlo;     //[fNDGloTot] ID of global variables for non-0 global derivatives
  float* fDLoc;    //[fNDLocTot] non-0 local derivatives
  float* fDGlo;    //[fNDGloTot] non-0 global derivatives
  //
  // aux info
  int fNResidBook;   //! number of slots booked for residuals
  int fNDLocTotBook; //! number of slots booked for local derivatives
  int fNDGloTotBook; //! number of slots booked for global derivatives
  //
  ClassDef(AliAlgMPRecord, 4);
};
} // namespace align
} // namespace o2
#endif
