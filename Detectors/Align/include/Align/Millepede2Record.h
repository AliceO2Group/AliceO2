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

/// @file   Millepede2Record.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Millepede record in root format (can be converted to proper pede binary format.

#ifndef MILLEPEDE2RECORD_H
#define MILLEPEDE2RECORD_H

#include <TObject.h>

namespace o2
{
namespace align
{

class AlignmentTrack;

class Millepede2Record : public TObject
{
 public:
  enum { kCosmicBit = BIT(14) };
  //
  Millepede2Record();
  ~Millepede2Record() final;
  //
  int getRun() const { return GetUniqueID(); }
  void setRun(int r) { SetUniqueID(r); }
  uint32_t getTimeStamp() const { return mTimeStamp; }
  void setTimeStamp(uint32_t t) { mTimeStamp = t; }
  uint32_t getTrackID() const { return mTrackID; }
  void setTrackID(uint32_t t) { mTrackID = t; }
  bool isCosmic() const { return TestBit(kCosmicBit); }
  void setCosmic(bool v = true) { SetBit(kCosmicBit, v); }
  //
  int getNVarGlo() const { return mNVarGlo; }
  void setNVarGlo(int n) { mNVarGlo = n; }
  //
  int getNResid() const { return mNResid; }
  int getNVarLoc() const { return mNVarLoc; }
  //
  int getNDLoc(int id) const { return mNDLoc[id]; }
  int getNDGlo(int id) const { return mNDGlo[id]; }
  int getVolID(int id) const { return mVolID ? mVolID[id] - 1 : -1; }
  float getResid(int id) const { return mResid[id]; }
  float getResErr(int id) const { return mResErr[id]; }
  //
  float getChi2Ini() const { return mChi2Ini; }
  float getQ2Pt() const { return mQ2Pt; }
  float getTgl() const { return mTgl; }
  int getNDLocTot() const { return mNDLocTot; }
  int getNDGloTot() const { return mNDGloTot; }
  const float* getArrGlo() const { return mDGlo; }
  const float* getArrLoc() const { return mDLoc; }
  const int16_t* getArrLabLoc() const { return mIDLoc; }
  const int* getArrLabGlo() const { return mIDGlo; }
  //
  bool fillTrack(AlignmentTrack* trc, const int* id2Lab = nullptr);
  void dummyRecord(float res, float err, float dGlo, int labGlo);
  //
  void resize(int nresid, int nloc, int nglo);
  //
  void Clear(const Option_t* opt = "") final;
  void Print(const Option_t* opt = "") const final;
  //
 protected:
  //
  // ------- dummies --------
  Millepede2Record(const Millepede2Record&);
  Millepede2Record& operator=(const Millepede2Record&);
  //
 protected:
  //
  uint32_t mTrackID;   // track in the event
  uint32_t mTimeStamp; // event time stamp
  int mNResid;         // number of residuals for the track (=2 npoints)
  int mNVarLoc;        // number of local variables for the track
  int mNVarGlo;        // number of global variables defined
  int mNDLocTot;       // total number of non-zero local derivatives
  int mNDGloTot;       // total number of non-zero global derivatives
  int mNMeas;          // number of measured points
  float mChi2Ini;      // chi2 of initial kalman fit
  float mQ2Pt;         // q/pt at ref point
  float mTgl;          // dip angle at ref point
  //
  int16_t* mNDLoc; //[mNResid] number of non-0 local derivatives per residual
  int* mNDGlo;     //[mNResid] number of non-0 global derivatives per residual
  int* mVolID;     //[mNResid] volume id + 1 (0 - not a volume)
  float* mResid;   //[mNResid] residuals
  float* mResErr;  //[mNResid] error associated to residual
  //
  int16_t* mIDLoc; //[mNDLocTot] ID of local variables for non-0 local derivatives
  int* mIDGlo;     //[mNDGloTot] ID of global variables for non-0 global derivatives
  float* mDLoc;    //[mNDLocTot] non-0 local derivatives
  float* mDGlo;    //[mNDGloTot] non-0 global derivatives
  //
  // aux info
  int mNResidBook;   //! number of slots booked for residuals
  int mNDLocTotBook; //! number of slots booked for local derivatives
  int mNDGloTotBook; //! number of slots booked for global derivatives
  //
  ClassDef(Millepede2Record, 4);
};
} // namespace align
} // namespace o2
#endif
