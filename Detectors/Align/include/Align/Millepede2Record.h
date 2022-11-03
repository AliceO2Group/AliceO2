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

#include <Rtypes.h>
#include "ReconstructionDataFormats/GlobalTrackID.h"

namespace o2
{
namespace align
{

class AlignmentTrack;

class Millepede2Record
{
 public:
  enum { CosmicBit = 0x1 };
  //
  Millepede2Record() = default;
  ~Millepede2Record();
  //
  int getRun() const { return mRunNumber; }
  void setRun(int r) { mRunNumber = r; }
  uint32_t getFirstTFOrbit() const { return mFirstTFOrbit; }
  void setFirstTFOrbit(uint32_t v) { mFirstTFOrbit = v; }
  o2::dataformats::GlobalTrackID getTrackID() const { return mTrackID; }
  void setTrackID(o2::dataformats::GlobalTrackID t) { mTrackID = t; }
  bool isCosmic() const { return testBit(CosmicBit); }
  void setCosmic(bool v = true) { setBit(CosmicBit, v); }
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
  bool fillTrack(AlignmentTrack& trc, const std::vector<int>& id2Lab);
  void dummyRecord(float res, float err, float dGlo, int labGlo);
  //
  void resize(int nresid, int nloc, int nglo);
  //
  void clear();
  void print() const;
  //
 protected:
  //
  uint16_t mBits = 0;
  int mRunNumber = 0;
  uint32_t mFirstTFOrbit = 0;                // event time stamp
  o2::dataformats::GlobalTrackID mTrackID{}; // track in the event
  int mNResid = 0;                           // number of residuals for the track (=2 npoints)
  int mNVarLoc = 0;                          // number of local variables for the track
  int mNVarGlo = 0;                          // number of global variables defined
  int mNDLocTot = 0;                         // total number of non-zero local derivatives
  int mNDGloTot = 0;                         // total number of non-zero global derivatives
  int mNMeas = 0;                            // number of measured points
  float mChi2Ini = 0;                        // chi2 of initial kalman fit
  float mQ2Pt = 0;                           // q/pt at ref point
  float mTgl = 0;                            // dip angle at ref point
  //
  int16_t* mNDLoc = nullptr; //[mNResid] number of non-0 local derivatives per residual
  int* mNDGlo = nullptr;     //[mNResid] number of non-0 global derivatives per residual
  int* mVolID = nullptr;     //[mNResid] volume id + 1 (0 - not a volume)
  float* mResid = nullptr;   //[mNResid] residuals
  float* mResErr = nullptr;  //[mNResid] error associated to residual
  //
  int16_t* mIDLoc = nullptr; //[mNDLocTot] ID of local variables for non-0 local derivatives
  int* mIDGlo = nullptr;     //[mNDGloTot] ID of global variables for non-0 global derivatives
  float* mDLoc = nullptr;    //[mNDLocTot] non-0 local derivatives
  float* mDGlo = nullptr;    //[mNDGloTot] non-0 global derivatives
  //
  // aux info
  int mNResidBook = 0;   //! number of slots booked for residuals
  int mNDLocTotBook = 0; //! number of slots booked for local derivatives
  int mNDGloTotBook = 0; //! number of slots booked for global derivatives
  //
  void setBit(uint16_t b, bool v)
  {
    if (v) {
      mBits |= b;
    } else {
      mBits &= ~(b & 0xffff);
    }
  }
  bool testBit(uint16_t b) const
  {
    return mBits & b;
  }

  ClassDefNV(Millepede2Record, 1);
};
} // namespace align
} // namespace o2
#endif
