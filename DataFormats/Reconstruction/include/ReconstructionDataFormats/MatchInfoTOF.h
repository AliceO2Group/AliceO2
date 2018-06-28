// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cluster.h
/// \brief Definition of the TOF cluster

#ifndef ALICEO2_MATCHINFOTOF_H
#define ALICEO2_MATCHINFOTOF_H

namespace o2
{
namespace dataformats
{

class MatchInfoTOF
{
 public:
  void setTOFClIndex(int index) { mTOFClIndex = index; }
  int getTOFClIndex() const { return mTOFClIndex; }

  void setTrackIndex(int index) { mTrackIndex = index; }
  int getTrackIndex() const { return mTrackIndex; }

  void setChi2(int chi2) { mChi2 = chi2; }
  int getChi2() const { return mChi2; }

 private:
  int mTOFClIndex; // index of the TOF cluster used for the matching
  int mTrackIndex; // index of the track used for the matching
  float mChi2;     // chi2 of the pair track-TOFcluster

  //  ClassDefNV(MatchInfoTOF, 1);
};
}
}
#endif
