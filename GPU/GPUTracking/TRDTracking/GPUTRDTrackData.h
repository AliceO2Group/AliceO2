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

/// \file GPUTRDTrackData.h
/// \brief This is a flat data structure for transporting TRD tracks via network between the components

/// \author Sergey Gorbunov, Ole Schmidt

#ifndef GPUTRDTRACKDATA_H
#define GPUTRDTRACKDATA_H

struct GPUTRDTrackDataRecord {
  float mAlpha;              // azimuthal angle of reference frame
  float fX;                  // x: radial distance
  float fY;                  // local Y-coordinate of a track (cm)
  float fZ;                  // local Z-coordinate of a track (cm)
  float mSinPhi;             // local sine of the track momentum azimuthal angle
  float fTgl;                // tangent of the track momentum dip angle
  float fq1Pt;               // 1/pt (1/(GeV/c))
  float fC[15];              // covariance matrix
  int fTPCTrackID;           // id of corresponding TPC track
  int fAttachedTracklets[6]; // IDs for attached tracklets sorted by layer
  unsigned char mIsPadrowCrossing; // bits 0 to 5 indicate whether a padrow was crossed

  int GetNTracklets() const
  {
    int n = 0;
    for (int i = 0; i < 6; i++) {
      if (fAttachedTracklets[i] >= 0) {
        n++;
      }
    }
    return n;
  }
};

typedef struct GPUTRDTrackDataRecord GPUTRDTrackDataRecord;

struct GPUTRDTrackData {
  unsigned int fCount; // number of tracklets
#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC)
  GPUTRDTrackDataRecord fTracks[1]; // array of tracklets
#else
  GPUTRDTrackDataRecord fTracks[0]; // array of tracklets
#endif
  static size_t GetSize(unsigned int nTracks)
  {
    return sizeof(GPUTRDTrackData) + nTracks * sizeof(GPUTRDTrackDataRecord);
  }
  size_t GetSize() const { return GetSize(fCount); }
};

typedef struct GPUTRDTrackData GPUTRDTrackData;

#endif // GPUTRDTRACKDATA_H
