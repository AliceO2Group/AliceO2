// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDTrackPoint.h
/// \brief This is a flat data structure for transporting TRD track points via network between the components

/// \author Sergey Gorbunov, Ole Schmidt

#ifndef GPUTRDTRACKPOINT_H
#define GPUTRDTRACKPOINT_H

struct GPUTRDTrackPoint {
  float fX[3];
  short fVolumeId;
};

struct GPUTRDTrackPointData {
  unsigned int fCount; // number of space points
#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC)
  GPUTRDTrackPoint fPoints[1]; // array of space points
#else
  GPUTRDTrackPoint fPoints[0]; // array of space points
#endif
};

typedef struct GPUTRDTrackPointData GPUTRDTrackPointData;

#endif // GPUTRDTRACKPOINT_H
