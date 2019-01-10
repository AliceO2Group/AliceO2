// $Id$
//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIGPUTRDTRACKDATA_H
#define ALIGPUTRDTRACKDATA_H

/**
 * @struct AliGPUTRDTrackData
 * This is a flat data structure (w/o virtual methods, i.e w/o pointer to virtual table) for transporting TRD tracks via network between the components.
 */

struct AliGPUTRDTrackDataRecord
{
  float fAlpha;  // azimuthal angle of reference frame
  float fX;      // x: radial distance
  float fY;      // local Y-coordinate of a track (cm)
  float fZ;      // local Z-coordinate of a track (cm)
  float fSinPhi; // local sine of the track momentum azimuthal angle
  float fTgl;    // tangent of the track momentum dip angle
  float fq1Pt;   // 1/pt (1/(GeV/c))
  float fC[15];  // covariance matrix
  int   fTPCTrackID;// id of corresponding TPC track
  int   fAttachedTracklets[6];  // IDs for attached tracklets sorted by layer

  int GetNTracklets() const {
    int n=0;
    for( int i=0; i<6; i++ ) if( fAttachedTracklets[i]>=0 ) n++;
    return n;
  }
};

typedef struct AliGPUTRDTrackDataRecord AliGPUTRDTrackDataRecord;

struct AliGPUTRDTrackData {
  unsigned int fCount; // number of tracklets
#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC)
  AliGPUTRDTrackDataRecord fTracks[1]; // array of tracklets
#else
  AliGPUTRDTrackDataRecord fTracks[0]; // array of tracklets
#endif
  static size_t GetSize( unsigned int nTracks ) { return sizeof(AliGPUTRDTrackData) + nTracks*sizeof(AliGPUTRDTrackDataRecord); }
  size_t GetSize() const { return GetSize( fCount ); }
};

typedef struct AliGPUTRDTrackData AliGPUTRDTrackData;

#endif
