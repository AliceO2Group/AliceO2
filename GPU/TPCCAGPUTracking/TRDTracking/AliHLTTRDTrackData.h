// $Id$
//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTRDTRACKDATA_H
#define ALIHLTTRDTRACKDATA_H

#include "AliHLTDataTypes.h"
#include "AliHLTStdIncludes.h"

/**
 * @struct AliHLTTRDTrackData 
 * This is a flat data structure (w/o virtual methods, i.e w/o pointer to virtual table) for transporting TRD tracks via network between the components.
 */

struct AliHLTTRDTrackDataRecord
{
  AliHLTFloat32_t fAlpha;  // azimuthal angle of reference frame
  AliHLTFloat32_t fX;      // x: radial distance
  AliHLTFloat32_t fY;      // local Y-coordinate of a track (cm)
  AliHLTFloat32_t fZ;      // local Z-coordinate of a track (cm)
  AliHLTFloat32_t fSinPsi; // local sine of the track momentum azimuthal angle
  AliHLTFloat32_t fTgl;    // tangent of the track momentum dip angle
  AliHLTFloat32_t fq1Pt;   // 1/pt (1/(GeV/c))
  AliHLTFloat32_t fC[15];  // covariance matrix
  AliHLTInt32_t   fTPCTrackID;// id of corresponding TPC track  
  AliHLTInt32_t   fAttachedTracklets[6];  // IDs for attached tracklets sorted by layer
};

typedef struct AliHLTTRDTrackDataRecord AliHLTTRDTrackDataRecord;

struct AliHLTTRDTrackData {
  AliHLTUInt32_t fCount; // number of tracklets
#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC)
  AliHLTTRDTrackDataRecord fTracks[1]; // array of tracklets
#else
  AliHLTTRDTrackDataRecord fTracks[0]; // array of tracklets
#endif
  static size_t GetSize( AliHLTUInt32_t nTracks ) { return sizeof(AliHLTTRDTrackData) + nTracks*sizeof(AliHLTTRDTrackDataRecord); }
  size_t GetSize() const { return GetSize( fCount ); }
};

typedef struct AliHLTTRDTrackData AliHLTTRDTrackData;

#endif
