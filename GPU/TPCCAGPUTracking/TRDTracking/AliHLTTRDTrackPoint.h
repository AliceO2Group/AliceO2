#ifndef ALIHLTTRDTRACKPOINT_H
#define ALIHLTTRDTRACKPOINT_H

#include "AliHLTDataTypes.h"

// struct to hold the information on the space points
struct AliHLTTRDTrackPoint {
  float fX[3];
  short fVolumeId; 
};

struct AliHLTTRDTrackPointData {
  AliHLTUInt32_t fCount; // number of space points
#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC)
  AliHLTTRDTrackPoint fPoints[1]; // array of space points
#else
  AliHLTTRDTrackPoint fPoints[0]; // array of space points
#endif
};

typedef struct AliHLTTRDTrackPointData AliHLTTRDTrackPointData;


#endif
