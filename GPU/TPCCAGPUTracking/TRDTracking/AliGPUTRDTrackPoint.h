#ifndef ALIGPUTRDTRACKPOINT_H
#define ALIGPUTRDTRACKPOINT_H

// struct to hold the information on the space points
struct AliGPUTRDTrackPoint {
  float fX[3];
  short fVolumeId; 
};

struct AliGPUTRDTrackPointData {
  unsigned int fCount; // number of space points
#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC)
  AliGPUTRDTrackPoint fPoints[1]; // array of space points
#else
  AliGPUTRDTrackPoint fPoints[0]; // array of space points
#endif
};

typedef struct AliGPUTRDTrackPointData AliGPUTRDTrackPointData;


#endif
