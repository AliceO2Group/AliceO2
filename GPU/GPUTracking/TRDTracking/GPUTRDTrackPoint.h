#ifndef GPUTRDTRACKPOINT_H
#define GPUTRDTRACKPOINT_H

// struct to hold the information on the space points
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


#endif
