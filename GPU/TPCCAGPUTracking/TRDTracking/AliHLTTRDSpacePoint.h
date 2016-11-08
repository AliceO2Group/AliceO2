#ifndef ALIHLTTRDSPACEPOINT_H
#define ALIHLTTRDSPACEPOINT_H

#include "AliHLTDataTypes.h"

// struct to hold the information on the space points
struct AliHLTTRDSpacePoint {
  float fX[3];
};

struct AliHLTTRDSpacePointData {
  AliHLTUInt32_t fCount; // number of space points
#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC)
  AliHLTTRDSpacePoint fPoints[1]; // array of space points
#else
  AliHLTTRDSpacePoint fPoints[0]; // array of space points
#endif
};

typedef struct AliHLTTRDSpacePointData AliHLTTRDSpacePointData;


#endif
