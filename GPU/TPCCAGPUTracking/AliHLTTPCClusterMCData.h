#ifndef _ALIHLTTPCCLUSTERMCDATA_H_
#define _ALIHLTTPCCLUSTERMCDATA_H_

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#include "AliHLTDataTypes.h"


/**
 * @struct AliHLTTPCClusterMCWeight
 * This in a struct for MC weights
 * @ingroup alihlt_tpc
 */
struct AliHLTTPCClusterMCWeight
{
  //* constructor **/
  AliHLTTPCClusterMCWeight(): fMCID(-1), fWeight(0)
  {}

  AliHLTInt32_t  fMCID;     // MC track ID
  AliHLTFloat32_t fWeight; // weight of the track ID
};  

typedef struct AliHLTTPCClusterMCWeight AliHLTTPCClusterMCWeight;

/**
 * @struct AliHLTTPCClusterMCLabel
 * This in a struct for MC labels
 * @ingroup alihlt_tpc
 */
struct AliHLTTPCClusterMCLabel
{
  AliHLTTPCClusterMCWeight fClusterID[3]; // three most relevant MC labels
};

typedef struct AliHLTTPCClusterMCLabel AliHLTTPCClusterMCLabel;


/**
 * @struct AliHLTTPCClusterMCData
 * This in a container for MC labels
 * @ingroup alihlt_tpc
 */
struct AliHLTTPCClusterMCData 
{
  AliHLTUInt32_t fCount;
#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC) || defined (__clang__)
  AliHLTTPCClusterMCLabel fLabels[1];
#else
  AliHLTTPCClusterMCLabel fLabels[];
#endif
};

typedef struct AliHLTTPCClusterMCData AliHLTTPCClusterMCData;

#endif
