#ifndef _ALIHLTTPCCLUSTERMCDATA_H_
#define _ALIHLTTPCCLUSTERMCDATA_H_

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

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

  int  fMCID;     // MC track ID
  float fWeight; // weight of the track ID
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
  int fCount;
#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC)
  AliHLTTPCClusterMCLabel fLabels[1];
#else
  AliHLTTPCClusterMCLabel fLabels[0];
#endif
};

typedef struct AliHLTTPCClusterMCData AliHLTTPCClusterMCData;

#endif
