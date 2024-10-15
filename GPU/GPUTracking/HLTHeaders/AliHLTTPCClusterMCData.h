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

/// \file AliHLTTPCClusterMCData.h
/// \author ALICE HLT Project

#ifndef _ALIHLTTPCCLUSTERMCDATA_H_
#define _ALIHLTTPCCLUSTERMCDATA_H_

/**
 * @struct AliHLTTPCClusterMCWeight
 * This in a struct for MC weights
 * @ingroup alihlt_tpc
 */
struct AliHLTTPCClusterMCWeight {
  //* constructor **/
  AliHLTTPCClusterMCWeight() : fMCID(-1), fWeight(0) {}

  int32_t fMCID; // MC track ID
  float fWeight; // weight of the track ID
};

typedef struct AliHLTTPCClusterMCWeight AliHLTTPCClusterMCWeight;

/**
 * @struct AliHLTTPCClusterMCLabel
 * This in a struct for MC labels
 * @ingroup alihlt_tpc
 */
struct AliHLTTPCClusterMCLabel {
  AliHLTTPCClusterMCWeight fClusterID[3]; // three most relevant MC labels
};

typedef struct AliHLTTPCClusterMCLabel AliHLTTPCClusterMCLabel;

/**
 * @struct AliHLTTPCClusterMCData
 * This in a container for MC labels
 * @ingroup alihlt_tpc
 */
struct AliHLTTPCClusterMCData {
  int32_t fCount;
#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC)
  AliHLTTPCClusterMCLabel fLabels[1];
#else
  AliHLTTPCClusterMCLabel fLabels[0];
#endif
};

typedef struct AliHLTTPCClusterMCData AliHLTTPCClusterMCData;

#endif
