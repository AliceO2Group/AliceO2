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

/// \file ClusterID.h
/// \brief Composition/decomposition of the ITS/MFT cluster IDs referring to per-layer cluster arrays
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_ITSMFT_CLUSTERID_H
#define ALICEO2_ITSMFT_CLUSTERID_H

namespace o2::itsmft
{

// max number of layers for which the ITS/MFT clusters, ROF records and patterns can be provided separately
constexpr int MaxITSClusLayers = 7;
constexpr int MaxMFTClusLayers = 10;
constexpr int MaxClusLayers = MaxITSClusLayers > MaxMFTClusLayers ? MaxITSClusLayers : MaxMFTClusLayers;

///< With the per-layer (staggered readout) ITS/MFT clusters input the clusters are referred to by the
///< composed ID (layer << ClusLayerShift) + index_in_layer. With a single (monolithic) clusters input
///< all clusters sit in the layer slot 0, hence the composed ID coincides with the flat cluster index
///< and the same decoding works for both cases.
///< Note: the bit 31 is excluded from the layer field, since the negative values of the composed ID
///< are reserved for the "no cluster" flags.
constexpr int ClusLayerShift = 27;
constexpr int ClusIndexMask = (0x1 << ClusLayerShift) - 1;
static_assert((1 << (31 - ClusLayerShift)) >= MaxClusLayers, "ClusLayerShift leaves no room for the layer ID");

constexpr int composeClusID(int lr, int idx) { return (lr << ClusLayerShift) + idx; }
constexpr int clusID2Layer(int id) { return id >> ClusLayerShift; }
constexpr int clusID2Index(int id) { return id & ClusIndexMask; }

} // namespace o2::itsmft

#endif
