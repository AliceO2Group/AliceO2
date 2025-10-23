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
///
/// \file IOUtils.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_EVENTLOADER_H_
#define TRACKINGITSU_INCLUDE_EVENTLOADER_H_

#include <vector>

#include "ITSMFTBase/SegmentationAlpide.h"
#include "ReconstructionDataFormats/BaseCluster.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/ROFRecord.h" // TODO this is just included since the alignment code include it now

namespace o2::its::ioutils
{

constexpr float DefClusErrorRow = o2::itsmft::SegmentationAlpide::PitchRow * 0.5;
constexpr float DefClusErrorCol = o2::itsmft::SegmentationAlpide::PitchCol * 0.5;
constexpr float DefClusError2Row = DefClusErrorRow * DefClusErrorRow;
constexpr float DefClusError2Col = DefClusErrorCol * DefClusErrorCol;

void convertCompactClusters(gsl::span<const itsmft::CompClusterExt> clusters,
                            gsl::span<const unsigned char>::iterator& pattIt,
                            std::vector<o2::BaseCluster<float>>& output,
                            const itsmft::TopologyDictionary* dict);

template <class iterator, typename T>
o2::math_utils::Point3D<T> extractClusterData(const itsmft::CompClusterExt& c, iterator& iter, const itsmft::TopologyDictionary* dict, T& sig2y, T& sig2z)
{
  auto pattID = c.getPatternID();
  sig2y = ioutils::DefClusError2Row;
  sig2z = ioutils::DefClusError2Col; // Dummy COG errors (about half pixel size)
  if (pattID != itsmft::CompCluster::InvalidPatternID) {
    sig2y = dict->getErr2X(pattID);
    sig2z = dict->getErr2Z(pattID);
    if (!dict->isGroup(pattID)) {
      return dict->getClusterCoordinates<T>(c);
    } else {
      o2::itsmft::ClusterPattern patt(iter);
      return dict->getClusterCoordinates<T>(c, patt);
    }
  } else {
    o2::itsmft::ClusterPattern patt(iter);
    return dict->getClusterCoordinates<T>(c, patt, false);
  }
}

// same method returning coordinates as an array (suitable for the TGeoMatrix)
template <class iterator, typename T>
std::array<T, 3> extractClusterDataA(const itsmft::CompClusterExt& c, iterator& iter, const itsmft::TopologyDictionary* dict, T& sig2y, T& sig2z)
{
  auto pattID = c.getPatternID();
  sig2y = ioutils::DefClusError2Row;
  sig2z = ioutils::DefClusError2Col; // Dummy COG errors (about half pixel size)
  if (pattID != itsmft::CompCluster::InvalidPatternID) {
    sig2y = dict->getErr2X(pattID);
    sig2z = dict->getErr2Z(pattID);
    if (!dict->isGroup(pattID)) {
      return dict->getClusterCoordinatesA<T>(c);
    } else {
      o2::itsmft::ClusterPattern patt(iter);
      return dict->getClusterCoordinatesA<T>(c, patt);
    }
  } else {
    o2::itsmft::ClusterPattern patt(iter);
    return dict->getClusterCoordinatesA<T>(c, patt, false);
  }
}

} // namespace o2::its::ioutils

#endif /* TRACKINGITSU_INCLUDE_EVENTLOADER_H_ */
