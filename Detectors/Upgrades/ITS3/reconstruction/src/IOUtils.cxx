// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITS3Reconstruction/IOUtils.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/BoundedAllocator.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITS3Reconstruction/TopologyDictionary.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITStracking/TrackingConfigParam.h"

namespace o2::its3::ioutils
{

/// convert compact clusters to 3D spacepoints
void convertCompactClusters(gsl::span<const itsmft::CompClusterExt> clusters,
                            gsl::span<const unsigned char>::iterator& pattIt,
                            std::vector<o2::BaseCluster<float>>& output,
                            const its3::TopologyDictionary* dict)
{
  auto geom = o2::its::GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  bool applyMisalignment = false;
  const auto& conf = o2::its::TrackerParamConfig::Instance();
  for (int il = 0; il < geom->getNumberOfLayers(); ++il) {
    if (conf.sysErrY2[il] > 0.f || conf.sysErrZ2[il] > 0.f) {
      applyMisalignment = true;
      break;
    }
  }

  for (auto& c : clusters) {
    float sigmaY2 = NAN, sigmaZ2 = NAN;
    auto locXYZ = extractClusterData(c, pattIt, dict, sigmaY2, sigmaZ2);
    const auto detID = c.getSensorID();
    // NOTE: this is not consistent with the TRK definition below!
    // There we put the alpha for everything cluster to its phi
    // here we extract it from the middle of the tile
    auto& cl3d = output.emplace_back(detID, geom->getMatrixT2L(detID) ^ locXYZ); // local --> tracking
    if (applyMisalignment) {
      const auto lrID = geom->getLayer(detID);
      sigmaY2 += conf.sysErrY2[lrID];
      sigmaZ2 += conf.sysErrZ2[lrID];
    }
    cl3d.setErrors(sigmaY2, sigmaZ2, 0.f);
  }
}

int loadROFrameDataITS3(its::TimeFrame<7>* tf,
                        gsl::span<const o2::itsmft::ROFRecord> rofs,
                        gsl::span<const itsmft::CompClusterExt> clusters,
                        gsl::span<const unsigned char>::iterator& pattIt,
                        const its3::TopologyDictionary* dict,
                        const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  auto geom = its::GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  // tf->resetROFrameData(rofs.size()); // FIXME
  // tf->prepareROFrameData(rofs, clusters); FIXME

  its::bounded_vector<uint8_t> clusterSizeVec(clusters.size(), tf->getMemoryPool().get());

  for (size_t iRof{0}; iRof < rofs.size(); ++iRof) {
    const auto& rof = rofs[iRof];
    for (int clusterId{rof.getFirstEntry()}; clusterId < rof.getFirstEntry() + rof.getNEntries(); ++clusterId) {
      const auto& c = clusters[clusterId];
      const auto sensorID = c.getSensorID();
      const auto layer = geom->getLayer(sensorID);

      float sigmaY2{0}, sigmaZ2{0}, sigmaYZ{0};
      uint8_t clusterSize{0};
      const auto locXYZ = extractClusterData(c, pattIt, dict, sigmaY2, sigmaZ2, clusterSize);
      clusterSizeVec.push_back(clusterSize);

      // Transformation to the local --> global
      const auto gloXYZ = geom->getMatrixL2G(sensorID) * locXYZ;

      // Inverse transformation to the local --> tracking
      const o2::math_utils::Point3D<float> trkXYZ = geom->getMatrixT2L(sensorID) ^ locXYZ;

      // Tracking alpha angle
      // We want that each cluster rotates its tracking frame to the clusters phi
      // that way the track linearization around the measurement is less biases to the arc
      // this means automatically that the measurement on the arc is at 0 for the curved layers
      float alpha = geom->getSensorRefAlpha(sensorID);
      float x = trkXYZ.x(), y = trkXYZ.y();
      if (constants::detID::isDetITS3(sensorID)) {
        y = 0.f;
        x = std::hypot(gloXYZ.x(), gloXYZ.y());
        alpha = std::atan2(gloXYZ.y(), gloXYZ.x());
      }
      math_utils::detail::bringToPMPi(alpha); // alpha is defined on -Pi,Pi

      tf->addTrackingFrameInfoToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), x, alpha,
                                      std::array<float, 2>{y, trkXYZ.z()},
                                      std::array<float, 3>{sigmaY2, sigmaYZ, sigmaZ2});

      /// Rotate to the global frame
      tf->addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), tf->getUnsortedClusters()[layer].size());
      tf->addClusterExternalIndexToLayer(layer, clusterId);
    }
    for (unsigned int iL{0}; iL < tf->getUnsortedClusters().size(); ++iL) {
      tf->mROFramesClusters[iL][iRof + 1] = (int)tf->getUnsortedClusters()[iL].size();
    }
  }

  // tf->setClusterSize(clusterSizeVec); FIXME

  for (auto& v : tf->mNTrackletsPerCluster) {
    v.resize(tf->getUnsortedClusters()[1].size());
  }
  for (auto& v : tf->mNTrackletsPerClusterSum) {
    v.resize(tf->getUnsortedClusters()[1].size() + 1);
  }

  if (mcLabels != nullptr) {
    // tf->mClusterLabels = mcLabels; // FIXME
  }
  return 0;
}
} // namespace o2::its3::ioutils
