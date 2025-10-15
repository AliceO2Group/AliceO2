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

#include "ITS3Reconstruction/IOUtils.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/BoundedAllocator.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITS3Reconstruction/TopologyDictionary.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITS3Base/SpecsV2.h"
#include "ITStracking/TrackingConfigParam.h"
#include "Framework/Logger.h"

#include <limits>

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
    float sigmaY2, sigmaZ2, sigmaYZ = 0;
    auto locXYZ = extractClusterData(c, pattIt, dict, sigmaY2, sigmaZ2);
    const auto detID = c.getSensorID();
    auto& cl3d = output.emplace_back(detID, geom->getMatrixT2L(detID) ^ locXYZ); // local --> tracking
    if (applyMisalignment) {
      auto lrID = geom->getLayer(detID);
      sigmaY2 += conf.sysErrY2[lrID];
      sigmaZ2 += conf.sysErrZ2[lrID];
    }
    cl3d.setErrors(sigmaY2, sigmaZ2, sigmaYZ);
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

  tf->resetROFrameData(rofs.size());
  tf->prepareROFrameData(rofs, clusters);

  its::bounded_vector<uint8_t> clusterSizeVec(clusters.size(), tf->getMemoryPool().get());

  for (size_t iRof{0}; iRof < rofs.size(); ++iRof) {
    const auto& rof = rofs[iRof];
    for (int clusterId{rof.getFirstEntry()}; clusterId < rof.getFirstEntry() + rof.getNEntries(); ++clusterId) {
      auto& c = clusters[clusterId];
      auto sensorID = c.getSensorID();
      auto layer = geom->getLayer(sensorID);

      float sigmaY2{0}, sigmaZ2{0}, sigmaYZ{0};
      uint8_t clusterSize{0};
      auto locXYZ = extractClusterData(c, pattIt, dict, sigmaY2, sigmaZ2, clusterSize);
      clusterSizeVec.push_back(clusterSize);

      // Transformation to the local --> global
      auto gloXYZ = geom->getMatrixL2G(sensorID) * locXYZ;

      // Inverse transformation to the local --> tracking
      o2::math_utils::Point3D<float> trkXYZ = geom->getMatrixT2L(sensorID) ^ locXYZ;

      // Tracking alpha angle
      float alpha = geom->getSensorRefAlpha(sensorID);

      tf->addTrackingFrameInfoToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), trkXYZ.x(), alpha,
                                      std::array<float, 2>{trkXYZ.y(), trkXYZ.z()},
                                      std::array<float, 3>{sigmaY2, sigmaYZ, sigmaZ2});

      /// Rotate to the global frame
      tf->addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), tf->getUnsortedClusters()[layer].size());
      tf->addClusterExternalIndexToLayer(layer, clusterId);
    }
    for (unsigned int iL{0}; iL < tf->getUnsortedClusters().size(); ++iL) {
      tf->mROFramesClusters[iL][iRof + 1] = tf->getUnsortedClusters()[iL].size();
    }
  }

  tf->setClusterSize(clusterSizeVec);

  for (auto& v : tf->mNTrackletsPerCluster) {
    v.resize(tf->getUnsortedClusters()[1].size());
  }
  for (auto& v : tf->mNTrackletsPerClusterSum) {
    v.resize(tf->getUnsortedClusters()[1].size() + 1);
  }

  if (mcLabels != nullptr) {
    tf->mClusterLabels = mcLabels;
  }
  return tf->mNrof;
}
} // namespace o2::its3::ioutils
