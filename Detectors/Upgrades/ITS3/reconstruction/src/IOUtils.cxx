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

#include <ITS3Reconstruction/IOUtils.h>
#include <ITStracking/IOUtils.h>
#include <ITStracking/TimeFrame.h>
#include <DataFormatsITS3/CompCluster.h>
#include <DataFormatsITSMFT/ROFRecord.h>
#include <ITS3Reconstruction/TopologyDictionary.h>
#include <ITSBase/GeometryTGeo.h>
#include <ITSMFTBase/SegmentationAlpide.h>
#include <ITS3Base/SegmentationSuperAlpide.h>
#include <Framework/Logger.h>

namespace o2
{
namespace its3
{
namespace ioutils
{
int loadROFrameDataITS3(its::TimeFrame* tf,
                        gsl::span<o2::itsmft::ROFRecord> rofs,
                        gsl::span<const its3::CompClusterExt> clusters,
                        gsl::span<const unsigned char>::iterator& pattIt,
                        const its3::TopologyDictionary* dict,
                        const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  its::GeometryTGeo* geom = its::GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  std::vector<o2::its3::SegmentationSuperAlpide> segITS3;
  for (int iLayer{0}; iLayer < geom->getNumberOfLayers() - 4; ++iLayer) {
    for (int iChip{0}; iChip < geom->getNumberOfChipsPerLayer(iLayer); ++iChip) {
      segITS3.push_back(SegmentationSuperAlpide(iLayer));
    }
  }
  int nChipsITS3 = segITS3.size();

  tf->mNrof = 0;
  for (auto& rof : rofs) {
    for (int clusterId{rof.getFirstEntry()}; clusterId < rof.getFirstEntry() + rof.getNEntries(); ++clusterId) {
      auto& c = clusters[clusterId];
      auto sensorID = c.getSensorID();
      int layer = layer = geom->getLayer(sensorID);

      auto pattID = c.getPatternID();
      o2::math_utils::Point3D<float> locXYZ;
      float sigmaY2 = o2::its::ioutils::DefClusError2Row, sigmaZ2 = o2::its::ioutils::DefClusError2Col, sigmaYZ = 0; // Dummy COG errors (about half pixel size)
      float pitchRow = ((sensorID < nChipsITS3) ? segITS3[sensorID].mPitchRow : o2::itsmft::SegmentationAlpide::PitchRow);
      float pitchCol = ((sensorID < nChipsITS3) ? segITS3[sensorID].mPitchCol : o2::itsmft::SegmentationAlpide::PitchCol);
      if (pattID != its3::CompCluster::InvalidPatternID) {
        sigmaY2 = dict->getErr2X(pattID) * pitchRow * pitchRow;
        sigmaZ2 = dict->getErr2Z(pattID) * pitchCol * pitchCol;
        if (!dict->isGroup(pattID)) {
          locXYZ = dict->getClusterCoordinates(c, nChipsITS3);
        } else {
          o2::itsmft::ClusterPattern patt(pattIt);
          locXYZ = dict->getClusterCoordinates(c, patt, nChipsITS3);
          sigmaY2 = patt.getRowSpan() * patt.getRowSpan() * pitchRow * pitchRow / 12.;
          sigmaZ2 = patt.getColumnSpan() * patt.getColumnSpan() * pitchCol * pitchCol / 12.;
        }
      } else {
        o2::itsmft::ClusterPattern patt(pattIt);
        sigmaY2 = patt.getRowSpan() * patt.getRowSpan() * pitchRow * pitchRow / 12.;
        sigmaZ2 = patt.getColumnSpan() * patt.getColumnSpan() * pitchCol * pitchCol / 12.;
        locXYZ = dict->getClusterCoordinates(c, patt, false, nChipsITS3);
      }

      // Transformation to the local --> global
      auto gloXYZ = geom->getMatrixL2G(sensorID) * locXYZ;

      // for cylindrical layers we have a different alpha for each cluster, for regular silicon detectors instead a single alpha for the whole sensor
      float alpha = 0.;
      o2::math_utils::Point3D<float> trkXYZ;
      if (layer < geom->getNumberOfLayers() - 4) {
        alpha = geom->getAlphaFromGlobalITS3(sensorID, gloXYZ);
        // Inverse transformation to the local --> tracking
        trkXYZ = geom->getT2LMatrixITS3(sensorID, alpha) ^ locXYZ;
      } else {
        alpha = geom->getSensorRefAlpha(sensorID);
        // Inverse transformation to the local --> tracking
        trkXYZ = geom->getMatrixT2L(sensorID) ^ locXYZ;
      }

      tf->addTrackingFrameInfoToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), trkXYZ.x(), alpha,
                                      std::array<float, 2>{trkXYZ.y(), trkXYZ.z()},
                                      std::array<float, 3>{sigmaY2, sigmaYZ, sigmaZ2});

      /// Rotate to the global frame
      tf->addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), tf->getUnsortedClusters()[layer].size());
      tf->addClusterExternalIndexToLayer(layer, clusterId);
    }
    for (unsigned int iL{0}; iL < tf->getUnsortedClusters().size(); ++iL) {
      tf->mROframesClusters[iL].push_back(tf->getUnsortedClusters()[iL].size());
    }
    tf->mNrof++;
  }

  for (auto& v : tf->mNTrackletsPerCluster) {
    v.resize(tf->getUnsortedClusters()[1].size());
  }

  if (mcLabels) {
    tf->mClusterLabels = mcLabels;
  }
  return tf->mNrof;
}
} // namespace ioutils
} // namespace its3
} // namespace o2
