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

#include <boost/histogram.hpp>
#include <boost/format.hpp>

#include "ITStracking/VertexerTraits.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"

#ifdef VTX_DEBUG
#include "TTree.h"
#include "TFile.h"
#include <fstream>
#include <ostream>
#endif

#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace o2
{
namespace its
{
using boost::histogram::indexed;
using constants::math::TwoPi;

template <TrackletMode Mode, bool DryRun>
void trackleterKernelHost(
  const gsl::span<const Cluster>& clustersNextLayer,    // 0 2
  const gsl::span<const Cluster>& clustersCurrentLayer, // 1 1
  int* indexTableNext,
  const float phiCut,
  std::vector<Tracklet>& tracklets,
  gsl::span<int> foundTracklets,
  const IndexTableUtils& utils,
  const int rof,
  const int rofFoundTrackletsOffset,
  const int maxTrackletsPerCluster = static_cast<int>(2e3))
{
  const int PhiBins{utils.getNphiBins()};
  const int ZBins{utils.getNzBins()};
  // loop on layer1 clusters
  int cumulativeStoredTracklets{0};
  for (int iCurrentLayerClusterIndex = 0; iCurrentLayerClusterIndex < clustersCurrentLayer.size(); ++iCurrentLayerClusterIndex) {
    int storedTracklets{0};
    const Cluster& currentCluster{clustersCurrentLayer[iCurrentLayerClusterIndex]};
    const int4 selectedBinsRect{VertexerTraits::getBinsRect(currentCluster, (int)Mode, 0.f, 50.f, phiCut / 2, utils)};
    if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
      int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
      if (phiBinsNum < 0) {
        phiBinsNum += PhiBins;
      }
      // loop on phi bins next layer
      for (int iPhiBin{selectedBinsRect.y}, iPhiCount{0}; iPhiCount < phiBinsNum; iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++) {
        const int firstBinIndex{utils.getBinIndex(selectedBinsRect.x, iPhiBin)};
        const int firstRowClusterIndex{indexTableNext[firstBinIndex]};
        const int maxRowClusterIndex{indexTableNext[firstBinIndex + ZBins]};
        // loop on clusters next layer
        for (int iNextLayerClusterIndex{firstRowClusterIndex}; iNextLayerClusterIndex < maxRowClusterIndex && iNextLayerClusterIndex < static_cast<int>(clustersNextLayer.size()); ++iNextLayerClusterIndex) {
          const Cluster& nextCluster{clustersNextLayer[iNextLayerClusterIndex]};
          if (o2::gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi) < phiCut) {
            if (storedTracklets < maxTrackletsPerCluster) {
              if constexpr (!DryRun) {
                if constexpr (Mode == TrackletMode::Layer0Layer1) {
                  tracklets[rofFoundTrackletsOffset + cumulativeStoredTracklets + storedTracklets] = Tracklet{iNextLayerClusterIndex, iCurrentLayerClusterIndex, nextCluster, currentCluster, rof, rof};
                } else {
                  tracklets[rofFoundTrackletsOffset + cumulativeStoredTracklets + storedTracklets] = Tracklet{iCurrentLayerClusterIndex, iNextLayerClusterIndex, currentCluster, nextCluster, rof, rof};
                }
              }
              ++storedTracklets;
            }
          }
        }
      }
    }
    if constexpr (DryRun) {
      foundTracklets[iCurrentLayerClusterIndex] = storedTracklets;
    } else {
      cumulativeStoredTracklets += storedTracklets;
    }
  }
}

void trackletSelectionKernelHost(
  const gsl::span<const Cluster> clusters0, // 0
  const gsl::span<const Cluster> clusters1, // 1
  const gsl::span<const Tracklet>& tracklets01,
  const gsl::span<const Tracklet>& tracklets12,
  const gsl::span<int> foundTracklets01,
  const gsl::span<int> foundTracklets12,
  std::vector<Line>& destTracklets,
  const gsl::span<const MCCompLabel>& trackletLabels,
  std::vector<MCCompLabel>& linesLabels,
  const float tanLambdaCut = 0.025f,
  const float phiCut = 0.005f,
  const int maxTracklets = static_cast<int>(1e2))
{
  int offset01{0}, offset12{0};
  std::vector<bool> usedTracklets(tracklets01.size(), false);
  for (unsigned int iCurrentLayerClusterIndex{0}; iCurrentLayerClusterIndex < clusters1.size(); ++iCurrentLayerClusterIndex) {
    int validTracklets{0};
    for (int iTracklet12{offset12}; iTracklet12 < offset12 + foundTracklets12[iCurrentLayerClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{offset01}; iTracklet01 < offset01 + foundTracklets01[iCurrentLayerClusterIndex]; ++iTracklet01) {
        const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(tracklets01[iTracklet01].tanLambda - tracklets12[iTracklet12].tanLambda)};
        const float deltaPhi{o2::gpu::GPUCommonMath::Abs(tracklets01[iTracklet01].phi - tracklets12[iTracklet12].phi)};
        if (!usedTracklets[iTracklet01] && deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets != maxTracklets) {
          usedTracklets[iTracklet01] = true;
          destTracklets.emplace_back(tracklets01[iTracklet01], clusters0.data(), clusters1.data());
          if (trackletLabels.size()) {
            linesLabels.emplace_back(trackletLabels[iTracklet01]);
          }
          ++validTracklets;
        }
      }
    }
    offset01 += foundTracklets01[iCurrentLayerClusterIndex];
    offset12 += foundTracklets12[iCurrentLayerClusterIndex];
  }
}

const std::vector<std::pair<int, int>> VertexerTraits::selectClusters(const int* indexTable,
                                                                      const std::array<int, 4>& selectedBinsRect,
                                                                      const IndexTableUtils& utils)
{
  std::vector<std::pair<int, int>> filteredBins{};
  int phiBinsNum{selectedBinsRect[3] - selectedBinsRect[1] + 1};
  if (phiBinsNum < 0) {
    phiBinsNum += utils.getNphiBins();
  }
  filteredBins.reserve(phiBinsNum);
  for (int iPhiBin{selectedBinsRect[1]}, iPhiCount{0}; iPhiCount < phiBinsNum;
       iPhiBin = ++iPhiBin == utils.getNphiBins() ? 0 : iPhiBin, iPhiCount++) {
    const int firstBinIndex{utils.getBinIndex(selectedBinsRect[0], iPhiBin)};
    filteredBins.emplace_back(
      indexTable[firstBinIndex],
      utils.countRowSelectedBins(indexTable, iPhiBin, selectedBinsRect[0], selectedBinsRect[2]));
  }
  return filteredBins;
}

void VertexerTraits::updateVertexingParameters(const VertexingParameters& vrtPar)
{
  mVrtParams = vrtPar;
  mIndexTableUtils.setTrackingParameters(vrtPar);
  mVrtParams.phiSpan = static_cast<int>(std::ceil(mIndexTableUtils.getNphiBins() * mVrtParams.phiCut /
                                                  constants::math::TwoPi));
  mVrtParams.zSpan = static_cast<int>(std::ceil(mVrtParams.zCut * mIndexTableUtils.getInverseZCoordinate(0)));
  setNThreads(mVrtParams.nThreads);
}

void VertexerTraits::computeTracklets()
{

#pragma omp parallel num_threads(mNThreads)
  {
#pragma omp for schedule(dynamic)
    for (int rofId = 0; rofId < mTimeFrame->getNrof(); ++rofId) {
      trackleterKernelHost<TrackletMode::Layer0Layer1, true>(
        mTimeFrame->getClustersOnLayer(rofId, 0),
        mTimeFrame->getClustersOnLayer(rofId, 1),
        mTimeFrame->getIndexTable(rofId, 0).data(),
        mVrtParams.phiCut,
        mTimeFrame->getTracklets()[0],
        mTimeFrame->getNTrackletsCluster(rofId, 0),
        mIndexTableUtils,
        rofId,
        0,
        mVrtParams.maxTrackletsPerCluster);
      trackleterKernelHost<TrackletMode::Layer1Layer2, true>(
        mTimeFrame->getClustersOnLayer(rofId, 2),
        mTimeFrame->getClustersOnLayer(rofId, 1),
        mTimeFrame->getIndexTable(rofId, 2).data(),
        mVrtParams.phiCut,
        mTimeFrame->getTracklets()[1],
        mTimeFrame->getNTrackletsCluster(rofId, 1),
        mIndexTableUtils,
        rofId,
        0,
        mVrtParams.maxTrackletsPerCluster);
      mTimeFrame->getNTrackletsROf(rofId, 0) = std::accumulate(mTimeFrame->getNTrackletsCluster(rofId, 0).begin(), mTimeFrame->getNTrackletsCluster(rofId, 0).end(), 0);
      mTimeFrame->getNTrackletsROf(rofId, 1) = std::accumulate(mTimeFrame->getNTrackletsCluster(rofId, 1).begin(), mTimeFrame->getNTrackletsCluster(rofId, 1).end(), 0);
    }
#pragma omp single
    mTimeFrame->computeTrackletsScans(mNThreads);
#pragma omp single
    mTimeFrame->getTracklets()[0].resize(mTimeFrame->getTotalTrackletsTF(0));
#pragma omp single
    mTimeFrame->getTracklets()[1].resize(mTimeFrame->getTotalTrackletsTF(1));

#pragma omp for schedule(dynamic)
    for (int rofId = 0; rofId < mTimeFrame->getNrof(); ++rofId) {
      trackleterKernelHost<TrackletMode::Layer0Layer1, false>(
        mTimeFrame->getClustersOnLayer(rofId, 0),
        mTimeFrame->getClustersOnLayer(rofId, 1),
        mTimeFrame->getIndexTable(rofId, 0).data(),
        mVrtParams.phiCut,
        mTimeFrame->getTracklets()[0],
        mTimeFrame->getNTrackletsCluster(rofId, 0),
        mIndexTableUtils,
        rofId,
        mTimeFrame->getNTrackletsROf(rofId, 0),
        mVrtParams.maxTrackletsPerCluster);
      trackleterKernelHost<TrackletMode::Layer1Layer2, false>(
        mTimeFrame->getClustersOnLayer(rofId, 2),
        mTimeFrame->getClustersOnLayer(rofId, 1),
        mTimeFrame->getIndexTable(rofId, 2).data(),
        mVrtParams.phiCut,
        mTimeFrame->getTracklets()[1],
        mTimeFrame->getNTrackletsCluster(rofId, 1),
        mIndexTableUtils,
        rofId,
        mTimeFrame->getNTrackletsROf(rofId, 1),
        mVrtParams.maxTrackletsPerCluster);
    }
  }

  /// Create tracklets labels for L0-L1, information is as flat as in tracklets vector (no rofId)
  if (mTimeFrame->hasMCinformation()) {
    for (auto& trk : mTimeFrame->getTracklets()[0]) {
      MCCompLabel label;
      int sortedId0{mTimeFrame->getSortedIndex(trk.rof[0], 0, trk.firstClusterIndex)};
      int sortedId1{mTimeFrame->getSortedIndex(trk.rof[0], 1, trk.secondClusterIndex)};
      for (auto& lab0 : mTimeFrame->getClusterLabels(0, mTimeFrame->getClusters()[0][sortedId0].clusterId)) {
        for (auto& lab1 : mTimeFrame->getClusterLabels(1, mTimeFrame->getClusters()[1][sortedId1].clusterId)) {
          if (lab0 == lab1 && lab0.isValid()) {
            label = lab0;
            break;
          }
        }
        if (label.isValid()) {
          break;
        }
      }
      mTimeFrame->getTrackletsLabel(0).emplace_back(label);
    }
  }

#ifdef VTX_DEBUG
  // Dump on file
  TFile* trackletFile = TFile::Open("artefacts_tf.root", "recreate");
  TTree* tr_tre = new TTree("tracklets", "tf");
  std::vector<o2::its::Tracklet> trkl_vec_0(0);
  std::vector<o2::its::Tracklet> trkl_vec_1(0);
  std::vector<o2::its::Cluster> clus0(0);
  std::vector<o2::its::Cluster> clus1(0);
  std::vector<o2::its::Cluster> clus2(0);
  tr_tre->Branch("Tracklets0", &trkl_vec_0);
  tr_tre->Branch("Tracklets1", &trkl_vec_1);
  for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    trkl_vec_0.clear();
    trkl_vec_1.clear();
    for (auto& tr : mTimeFrame->getFoundTracklets(rofId, 0)) {
      trkl_vec_0.push_back(tr);
    }
    for (auto& tr : mTimeFrame->getFoundTracklets(rofId, 1)) {
      trkl_vec_1.push_back(tr);
    }
    tr_tre->Fill();
  }
  trackletFile->cd();
  tr_tre->Write();
  trackletFile->Close();

  std::ofstream out01("NTC01_cpu.txt"), out12("NTC12_cpu.txt");
  for (int iRof{0}; iRof < mTimeFrame->getNrof(); ++iRof) {
    std::copy(mTimeFrame->getNTrackletsCluster(iRof, 0).begin(), mTimeFrame->getNTrackletsCluster(iRof, 0).end(), std::ostream_iterator<double>(out01, "\t"));
    std::copy(mTimeFrame->getNTrackletsCluster(iRof, 1).begin(), mTimeFrame->getNTrackletsCluster(iRof, 1).end(), std::ostream_iterator<double>(out12, "\t"));
    out01 << std::endl;
    out12 << std::endl;
  }
  out01.close();
  out12.close();
#endif
} // namespace its

void VertexerTraits::computeTrackletMatching()
{
#pragma omp parallel for num_threads(mNThreads) schedule(dynamic)
  for (int rofId = 0; rofId < mTimeFrame->getNrof(); ++rofId) {
    mTimeFrame->getLines(rofId).reserve(mTimeFrame->getNTrackletsCluster(rofId, 0).size());
    trackletSelectionKernelHost(
      mTimeFrame->getClustersOnLayer(rofId, 0),
      mTimeFrame->getClustersOnLayer(rofId, 1),
      mTimeFrame->getFoundTracklets(rofId, 0),
      mTimeFrame->getFoundTracklets(rofId, 1),
      mTimeFrame->getNTrackletsCluster(rofId, 0),
      mTimeFrame->getNTrackletsCluster(rofId, 1),
      mTimeFrame->getLines(rofId),
      mTimeFrame->getLabelsFoundTracklets(rofId, 0),
      mTimeFrame->getLinesLabel(rofId),
      mVrtParams.tanLambdaCut,
      mVrtParams.phiCut);
  }

#ifdef VTX_DEBUG
  TFile* trackletFile = TFile::Open("artefacts_tf.root", "update");
  TTree* ln_tre = new TTree("lines", "tf");
  std::vector<o2::its::Line> lines_vec(0);
  std::vector<int> nTrackl01(0);
  std::vector<int> nTrackl12(0);
  ln_tre->Branch("Lines", &lines_vec);
  ln_tre->Branch("NTrackletCluster01", &nTrackl01);
  ln_tre->Branch("NTrackletCluster12", &nTrackl12);
  for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    lines_vec.clear();
    nTrackl01.clear();
    nTrackl12.clear();
    for (auto& ln : mTimeFrame->getLines(rofId)) {
      lines_vec.push_back(ln);
    }
    for (auto& n : mTimeFrame->getNTrackletsCluster(rofId, 0)) {
      nTrackl01.push_back(n);
    }
    for (auto& n : mTimeFrame->getNTrackletsCluster(rofId, 1)) {
      nTrackl12.push_back(n);
    }

    ln_tre->Fill();
  }
  trackletFile->cd();
  ln_tre->Write();
  trackletFile->Close();
#endif
}

void VertexerTraits::computeVertices()
{
#ifdef VTX_DEBUG
  std::vector<std::vector<ClusterLines>> dbg_clusLines(mTimeFrame->getNrof());
#endif
  std::vector<int> noClustersVec(mTimeFrame->getNrof(), 0);
  for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    const int numTracklets{static_cast<int>(mTimeFrame->getLines(rofId).size())};
    std::vector<bool> usedTracklets(numTracklets, false);
    for (int line1{0}; line1 < numTracklets; ++line1) {
      if (usedTracklets[line1]) {
        continue;
      }
      for (int line2{line1 + 1}; line2 < numTracklets; ++line2) {
        if (usedTracklets[line2]) {
          continue;
        }
        if (Line::getDCA(mTimeFrame->getLines(rofId)[line1], mTimeFrame->getLines(rofId)[line2]) < mVrtParams.pairCut) {
          mTimeFrame->getTrackletClusters(rofId).emplace_back(line1, mTimeFrame->getLines(rofId)[line1], line2, mTimeFrame->getLines(rofId)[line2]);
          std::array<float, 3> tmpVertex{mTimeFrame->getTrackletClusters(rofId).back().getVertex()};
          if (tmpVertex[0] * tmpVertex[0] + tmpVertex[1] * tmpVertex[1] > 4.f) {
            mTimeFrame->getTrackletClusters(rofId).pop_back();
            break;
          }
          usedTracklets[line1] = true;
          usedTracklets[line2] = true;
          for (int tracklet3{0}; tracklet3 < numTracklets; ++tracklet3) {
            if (usedTracklets[tracklet3]) {
              continue;
            }
            if (Line::getDistanceFromPoint(mTimeFrame->getLines(rofId)[tracklet3], tmpVertex) < mVrtParams.pairCut) {
              mTimeFrame->getTrackletClusters(rofId).back().add(tracklet3, mTimeFrame->getLines(rofId)[tracklet3]);
              usedTracklets[tracklet3] = true;
              tmpVertex = mTimeFrame->getTrackletClusters(rofId).back().getVertex();
            }
          }
          break;
        }
      }
    }
    std::sort(mTimeFrame->getTrackletClusters(rofId).begin(), mTimeFrame->getTrackletClusters(rofId).end(),
              [](ClusterLines& cluster1, ClusterLines& cluster2) { return cluster1.getSize() > cluster2.getSize(); });
    noClustersVec[rofId] = static_cast<int>(mTimeFrame->getTrackletClusters(rofId).size());
    for (int iCluster1{0}; iCluster1 < noClustersVec[rofId]; ++iCluster1) {
      std::array<float, 3> vertex1{mTimeFrame->getTrackletClusters(rofId)[iCluster1].getVertex()};
      std::array<float, 3> vertex2{};
      for (int iCluster2{iCluster1 + 1}; iCluster2 < noClustersVec[rofId]; ++iCluster2) {
        vertex2 = mTimeFrame->getTrackletClusters(rofId)[iCluster2].getVertex();
        if (std::abs(vertex1[2] - vertex2[2]) < mVrtParams.clusterCut) {
          float distance{(vertex1[0] - vertex2[0]) * (vertex1[0] - vertex2[0]) +
                         (vertex1[1] - vertex2[1]) * (vertex1[1] - vertex2[1]) +
                         (vertex1[2] - vertex2[2]) * (vertex1[2] - vertex2[2])};
          if (distance < mVrtParams.pairCut * mVrtParams.pairCut) {
            for (auto label : mTimeFrame->getTrackletClusters(rofId)[iCluster2].getLabels()) {
              mTimeFrame->getTrackletClusters(rofId)[iCluster1].add(label, mTimeFrame->getLines(rofId)[label]);
              vertex1 = mTimeFrame->getTrackletClusters(rofId)[iCluster1].getVertex();
            }
          }
          mTimeFrame->getTrackletClusters(rofId).erase(mTimeFrame->getTrackletClusters(rofId).begin() + iCluster2);
          --iCluster2;
          --noClustersVec[rofId];
        }
      }
    }
  }

  for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
#ifdef VTX_DEBUG
    for (auto& cl : mTimeFrame->getTrackletClusters(rofId)) {
      dbg_clusLines[rofId].push_back(cl);
    }
#endif
    for (int iCluster{0}; iCluster < noClustersVec[rofId]; ++iCluster) {
      if (mTimeFrame->getTrackletClusters(rofId)[iCluster].getSize() < mVrtParams.clusterContributorsCut && noClustersVec[rofId] > 1) {
        mTimeFrame->getTrackletClusters(rofId).erase(mTimeFrame->getTrackletClusters(rofId).begin() + iCluster);
        noClustersVec[rofId]--;
        continue;
      }
      if (mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[0] * mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[0] +
            mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[1] * mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[1] <
          1.98 * 1.98) {
        mVertices.emplace_back(mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[0],
                               mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[1],
                               mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[2],
                               mTimeFrame->getTrackletClusters(rofId)[iCluster].getRMS2(),         // Symm matrix. Diagonal: RMS2 components,
                                                                                                   // off-diagonal: square mean of projections on planes.
                               mTimeFrame->getTrackletClusters(rofId)[iCluster].getSize(),         // Contributors
                               mTimeFrame->getTrackletClusters(rofId)[iCluster].getAvgDistance2(), // In place of chi2
                               rofId);
        if (mTimeFrame->hasMCinformation()) {
          mTimeFrame->getVerticesLabels().emplace_back();
          for (auto& index : mTimeFrame->getTrackletClusters(rofId)[iCluster].getLabels()) {
            mTimeFrame->getVerticesLabels().back().push_back(mTimeFrame->getLinesLabel(rofId)[index]); // then we can use nContributors from vertices to get the labels
          }
        }
      }
    }
    std::vector<Vertex> vertices;
    for (auto& vertex : mVertices) {
      vertices.emplace_back(o2::math_utils::Point3D<float>(vertex.mX, vertex.mY, vertex.mZ), vertex.mRMS2, vertex.mContributors, vertex.mAvgDistance2);
      vertices.back().setTimeStamp(vertex.mTimeStamp);
    }
    mTimeFrame->addPrimaryVertices(vertices);
    mVertices.clear();
  }
#ifdef VTX_DEBUG
  TFile* dbg_file = TFile::Open("artefacts_tf.root", "update");
  TTree* ln_clus_lines_tree = new TTree("clusterlines", "tf");
  std::vector<o2::its::ClusterLines> cl_lines_vec_pre(0);
  std::vector<o2::its::ClusterLines> cl_lines_vec_post(0);
  ln_clus_lines_tree->Branch("cllines_pre", &cl_lines_vec_pre);
  ln_clus_lines_tree->Branch("cllines_post", &cl_lines_vec_post);
  for (auto rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    cl_lines_vec_pre.clear();
    cl_lines_vec_post.clear();
    for (auto& clln : mTimeFrame->getTrackletClusters(rofId)) {
      cl_lines_vec_post.push_back(clln);
    }
    for (auto& cl : dbg_clusLines[rofId]) {
      cl_lines_vec_pre.push_back(cl);
    }
    ln_clus_lines_tree->Fill();
  }
  dbg_file->cd();
  ln_clus_lines_tree->Write();
  dbg_file->Close();
#endif
}

void VertexerTraits::setNThreads(int n)
{
#ifdef WITH_OPENMP
  mNThreads = n > 0 ? n : 1;
#else
  mNThreads = 1;
#endif
  LOGP(info, "Setting seeding vertexer with {} threads.", mNThreads);
}

// void VertexerTraits::computeHistVertices()
// {
//   o2::its::VertexerHistogramsConfiguration histConf;
//   std::vector<boost::histogram::axis::regular<float>> axes;
//   axes.reserve(3);
//   for (size_t iAxis{0}; iAxis < 3; ++iAxis) {
//     axes.emplace_back(histConf.nBinsXYZ[iAxis] - 1, histConf.lowHistBoundariesXYZ[iAxis], histConf.highHistBoundariesXYZ[iAxis]);
//   }

//   auto histX = boost::histogram::make_histogram(axes[0]);
//   auto histY = boost::histogram::make_histogram(axes[1]);
//   auto histZ = boost::histogram::make_histogram(axes[2]);

//   // Loop over lines, calculate transverse vertices within beampipe and fill XY histogram to find pseudobeam projection
//   for (size_t iTracklet1{0}; iTracklet1 < mTracklets.size(); ++iTracklet1) {
//     for (size_t iTracklet2{iTracklet1 + 1}; iTracklet2 < mTracklets.size(); ++iTracklet2) {
//       if (Line::getDCA(mTracklets[iTracklet1], mTracklets[iTracklet2]) < mVrtParams.histPairCut) {
//         ClusterLines cluster{mTracklets[iTracklet1], mTracklets[iTracklet2]};
//         if (cluster.getVertex()[0] * cluster.getVertex()[0] + cluster.getVertex()[1] * cluster.getVertex()[1] < 1.98f * 1.98f) {
//           histX(cluster.getVertex()[0]);
//           histY(cluster.getVertex()[1]);
//         }
//       }
//     }
//   }

//   // Try again to use std::max_element as soon as boost is upgraded to 1.71...
//   // atm you can iterate over histograms, not really possible to get bin index. Need to use iterate(histogram)
//   int maxXBinContent{0};
//   int maxYBinContent{0};
//   int maxXIndex{0};
//   int maxYIndex{0};
//   for (auto x : indexed(histX)) {
//     if (x.get() > maxXBinContent) {
//       maxXBinContent = x.get();
//       maxXIndex = x.index();
//     }
//   }

//   for (auto y : indexed(histY)) {
//     if (y.get() > maxYBinContent) {
//       maxYBinContent = y.get();
//       maxYIndex = y.index();
//     }
//   }

//   // Compute weighted average around XY to smooth the position
//   if (maxXBinContent || maxYBinContent) {
//     float tmpX{histConf.lowHistBoundariesXYZ[0] + histConf.binSizeHistX * maxXIndex + histConf.binSizeHistX / 2};
//     float tmpY{histConf.lowHistBoundariesXYZ[1] + histConf.binSizeHistY * maxYIndex + histConf.binSizeHistY / 2};
//     int sumX{maxXBinContent};
//     int sumY{maxYBinContent};
//     float wX{tmpX * static_cast<float>(maxXBinContent)};
//     float wY{tmpY * static_cast<float>(maxYBinContent)};
//     for (int iBinX{std::max(0, maxXIndex - histConf.binSpanXYZ[0])}; iBinX < std::min(maxXIndex + histConf.binSpanXYZ[0] + 1, histConf.nBinsXYZ[0] - 1); ++iBinX) {
//       if (iBinX != maxXIndex) {
//         wX += (histConf.lowHistBoundariesXYZ[0] + histConf.binSizeHistX * iBinX + histConf.binSizeHistX / 2) * histX.at(iBinX);
//         sumX += histX.at(iBinX);
//       }
//     }
//     for (int iBinY{std::max(0, maxYIndex - histConf.binSpanXYZ[1])}; iBinY < std::min(maxYIndex + histConf.binSpanXYZ[1] + 1, histConf.nBinsXYZ[1] - 1); ++iBinY) {
//       if (iBinY != maxYIndex) {
//         wY += (histConf.lowHistBoundariesXYZ[1] + histConf.binSizeHistY * iBinY + histConf.binSizeHistY / 2) * histY.at(iBinY);
//         sumY += histY.at(iBinY);
//       }
//     }

//     const float beamCoordinateX{wX / sumX};
//     const float beamCoordinateY{wY / sumY};

//     // create actual pseudobeam line
//     Line pseudoBeam{std::array<float, 3>{beamCoordinateX, beamCoordinateY, 1}, std::array<float, 3>{beamCoordinateX, beamCoordinateY, -1}};

//     // Fill z coordinate histogram
//     for (auto& line : mTracklets) {
//       if (Line::getDCA(line, pseudoBeam) < mVrtParams.histPairCut) {
//         ClusterLines cluster{line, pseudoBeam};
//         histZ(cluster.getVertex()[2]);
//       }
//     }
//     for (int iVertex{0};; ++iVertex) {
//       int maxZBinContent{0};
//       int maxZIndex{0};
//       // find maximum
//       for (auto z : indexed(histZ)) {
//         if (z.get() > maxZBinContent) {
//           maxZBinContent = z.get();
//           maxZIndex = z.index();
//         }
//       }
//       float tmpZ{histConf.lowHistBoundariesXYZ[2] + histConf.binSizeHistZ * maxZIndex + histConf.binSizeHistZ / 2};
//       int sumZ{maxZBinContent};
//       float wZ{tmpZ * static_cast<float>(maxZBinContent)};
//       for (int iBinZ{std::max(0, maxZIndex - histConf.binSpanXYZ[2])}; iBinZ < std::min(maxZIndex + histConf.binSpanXYZ[2] + 1, histConf.nBinsXYZ[2] - 1); ++iBinZ) {
//         if (iBinZ != maxZIndex) {
//           wZ += (histConf.lowHistBoundariesXYZ[2] + histConf.binSizeHistZ * iBinZ + histConf.binSizeHistZ / 2) * histZ.at(iBinZ);
//           sumZ += histZ.at(iBinZ);
//           histZ.at(iBinZ) = 0;
//         }
//       }
//       if ((sumZ < mVrtParams.clusterContributorsCut) && (iVertex != 0)) {
//         break;
//       }
//       histZ.at(maxZIndex) = 0;
//       const float vertexZCoordinate{wZ / sumZ};
//       mVertices.emplace_back(beamCoordinateX,
//                              beamCoordinateY,
//                              vertexZCoordinate,
//                              std::array<float, 6>{0., 0., 0., 0., 0., 0.},
//                              sumZ,
//                              0.,
//                              mEvent->getROFrameId());
//     }
//   }
// }
} // namespace its
} // namespace o2
