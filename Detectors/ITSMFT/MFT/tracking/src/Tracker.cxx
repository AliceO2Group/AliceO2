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
/// \file Tracker.cxx
///

#include "MFTTracking/Tracker.h"
#include "MFTTracking/Cluster.h"
#include "MFTTracking/Cell.h"
#include "MFTTracking/TrackCA.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "ReconstructionDataFormats/Track.h"
#include "Framework/Logger.h"

namespace o2
{
namespace mft
{

//_________________________________________________________________________________________________
template <typename T>
Tracker<T>::Tracker(bool useMC) : mUseMC{useMC}
{
  mTrackFitter = std::make_unique<o2::mft::TrackFitter<T>>();
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::setBz(Float_t bz)
{
  /// Configure track propagation
  mBz = bz;
  mTrackFitter->setBz(bz);
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::configure(const MFTTrackingParam& trkParam, bool printConfig)
{
  /// initialize from MFTTrackingParam (command line configuration parameters)
  initialize(trkParam);
  initializeFinder();

  mTrackFitter->setMFTRadLength(trkParam.MFTRadLength);
  mTrackFitter->setVerbosity(trkParam.verbose);
  mTrackFitter->setAlignResiduals(trkParam.alignResidual);
  if (trkParam.forceZeroField || (std::abs(mBz) < o2::constants::math::Almost0)) {
    mTrackFitter->setTrackModel(o2::mft::MFTTrackModel::Linear);
  } else {
    mTrackFitter->setTrackModel(trkParam.trackmodel);
  }

  if (printConfig) {
    LOG(info) << "Configurable tracker parameters:";
    switch (mTrackFitter->getTrackModel()) {
      case o2::mft::MFTTrackModel::Helix:
        LOG(info) << "Fwd track model     = Helix";
        break;
      case o2::mft::MFTTrackModel::Quadratic:
        LOG(info) << "Fwd track model     = Quadratic";
        break;
      case o2::mft::MFTTrackModel::Linear:
        LOG(info) << "Fwd track model     = Linear";
        break;
      case o2::mft::MFTTrackModel::Optimized:
        LOG(info) << "Fwd track model     = Optimized";
        break;
    }
    LOG(info) << "alignResidual       = " << trkParam.alignResidual;
    LOG(info) << "MinTrackPointsLTF   = " << mMinTrackPointsLTF;
    LOG(info) << "MinTrackPointsCA    = " << mMinTrackPointsCA;
    LOG(info) << "MinTrackStationsLTF = " << mMinTrackStationsLTF;
    LOG(info) << "MinTrackStationsCA  = " << mMinTrackStationsCA;
    LOG(info) << "LTFConeRadius       = " << (trkParam.LTFConeRadius ? "true" : "false");
    LOG(info) << "CAConeRadius        = " << (trkParam.CAConeRadius ? "true" : "false");
    LOG(info) << "LTFclsRCut          = " << mLTFclsRCut;
    LOG(info) << "ROADclsRCut         = " << mROADclsRCut;
    LOG(info) << "RBins               = " << mRBins;
    LOG(info) << "PhiBins             = " << mPhiBins;
    LOG(info) << "LTFseed2BinWin      = " << mLTFseed2BinWin;
    LOG(info) << "LTFinterBinWin      = " << mLTFinterBinWin;
    LOG(info) << "FullClusterScan     = " << (trkParam.FullClusterScan ? "true" : "false");
    LOG(info) << "forceZeroField      = " << (trkParam.forceZeroField ? "true" : "false");
    LOG(info) << "MFTRadLength        = " << trkParam.MFTRadLength;
    LOG(info) << "irFramesOnly        = " << (trkParam.irFramesOnly ? "true" : "false");
    LOG(info) << "isMultCutRequested  = " << (trkParam.isMultCutRequested() ? "true" : "false");
    if (trkParam.isMultCutRequested()) {
      LOG(info) << "cutMultClusLow      = " << trkParam.cutMultClusLow;
      LOG(info) << "cutMultClusHigh     = " << trkParam.cutMultClusHigh;
    }
  }
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::initializeFinder()
{
  mRoad.initialize();

  if (mFullClusterScan) {
    return;
  }

  /// calculate Look-Up-Table of the R-Phi bins projection from one layer to another
  /// layer1 + global R-Phi bin index ---> layer2 + R bin index + Phi bin index

  Float_t dz, x, y, r, phi, x_proj, y_proj, r_proj, phi_proj;
  Int_t binIndex1, binIndex2, binIndex2S, binR_proj, binPhi_proj;

  for (Int_t layer1 = 0; layer1 < (constants::mft::LayersNumber - 1); ++layer1) {

    for (Int_t iRBin = 0; iRBin < mRBins; ++iRBin) {

      r = (iRBin + 0.5) * mRBinSize + constants::index_table::RMin;

      for (Int_t iPhiBin = 0; iPhiBin < mPhiBins; ++iPhiBin) {

        phi = (iPhiBin + 0.5) * mPhiBinSize + constants::index_table::PhiMin;

        binIndex1 = getBinIndex(iRBin, iPhiBin);

        x = r * TMath::Cos(phi);
        y = r * TMath::Sin(phi);

        for (Int_t layer2 = (layer1 + 1); layer2 < constants::mft::LayersNumber; ++layer2) {

          dz = constants::mft::LayerZCoordinate()[layer2] - constants::mft::LayerZCoordinate()[layer1];
          x_proj = x + dz * x * constants::mft::InverseLayerZCoordinate()[layer1];
          y_proj = y + dz * y * constants::mft::InverseLayerZCoordinate()[layer1];
          auto clsPoint2D = math_utils::Point2D<Float_t>(x_proj, y_proj);
          r_proj = clsPoint2D.R();
          phi_proj = clsPoint2D.Phi();
          o2::math_utils::bringTo02PiGen(phi_proj);

          binR_proj = getRBinIndex(r_proj);
          binPhi_proj = getPhiBinIndex(phi_proj);

          int binRS, binPhiS;

          int binwRS = mLTFseed2BinWin;
          int binhwRS = binwRS / 2;

          int binwPhiS = mLTFseed2BinWin;
          int binhwPhiS = binwPhiS / 2;

          for (Int_t iR = 0; iR < binwRS; ++iR) {
            binRS = binR_proj + (iR - binhwRS);
            if (binRS < 0) {
              continue;
            }

            for (Int_t iPhi = 0; iPhi < binwPhiS; ++iPhi) {
              binPhiS = binPhi_proj + (iPhi - binhwPhiS);
              if (binPhiS < 0) {
                continue;
              }

              binIndex2S = getBinIndex(binRS, binPhiS);
              mBinsS[layer1][layer2 - 1][binIndex1].emplace_back(binIndex2S);
            }
          }

          int binR, binPhi;

          int binwR = mLTFinterBinWin;
          int binhwR = binwR / 2;

          int binwPhi = mLTFinterBinWin;
          int binhwPhi = binwPhi / 2;

          for (Int_t iR = 0; iR < binwR; ++iR) {
            binR = binR_proj + (iR - binhwR);
            if (binR < 0) {
              continue;
            }

            for (Int_t iPhi = 0; iPhi < binwPhi; ++iPhi) {
              binPhi = binPhi_proj + (iPhi - binhwPhi);
              if (binPhi < 0) {
                continue;
              }

              binIndex2 = getBinIndex(binR, binPhi);
              mBins[layer1][layer2 - 1][binIndex1].emplace_back(binIndex2);
            }
          }

        } // end loop layer2
      }   // end loop PhiBinIndex
    }     // end loop RBinIndex
  }       // end loop layer1
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::findLTFTracks(ROframe<T>& event)
{
  if (!mFullClusterScan) {
    findTracksLTF(event);
  } else {
    findTracksLTFfcs(event);
  }
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::findCATracks(ROframe<T>& event)
{
  if (!mFullClusterScan) {
    findTracksCA(event);
  } else {
    findTracksCAfcs(event);
  }
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::findTracksLTF(ROframe<T>& event)
{
  // find (high momentum) tracks by the Linear Track Finder (LTF) method

  MCCompLabel mcCompLabel;
  Int_t layer1, layer2, nPointDisks;
  Int_t binR_proj, binPhi_proj, bin;
  Int_t binIndex, clsMinIndex, clsMaxIndex, clsMinIndexS, clsMaxIndexS;
  Int_t extClsIndex;
  Float_t dz = 0., dRCone = 1.;
  Float_t dR2, dR2min, dR2cut = mLTFclsR2Cut;
  Bool_t hasDisk[constants::mft::DisksNumber], newPoint;

  Int_t clsInLayer1, clsInLayer2, clsInLayer;

  Int_t nPoints;
  TrackElement trackPoints[constants::mft::LayersNumber];

  Int_t step = 0;
  layer1 = 0;

  while (true) {

    layer2 = (step == 0) ? (constants::mft::LayersNumber - 1) : (layer2 - 1);
    step++;

    if (layer2 < layer1 + (mMinTrackPointsLTF - 1)) {
      ++layer1;
      if (layer1 > (constants::mft::LayersNumber - (mMinTrackPointsLTF - 1))) {
        break;
      }
      step = 0;
      continue;
    }

    for (std::vector<Cluster>::iterator it1 = event.getClustersInLayer(layer1).begin(); it1 != event.getClustersInLayer(layer1).end(); ++it1) {
      Cluster& cluster1 = *it1;
      if (cluster1.getUsed()) {
        continue;
      }
      clsInLayer1 = it1 - event.getClustersInLayer(layer1).begin();

      // loop over the bins in the search window
      for (auto& binS : mBinsS[layer1][layer2 - 1][cluster1.indexTableBin]) {

        getBinClusterRange(event, layer2, binS, clsMinIndexS, clsMaxIndexS);

        for (std::vector<Cluster>::iterator it2 = (event.getClustersInLayer(layer2).begin() + clsMinIndexS); it2 != (event.getClustersInLayer(layer2).begin() + clsMaxIndexS + 1); ++it2) {
          Cluster& cluster2 = *it2;
          if (cluster2.getUsed()) {
            continue;
          }
          clsInLayer2 = it2 - event.getClustersInLayer(layer2).begin();

          // start a Track type T
          nPoints = 0;

          // add the first seed point
          trackPoints[nPoints].layer = layer1;
          trackPoints[nPoints].idInLayer = clsInLayer1;
          nPoints++;

          // intermediate layers
          for (Int_t layer = (layer1 + 1); layer <= (layer2 - 1); ++layer) {

            newPoint = kTRUE;

            // check if road is a cylinder or a cone
            dz = constants::mft::LayerZCoordinate()[layer2] - constants::mft::LayerZCoordinate()[layer1];
            dRCone = 1 + dz * constants::mft::InverseLayerZCoordinate()[layer1];

            // loop over the bins in the search window
            dR2min = mLTFConeRadius ? dR2cut * dRCone * dRCone : dR2cut;
            for (auto& bin : mBins[layer1][layer - 1][cluster1.indexTableBin]) {

              getBinClusterRange(event, layer, bin, clsMinIndex, clsMaxIndex);

              for (std::vector<Cluster>::iterator it = (event.getClustersInLayer(layer).begin() + clsMinIndex); it != (event.getClustersInLayer(layer).begin() + clsMaxIndex + 1); ++it) {
                Cluster& cluster = *it;
                if (cluster.getUsed()) {
                  continue;
                }
                clsInLayer = it - event.getClustersInLayer(layer).begin();

                dR2 = getDistanceToSeed(cluster1, cluster2, cluster);
                // retain the closest point within a radius dR2cut
                if (dR2 >= dR2min) {
                  continue;
                }
                dR2min = dR2;

                if (newPoint) {
                  trackPoints[nPoints].layer = layer;
                  trackPoints[nPoints].idInLayer = clsInLayer;
                  nPoints++;
                }
                // retain only the closest point in DistanceToSeed
                newPoint = false;
              } // end clusters bin intermediate layer
            }   // end intermediate layers
          }     // end binRPhi

          // add the second seed point
          trackPoints[nPoints].layer = layer2;
          trackPoints[nPoints].idInLayer = clsInLayer2;
          nPoints++;

          // keep only tracks fulfilling the minimum length condition
          if (nPoints < mMinTrackPointsLTF) {
            continue;
          }
          for (Int_t i = 0; i < (constants::mft::DisksNumber); i++) {
            hasDisk[i] = kFALSE;
          }
          for (Int_t point = 0; point < nPoints; ++point) {
            auto layer = trackPoints[point].layer;
            hasDisk[layer / 2] = kTRUE;
          }
          nPointDisks = 0;
          for (Int_t disk = 0; disk < (constants::mft::DisksNumber); ++disk) {
            if (hasDisk[disk]) {
              ++nPointDisks;
            }
          }
          if (nPointDisks < mMinTrackStationsLTF) {
            continue;
          }

          // add a new Track
          event.addTrack();
          for (Int_t point = 0; point < nPoints; ++point) {
            auto layer = trackPoints[point].layer;
            auto clsInLayer = trackPoints[point].idInLayer;
            Cluster& cluster = event.getClustersInLayer(layer)[clsInLayer];
            mcCompLabel = mUseMC ? event.getClusterLabels(layer, cluster.clusterId) : MCCompLabel();
            extClsIndex = event.getClusterExternalIndex(layer, cluster.clusterId);
            event.getCurrentTrack().setPoint(cluster, layer, clsInLayer, mcCompLabel, extClsIndex);
            // mark the used clusters
            cluster.setUsed(true);
          }
        } // end seed clusters bin layer2
      }   // end binRPhi
    }     // end clusters layer1

  } // end seeding
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::findTracksLTFfcs(ROframe<T>& event)
{
  // find (high momentum) tracks by the Linear Track Finder (LTF) method
  // with full scan of the clusters in the target plane

  MCCompLabel mcCompLabel;
  Int_t layer1, layer2, nPointDisks;
  Int_t binR_proj, binPhi_proj, bin;
  Int_t binIndex, clsMinIndex, clsMaxIndex, clsMinIndexS, clsMaxIndexS;
  Int_t extClsIndex;
  Float_t dR2, dR2min, dR2cut = mLTFclsR2Cut;
  Bool_t hasDisk[constants::mft::DisksNumber], newPoint;

  Int_t clsInLayer1, clsInLayer2, clsInLayer;

  Int_t nPoints;
  TrackElement trackPoints[constants::mft::LayersNumber];

  Int_t step = 0;
  layer1 = 0;

  while (true) {

    layer2 = (step == 0) ? (constants::mft::LayersNumber - 1) : (layer2 - 1);
    step++;

    if (layer2 < layer1 + (mMinTrackPointsLTF - 1)) {
      ++layer1;
      if (layer1 > (constants::mft::LayersNumber - (mMinTrackPointsLTF - 1))) {
        break;
      }
      step = 0;
      continue;
    }

    for (std::vector<Cluster>::iterator it1 = event.getClustersInLayer(layer1).begin(); it1 != event.getClustersInLayer(layer1).end(); ++it1) {
      Cluster& cluster1 = *it1;
      if (cluster1.getUsed()) {
        continue;
      }
      clsInLayer1 = it1 - event.getClustersInLayer(layer1).begin();

      for (std::vector<Cluster>::iterator it2 = event.getClustersInLayer(layer2).begin(); it2 != event.getClustersInLayer(layer2).end(); ++it2) {
        Cluster& cluster2 = *it2;
        if (cluster2.getUsed()) {
          continue;
        }
        clsInLayer2 = it2 - event.getClustersInLayer(layer2).begin();

        // start a track type T
        nPoints = 0;

        // add the first seed point
        trackPoints[nPoints].layer = layer1;
        trackPoints[nPoints].idInLayer = clsInLayer1;
        nPoints++;

        // intermediate layers
        for (Int_t layer = (layer1 + 1); layer <= (layer2 - 1); ++layer) {

          newPoint = kTRUE;

          // loop over the bins in the search window
          dR2min = dR2cut;
          for (std::vector<Cluster>::iterator it = event.getClustersInLayer(layer).begin(); it != event.getClustersInLayer(layer).end(); ++it) {
            Cluster& cluster = *it;
            if (cluster.getUsed()) {
              continue;
            }
            clsInLayer = it - event.getClustersInLayer(layer).begin();

            dR2 = getDistanceToSeed(cluster1, cluster2, cluster);
            // retain the closest point within a radius dR2cut
            if (dR2 >= dR2min) {
              continue;
            }
            dR2min = dR2;

            if (newPoint) {
              trackPoints[nPoints].layer = layer;
              trackPoints[nPoints].idInLayer = clsInLayer;
              nPoints++;
            }
            // retain only the closest point in DistanceToSeed
            newPoint = false;
          } // end clusters bin intermediate layer
        }   // end intermediate layers

        // add the second seed point
        trackPoints[nPoints].layer = layer2;
        trackPoints[nPoints].idInLayer = clsInLayer2;
        nPoints++;

        // keep only tracks fulfilling the minimum length condition
        if (nPoints < mMinTrackPointsLTF) {
          continue;
        }
        for (Int_t i = 0; i < (constants::mft::DisksNumber); i++) {
          hasDisk[i] = kFALSE;
        }
        for (Int_t point = 0; point < nPoints; ++point) {
          auto layer = trackPoints[point].layer;
          hasDisk[layer / 2] = kTRUE;
        }
        nPointDisks = 0;
        for (Int_t disk = 0; disk < (constants::mft::DisksNumber); ++disk) {
          if (hasDisk[disk]) {
            ++nPointDisks;
          }
        }
        if (nPointDisks < mMinTrackStationsLTF) {
          continue;
        }

        // add a new Track
        event.addTrack();
        for (Int_t point = 0; point < nPoints; ++point) {
          auto layer = trackPoints[point].layer;
          auto clsInLayer = trackPoints[point].idInLayer;
          Cluster& cluster = event.getClustersInLayer(layer)[clsInLayer];
          mcCompLabel = mUseMC ? event.getClusterLabels(layer, cluster.clusterId) : MCCompLabel();
          extClsIndex = event.getClusterExternalIndex(layer, cluster.clusterId);
          event.getCurrentTrack().setPoint(cluster, layer, clsInLayer, mcCompLabel, extClsIndex);
          // mark the used clusters
          cluster.setUsed(true);
        }
      } // end seed clusters bin layer2
    }   // end clusters layer1

  } // end seeding
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::findTracksCA(ROframe<T>& event)
{
  // layers: 0, 1, 2, ..., 9
  // rules for combining first/last plane in a road:
  // 0 with 6, 7, 8, 9
  // 1 with 6, 7, 8, 9
  // 2 with 8, 9
  // 3 with 8, 9
  Int_t layer1Min = 0, layer1Max = 3;
  Int_t layer2Min[4] = {6, 6, 8, 8};
  constexpr Int_t layer2Max = constants::mft::LayersNumber - 1;

  MCCompLabel mcCompLabel;
  Int_t roadId, nPointDisks;
  Int_t binR_proj, binPhi_proj, bin;
  Int_t binIndex, clsMinIndex, clsMaxIndex, clsMinIndexS, clsMaxIndexS;
  Float_t dz = 0., dRCone = 1.;
  Float_t dR2, dR2min, dR2cut = mROADclsR2Cut;
  Bool_t hasDisk[constants::mft::DisksNumber];

  Int_t clsInLayer1, clsInLayer2, clsInLayer;

  Int_t nPoints;
  std::vector<TrackElement> roadPoints;

  roadId = 0;

  for (Int_t layer1 = layer1Min; layer1 <= layer1Max; ++layer1) {

    for (Int_t layer2 = layer2Max; layer2 >= layer2Min[layer1]; --layer2) {

      for (std::vector<Cluster>::iterator it1 = event.getClustersInLayer(layer1).begin(); it1 != event.getClustersInLayer(layer1).end(); ++it1) {
        Cluster& cluster1 = *it1;
        if (cluster1.getUsed()) {
          continue;
        }
        clsInLayer1 = it1 - event.getClustersInLayer(layer1).begin();

        // loop over the bins in the search window
        for (auto& binS : mBinsS[layer1][layer2 - 1][cluster1.indexTableBin]) {

          getBinClusterRange(event, layer2, binS, clsMinIndexS, clsMaxIndexS);

          for (std::vector<Cluster>::iterator it2 = (event.getClustersInLayer(layer2).begin() + clsMinIndexS); it2 != (event.getClustersInLayer(layer2).begin() + clsMaxIndexS + 1); ++it2) {
            Cluster& cluster2 = *it2;
            if (cluster2.getUsed()) {
              continue;
            }
            clsInLayer2 = it2 - event.getClustersInLayer(layer2).begin();

            // start a road
            roadPoints.clear();

            // add the first seed point
            roadPoints.emplace_back(layer1, clsInLayer1);

            for (Int_t layer = (layer1 + 1); layer <= (layer2 - 1); ++layer) {

              // check if road is a cylinder or a cone
              dz = constants::mft::LayerZCoordinate()[layer2] - constants::mft::LayerZCoordinate()[layer1];
              dRCone = 1 + dz * constants::mft::InverseLayerZCoordinate()[layer1];

              dR2min = mLTFConeRadius ? dR2cut * dRCone * dRCone : dR2cut;

              // loop over the bins in the search window
              for (auto& bin : mBins[layer1][layer - 1][cluster1.indexTableBin]) {

                getBinClusterRange(event, layer, bin, clsMinIndex, clsMaxIndex);

                for (std::vector<Cluster>::iterator it = (event.getClustersInLayer(layer).begin() + clsMinIndex); it != (event.getClustersInLayer(layer).begin() + clsMaxIndex + 1); ++it) {
                  Cluster& cluster = *it;
                  if (cluster.getUsed()) {
                    continue;
                  }
                  clsInLayer = it - event.getClustersInLayer(layer).begin();

                  dR2 = getDistanceToSeed(cluster1, cluster2, cluster);
                  // add all points within a radius dR2cut
                  if (dR2 >= dR2min) {
                    continue;
                  }

                  roadPoints.emplace_back(layer, clsInLayer);

                } // end clusters bin intermediate layer
              }   // end intermediate layers
            }     // end binR

            // add the second seed point
            roadPoints.emplace_back(layer2, clsInLayer2);
            nPoints = roadPoints.size();

            // keep only roads fulfilling the minimum length condition
            if (nPoints < mMinTrackPointsCA) {
              continue;
            }
            for (Int_t i = 0; i < (constants::mft::DisksNumber); i++) {
              hasDisk[i] = kFALSE;
            }
            for (Int_t point = 0; point < nPoints; ++point) {
              auto layer = roadPoints[point].layer;
              hasDisk[layer / 2] = kTRUE;
            }
            nPointDisks = 0;
            for (Int_t disk = 0; disk < (constants::mft::DisksNumber); ++disk) {
              if (hasDisk[disk]) {
                ++nPointDisks;
              }
            }
            if (nPointDisks < mMinTrackStationsCA) {
              continue;
            }

            mRoad.reset();
            for (Int_t point = 0; point < nPoints; ++point) {
              auto layer = roadPoints[point].layer;
              auto clsInLayer = roadPoints[point].idInLayer;
              mRoad.setPoint(layer, clsInLayer);
            }
            mRoad.setRoadId(roadId);
            ++roadId;

            computeCellsInRoad(event);
            runForwardInRoad();
            runBackwardInRoad(event);

          } // end clusters in layer2
        }   // end binRPhi
      }     // end clusters in layer1
    }       // end layer2
  }         // end layer1
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::findTracksCAfcs(ROframe<T>& event)
{
  // layers: 0, 1, 2, ..., 9
  // rules for combining first/last plane in a road:
  // 0 with 6, 7, 8, 9
  // 1 with 6, 7, 8, 9
  // 2 with 8, 9
  // 3 with 8, 9
  Int_t layer1Min = 0, layer1Max = 3;
  Int_t layer2Min[4] = {6, 6, 8, 8};
  constexpr Int_t layer2Max = constants::mft::LayersNumber - 1;

  MCCompLabel mcCompLabel;
  Int_t roadId, nPointDisks;
  Int_t binR_proj, binPhi_proj, bin;
  Int_t binIndex, clsMinIndex, clsMaxIndex, clsMinIndexS, clsMaxIndexS;
  Float_t dR2, dR2cut = mROADclsR2Cut;
  Bool_t hasDisk[constants::mft::DisksNumber];

  Int_t clsInLayer1, clsInLayer2, clsInLayer;

  Int_t nPoints;
  std::vector<TrackElement> roadPoints;

  roadId = 0;

  for (Int_t layer1 = layer1Min; layer1 <= layer1Max; ++layer1) {

    for (Int_t layer2 = layer2Max; layer2 >= layer2Min[layer1]; --layer2) {

      for (std::vector<Cluster>::iterator it1 = event.getClustersInLayer(layer1).begin(); it1 != event.getClustersInLayer(layer1).end(); ++it1) {
        Cluster& cluster1 = *it1;
        if (cluster1.getUsed()) {
          continue;
        }
        clsInLayer1 = it1 - event.getClustersInLayer(layer1).begin();

        for (std::vector<Cluster>::iterator it2 = event.getClustersInLayer(layer2).begin(); it2 != event.getClustersInLayer(layer2).end(); ++it2) {
          Cluster& cluster2 = *it2;
          if (cluster2.getUsed()) {
            continue;
          }
          clsInLayer2 = it2 - event.getClustersInLayer(layer2).begin();

          // start a road
          roadPoints.clear();

          // add the first seed point
          roadPoints.emplace_back(layer1, clsInLayer1);

          for (Int_t layer = (layer1 + 1); layer <= (layer2 - 1); ++layer) {

            for (std::vector<Cluster>::iterator it = event.getClustersInLayer(layer).begin(); it != event.getClustersInLayer(layer).end(); ++it) {
              Cluster& cluster = *it;
              if (cluster.getUsed()) {
                continue;
              }
              clsInLayer = it - event.getClustersInLayer(layer).begin();

              dR2 = getDistanceToSeed(cluster1, cluster2, cluster);
              // add all points within a radius dR2cut
              if (dR2 >= dR2cut) {
                continue;
              }

              roadPoints.emplace_back(layer, clsInLayer);

            } // end clusters bin intermediate layer
          }   // end intermediate layers

          // add the second seed point
          roadPoints.emplace_back(layer2, clsInLayer2);
          nPoints = roadPoints.size();

          // keep only roads fulfilling the minimum length condition
          if (nPoints < mMinTrackPointsCA) {
            continue;
          }
          for (Int_t i = 0; i < (constants::mft::DisksNumber); i++) {
            hasDisk[i] = kFALSE;
          }
          for (Int_t point = 0; point < nPoints; ++point) {
            auto layer = roadPoints[point].layer;
            hasDisk[layer / 2] = kTRUE;
          }
          nPointDisks = 0;
          for (Int_t disk = 0; disk < (constants::mft::DisksNumber); ++disk) {
            if (hasDisk[disk]) {
              ++nPointDisks;
            }
          }
          if (nPointDisks < mMinTrackStationsCA) {
            continue;
          }

          mRoad.reset();
          for (Int_t point = 0; point < nPoints; ++point) {
            auto layer = roadPoints[point].layer;
            auto clsInLayer = roadPoints[point].idInLayer;
            mRoad.setPoint(layer, clsInLayer);
          }
          mRoad.setRoadId(roadId);
          ++roadId;

          computeCellsInRoad(event);
          runForwardInRoad();
          runBackwardInRoad(event);

        } // end clusters in layer2
      }   // end clusters in layer1
    }     // end layer2
  }       // end layer1
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::computeCellsInRoad(ROframe<T>& event)
{
  Int_t layer1, layer1min, layer1max, layer2, layer2min, layer2max;
  Int_t nPtsInLayer1, nPtsInLayer2;
  Int_t clsInLayer1, clsInLayer2;
  Int_t cellId;
  Bool_t noCell;

  mRoad.getLength(layer1min, layer1max);
  --layer1max;

  for (layer1 = layer1min; layer1 <= layer1max; ++layer1) {

    cellId = 0;

    layer2min = layer1 + 1;
    layer2max = std::min(layer1 + (constants::mft::DisksNumber - isDiskFace(layer1)), constants::mft::LayersNumber - 1);

    nPtsInLayer1 = mRoad.getNPointsInLayer(layer1);

    for (Int_t point1 = 0; point1 < nPtsInLayer1; ++point1) {

      clsInLayer1 = mRoad.getClustersIdInLayer(layer1)[point1];

      layer2 = layer2min;

      noCell = kTRUE;
      while (noCell && (layer2 <= layer2max)) {

        nPtsInLayer2 = mRoad.getNPointsInLayer(layer2);
        /*
        if (nPtsInLayer2 > 1) {
          LOG(info) << "BV===== more than one point in road " << mRoad.getRoadId() << " in layer " << layer2 << " : " << nPtsInLayer2 << "\n";
        }
  */
        for (Int_t point2 = 0; point2 < nPtsInLayer2; ++point2) {

          clsInLayer2 = mRoad.getClustersIdInLayer(layer2)[point2];

          noCell = kFALSE;
          // create a cell
          addCellToCurrentRoad(event, layer1, layer2, clsInLayer1, clsInLayer2, cellId);
        } // end points in layer2
        ++layer2;

      } // end while(noCell && (layer2 <= layer2max))
    }   // end points in layer1
  }     // end layer1
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::runForwardInRoad()
{
  Int_t layerR, layerL, icellR, icellL;
  Int_t iter = 0;
  Bool_t levelChange = kTRUE;

  while (levelChange) {

    levelChange = kFALSE;
    ++iter;

    // R = right, L = left
    for (layerL = 0; layerL < (constants::mft::LayersNumber - 2); ++layerL) {

      for (icellL = 0; icellL < mRoad.getCellsInLayer(layerL).size(); ++icellL) {

        Cell& cellL = mRoad.getCellsInLayer(layerL)[icellL];

        layerR = cellL.getSecondLayerId();

        if (layerR == (constants::mft::LayersNumber - 1)) {
          continue;
        }

        for (icellR = 0; icellR < mRoad.getCellsInLayer(layerR).size(); ++icellR) {

          Cell& cellR = mRoad.getCellsInLayer(layerR)[icellR];

          if ((cellL.getLevel() == cellR.getLevel()) && getCellsConnect(cellL, cellR)) {
            if (iter == 1) {
              mRoad.addRightNeighbourToCell(layerL, icellL, layerR, icellR);
              mRoad.addLeftNeighbourToCell(layerR, icellR, layerL, icellL);
            }
            mRoad.incrementCellLevel(layerR, icellR);
            levelChange = kTRUE;

          } // end matching cells
        }   // end loop cellR
      }     // end loop cellL
    }       // end loop layer

    updateCellStatusInRoad();

  } // end while (levelChange)
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::runBackwardInRoad(ROframe<T>& event)
{
  if (mMaxCellLevel == 1) {
    return; // we have only isolated cells
  }

  Bool_t addCellToNewTrack, hasDisk[constants::mft::DisksNumber];

  Int_t lastCellLayer, lastCellId, icell;
  Int_t cellId, layerC, cellIdC, layerRC, cellIdRC, layerL, cellIdL;
  Int_t nPointDisks;
  Float_t deviationPrev, deviation;

  Int_t minLayer = 6;
  Int_t maxLayer = 8;

  Int_t nCells;
  TrackElement trackCells[constants::mft::LayersNumber - 1];

  for (Int_t layer = maxLayer; layer >= minLayer; --layer) {

    for (cellId = 0; cellId < mRoad.getCellsInLayer(layer).size(); ++cellId) {

      if (mRoad.isCellUsed(layer, cellId) || (mRoad.getCellLevel(layer, cellId) < (mMinTrackPointsCA - 1))) {
        continue;
      }

      // start a TrackCA
      nCells = 0;

      trackCells[nCells].layer = layer;
      trackCells[nCells].idInLayer = cellId;
      nCells++;

      // add cells to the new track
      addCellToNewTrack = kTRUE;
      while (addCellToNewTrack) {

        layerRC = trackCells[nCells - 1].layer;
        cellIdRC = trackCells[nCells - 1].idInLayer;

        const Cell& cellRC = mRoad.getCellsInLayer(layerRC)[cellIdRC];

        addCellToNewTrack = kFALSE;

        // loop over left neighbours
        deviationPrev = o2::constants::math::TwoPI;

        for (Int_t iLN = 0; iLN < cellRC.getNLeftNeighbours(); ++iLN) {

          const auto& leftNeighbour = cellRC.getLeftNeighbours()[iLN];
          layerL = leftNeighbour.first;
          cellIdL = leftNeighbour.second;

          const Cell& cellL = mRoad.getCellsInLayer(layerL)[cellIdL];

          if (mRoad.isCellUsed(layerL, cellIdL) || (mRoad.getCellLevel(layerL, cellIdL) != (mRoad.getCellLevel(layerRC, cellIdRC) - 1))) {
            continue;
          }

          deviation = getCellDeviation(cellL, cellRC);

          if (deviation < deviationPrev) {

            deviationPrev = deviation;

            if (iLN > 0) {
              // delete the last added cell
              nCells--;
            }

            trackCells[nCells].layer = layerL;
            trackCells[nCells].idInLayer = cellIdL;
            nCells++;

            addCellToNewTrack = kTRUE;
          }

        } // end loop left neighbour
      }   // end  while(addCellToNewTrack)

      // check the track length
      for (Int_t i = 0; i < (constants::mft::DisksNumber); ++i) {
        hasDisk[i] = kFALSE;
      }

      layerC = trackCells[0].layer;
      cellIdC = trackCells[0].idInLayer;
      const Cell& cellC = mRoad.getCellsInLayer(layerC)[cellIdC];
      hasDisk[cellC.getSecondLayerId() / 2] = kTRUE;
      for (icell = 0; icell < nCells; ++icell) {
        layerC = trackCells[icell].layer;
        cellIdC = trackCells[icell].idInLayer;
        hasDisk[layerC / 2] = kTRUE;
      }

      nPointDisks = 0;
      for (Int_t disk = 0; disk < (constants::mft::DisksNumber); ++disk) {
        if (hasDisk[disk]) {
          ++nPointDisks;
        }
      }

      if (nPointDisks < mMinTrackStationsCA) {
        continue;
      }

      // add a new Track setting isCA = true
      event.addTrack(true);
      for (icell = 0; icell < nCells; ++icell) {
        layerC = trackCells[icell].layer;
        cellIdC = trackCells[icell].idInLayer;
        addCellToCurrentTrackCA(layerC, cellIdC, event);
        mRoad.setCellUsed(layerC, cellIdC, kTRUE);
        // marked the used clusters
        const Cell& cellC = mRoad.getCellsInLayer(layerC)[cellIdC];
        event.getClustersInLayer(cellC.getFirstLayerId())[cellC.getFirstClusterIndex()].setUsed(true);
        event.getClustersInLayer(cellC.getSecondLayerId())[cellC.getSecondClusterIndex()].setUsed(true);
      }
      event.getCurrentTrack().sort();
    } // end loop cells
  }   // end loop start layer
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::updateCellStatusInRoad()
{
  Int_t layerMin, layerMax;
  mRoad.getLength(layerMin, layerMax);
  for (Int_t layer = layerMin; layer < layerMax; ++layer) {
    for (Int_t icell = 0; icell < mRoad.getCellsInLayer(layer).size(); ++icell) {
      mRoad.updateCellLevel(layer, icell);
      mMaxCellLevel = std::max(mMaxCellLevel, mRoad.getCellLevel(layer, icell));
    }
  }
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::addCellToCurrentRoad(ROframe<T>& event, const Int_t layer1, const Int_t layer2, const Int_t clsInLayer1, const Int_t clsInLayer2, Int_t& cellId)
{
  Cell& cell = mRoad.addCellInLayer(layer1, layer2, clsInLayer1, clsInLayer2, cellId);

  Cluster& cluster1 = event.getClustersInLayer(layer1)[clsInLayer1];
  Cluster& cluster2 = event.getClustersInLayer(layer2)[clsInLayer2];

  Float_t coord[6];
  coord[0] = cluster1.getX();
  coord[1] = cluster1.getY();
  coord[2] = cluster1.getZ();
  coord[3] = cluster2.getX();
  coord[4] = cluster2.getY();
  coord[5] = cluster2.getZ();

  cell.setCoordinates(coord);
  cellId++;
}

//_________________________________________________________________________________________________
template <typename T>
void Tracker<T>::addCellToCurrentTrackCA(const Int_t layer1, const Int_t cellId, ROframe<T>& event)
{
  auto& trackCA = event.getCurrentTrack();
  const Cell& cell = mRoad.getCellsInLayer(layer1)[cellId];
  const Int_t layer2 = cell.getSecondLayerId();
  const Int_t clsInLayer1 = cell.getFirstClusterIndex();
  const Int_t clsInLayer2 = cell.getSecondClusterIndex();

  Cluster& cluster1 = event.getClustersInLayer(layer1)[clsInLayer1];
  Cluster& cluster2 = event.getClustersInLayer(layer2)[clsInLayer2];

  MCCompLabel mcCompLabel1 = mUseMC ? event.getClusterLabels(layer1, cluster1.clusterId) : MCCompLabel();
  MCCompLabel mcCompLabel2 = mUseMC ? event.getClusterLabels(layer2, cluster2.clusterId) : MCCompLabel();

  Int_t extClsIndex;

  if (trackCA.getNumberOfPoints() == 0) {
    extClsIndex = event.getClusterExternalIndex(layer2, cluster2.clusterId);
    trackCA.setPoint(cluster2, layer2, clsInLayer2, mcCompLabel2, extClsIndex);
  }

  extClsIndex = event.getClusterExternalIndex(layer1, cluster1.clusterId);
  trackCA.setPoint(cluster1, layer1, clsInLayer1, mcCompLabel1, extClsIndex);
}

//_________________________________________________________________________________________________
template <typename T>
bool Tracker<T>::fitTracks(ROframe<T>& event)
{
  for (auto& track : event.getTracks()) {
    T outParam = track;
    mTrackFitter->initTrack(track);
    mTrackFitter->fit(track);
    mTrackFitter->initTrack(outParam, true);
    mTrackFitter->fit(outParam, true);
    track.setOutParam(outParam);
  }
  return true;
}

template class Tracker<o2::mft::TrackLTF>;
template class Tracker<o2::mft::TrackLTFL>;

} // namespace mft
} // namespace o2
