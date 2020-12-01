// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
Tracker::Tracker(bool useMC) : mUseMC{useMC}
{
  mTrackFitter = std::make_unique<o2::mft::TrackFitter>();
}

//_________________________________________________________________________________________________
void Tracker::setBz(Float_t bz)
{
  /// Configure track propagation
  mBz = bz;
  mTrackFitter->setBz(bz);
}

//_________________________________________________________________________________________________
void Tracker::initialize()
{
  /// calculate Look-Up-Table of the R-Phi bins projection from one layer to another
  /// layer1 + global R-Phi bin index ---> layer2 + R bin index + Phi bin index

  Float_t dz, x, y, r, phi, x_proj, y_proj, r_proj, phi_proj;
  Int_t binIndex1, binIndex2, binIndex2S, binR_proj, binPhi_proj;

  for (Int_t layer1 = 0; layer1 < (constants::mft::LayersNumber - 1); ++layer1) {

    for (Int_t iRBin = 0; iRBin < mTrackerConfig.get()->mRBins; ++iRBin) {

      r = (iRBin + 0.5) * mTrackerConfig.get()->mRBinSize + constants::index_table::RMin;

      for (Int_t iPhiBin = 0; iPhiBin < mTrackerConfig.get()->mPhiBins; ++iPhiBin) {

        phi = (iPhiBin + 0.5) * mTrackerConfig.get()->mPhiBinSize + constants::index_table::PhiMin;

        binIndex1 = mTrackerConfig.get()->getBinIndex(iRBin, iPhiBin);

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

          binR_proj = mTrackerConfig.get()->getRBinIndex(r_proj);
          binPhi_proj = mTrackerConfig.get()->getPhiBinIndex(phi_proj);

          int binRS, binPhiS;

          int binwRS = mTrackerConfig.get()->mLTFseed2BinWin;
          if (abs(dz) < 1.5) {
            binwRS = 3;
          } else if (abs(dz) < 5.0) {
            binwRS = 3;
          } else if (abs(dz) < 10.0) {
            binwRS = 3;
          } else if (abs(dz) < 15.0) {
            binwRS = 3;
          } else if (abs(dz) < 20.0) {
            binwRS = 3;
          } else if (abs(dz) < 25.0) {
            binwRS = 3;
          } else {
            binwRS = 3;
          }
          int binhwRS = binwRS / 2;

          int binwPhiS = mTrackerConfig.get()->mLTFseed2BinWin;
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

              binIndex2S = mTrackerConfig.get()->getBinIndex(binRS, binPhiS);
              mBinsS[layer1][layer2 - 1][binIndex1].emplace_back(binIndex2S);
            }
          }

          int binR, binPhi;

          int binwR = mTrackerConfig.get()->mLTFinterBinWin;
          int binhwR = binwR / 2;

          int binwPhi = mTrackerConfig.get()->mLTFinterBinWin;
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

              binIndex2 = mTrackerConfig.get()->getBinIndex(binR, binPhi);
              mBins[layer1][layer2 - 1][binIndex1].emplace_back(binIndex2);
            }
          }

        } // end loop layer2
      }   // end loop PhiBinIndex
    }     // end loop RBinIndex
  }       // end loop layer1
}

//_________________________________________________________________________________________________
void Tracker::clustersToTracks(ROframe& event, std::ostream& timeBenchmarkOutputStream)
{
  mTracks.clear();
  mTrackLabels.clear();
  findTracks(event);
  fitTracks(event);
}

//_________________________________________________________________________________________________
void Tracker::findTracks(ROframe& event)
{
  //computeCells(event);
  findTracksLTF(event);
  findTracksCA(event);
}

//_________________________________________________________________________________________________
void Tracker::findTracksLTF(ROframe& event)
{
  // find (high momentum) tracks by the Linear Track Finder (LTF) method

  MCCompLabel mcCompLabel;
  Int_t layer1, layer2, nPointDisks;
  Int_t binR_proj, binPhi_proj, bin;
  Int_t binIndex, clsMinIndex, clsMaxIndex, clsMinIndexS, clsMaxIndexS;
  Int_t extClsIndex;
  Float_t dR, dRmin, dRcut = mTrackerConfig.get()->mLTFclsRCut;
  Bool_t hasDisk[constants::mft::DisksNumber], newPoint, seed;

  Int_t clsInLayer1, clsInLayer2, clsInLayer;

  Int_t nPoints;
  TrackElement trackPoints[constants::mft::LayersNumber];

  Int_t step = 0;
  seed = kTRUE;
  layer1 = 0;

  while (seed) {

    layer2 = (step == 0) ? (constants::mft::LayersNumber - 1) : (layer2 - 1);
    step++;

    if (layer2 < layer1 + (mTrackerConfig.get()->mMinTrackPointsLTF - 1)) {
      ++layer1;
      if (layer1 > (constants::mft::LayersNumber - (mTrackerConfig.get()->mMinTrackPointsLTF - 1))) {
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

          // start a TrackLTF
          nPoints = 0;

          // add the first seed point
          trackPoints[nPoints].layer = layer1;
          trackPoints[nPoints].idInLayer = clsInLayer1;
          nPoints++;

          // intermediate layers
          for (Int_t layer = (layer1 + 1); layer <= (layer2 - 1); ++layer) {

            newPoint = kTRUE;

            // loop over the bins in the search window
            dRmin = dRcut;
            for (auto& bin : mBins[layer1][layer - 1][cluster1.indexTableBin]) {

              getBinClusterRange(event, layer, bin, clsMinIndex, clsMaxIndex);

              for (std::vector<Cluster>::iterator it = (event.getClustersInLayer(layer).begin() + clsMinIndex); it != (event.getClustersInLayer(layer).begin() + clsMaxIndex + 1); ++it) {
                Cluster& cluster = *it;
                if (cluster.getUsed()) {
                  continue;
                }
                clsInLayer = it - event.getClustersInLayer(layer).begin();

                dR = getDistanceToSeed(cluster1, cluster2, cluster);
                // retain the closest point within a radius dRcut
                if (dR >= dRmin) {
                  continue;
                }
                dRmin = dR;

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
          if (nPoints < mTrackerConfig.get()->mMinTrackPointsLTF) {
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
          if (nPointDisks < mTrackerConfig.get()->mMinTrackStationsLTF) {
            continue;
          }

          // add a new TrackLTF
          event.addTrackLTF();
          for (Int_t point = 0; point < nPoints; ++point) {
            auto layer = trackPoints[point].layer;
            auto clsInLayer = trackPoints[point].idInLayer;
            Cluster& cluster = event.getClustersInLayer(layer)[clsInLayer];
            mcCompLabel = mUseMC ? event.getClusterLabels(layer, cluster.clusterId) : MCCompLabel();
            extClsIndex = event.getClusterExternalIndex(layer, cluster.clusterId);
            event.getCurrentTrackLTF().setPoint(cluster, layer, clsInLayer, mcCompLabel, extClsIndex);
            // mark the used clusters
            cluster.setUsed(true);
          }
        } // end seed clusters bin layer2
      }   // end binRPhi
    }     // end clusters layer1

  } // end seeding
}

//_________________________________________________________________________________________________
void Tracker::findTracksCA(ROframe& event)
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
  Float_t dR, dRcut = mTrackerConfig.get()->mROADclsRCut;
  Bool_t hasDisk[constants::mft::DisksNumber];

  Int_t clsInLayer1, clsInLayer2, clsInLayer;

  Int_t nPoints;
  //TrackElement roadPoints[10 * constants::mft::LayersNumber];

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
            //nPoints = 0;
            roadPoints.clear();

            // add the first seed point
            //roadPoints[nPoints].layer = layer1;
            //roadPoints[nPoints].idInLayer = clsInLayer1;
            //nPoints++;
            roadPoints.emplace_back(layer1, clsInLayer1);

            for (Int_t layer = (layer1 + 1); layer <= (layer2 - 1); ++layer) {

              // loop over the bins in the search window
              for (auto& bin : mBins[layer1][layer - 1][cluster1.indexTableBin]) {

                getBinClusterRange(event, layer, bin, clsMinIndex, clsMaxIndex);

                for (std::vector<Cluster>::iterator it = (event.getClustersInLayer(layer).begin() + clsMinIndex); it != (event.getClustersInLayer(layer).begin() + clsMaxIndex + 1); ++it) {
                  Cluster& cluster = *it;
                  if (cluster.getUsed()) {
                    continue;
                  }
                  clsInLayer = it - event.getClustersInLayer(layer).begin();

                  dR = getDistanceToSeed(cluster1, cluster2, cluster);
                  // add all points within a radius dRcut
                  if (dR >= dRcut) {
                    continue;
                  }

                  //roadPoints[nPoints].layer = layer;
                  //roadPoints[nPoints].idInLayer = clsInLayer;
                  //nPoints++;
                  roadPoints.emplace_back(layer, clsInLayer);

                } // end clusters bin intermediate layer
              }   // end intermediate layers
            }     // end binR

            // add the second seed point
            //roadPoints[nPoints].layer = layer2;
            //roadPoints[nPoints].idInLayer = clsInLayer2;
            //nPoints++;
            roadPoints.emplace_back(layer2, clsInLayer2);
            nPoints = roadPoints.size();

            // keep only roads fulfilling the minimum length condition
            if (nPoints < mTrackerConfig.get()->mMinTrackPointsCA) {
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
            if (nPointDisks < mTrackerConfig.get()->mMinTrackStationsCA) {
              continue;
            }

            event.addRoad();
            for (Int_t point = 0; point < nPoints; ++point) {
              auto layer = roadPoints[point].layer;
              auto clsInLayer = roadPoints[point].idInLayer;
              event.getCurrentRoad().setPoint(layer, clsInLayer);
            }
            event.getCurrentRoad().setRoadId(roadId);
            ++roadId;

            computeCellsInRoad(event);
            runForwardInRoad(event);
            runBackwardInRoad(event);

          } // end clusters in layer2
        }   // end binRPhi
      }     // end clusters in layer1
    }       // end layer2
  }         // end layer1
}

//_________________________________________________________________________________________________
void Tracker::computeCellsInRoad(ROframe& event)
{
  Int_t layer1, layer1min, layer1max, layer2, layer2min, layer2max;
  Int_t nPtsInLayer1, nPtsInLayer2;
  Int_t clsInLayer1, clsInLayer2;
  Int_t cellId;
  Bool_t noCell;

  Road& road = event.getCurrentRoad();

  road.getLength(layer1min, layer1max);
  --layer1max;

  for (layer1 = layer1min; layer1 <= layer1max; ++layer1) {

    cellId = 0;

    layer2min = layer1 + 1;
    layer2max = std::min(layer1 + (constants::mft::DisksNumber - isDiskFace(layer1)), constants::mft::LayersNumber - 1);

    nPtsInLayer1 = road.getNPointsInLayer(layer1);

    for (Int_t point1 = 0; point1 < nPtsInLayer1; ++point1) {

      clsInLayer1 = road.getClustersIdInLayer(layer1)[point1];

      layer2 = layer2min;

      noCell = kTRUE;
      while (noCell && (layer2 <= layer2max)) {

        nPtsInLayer2 = road.getNPointsInLayer(layer2);
        /*
        if (nPtsInLayer2 > 1) {
          LOG(INFO) << "BV===== more than one point in road " << road.getRoadId() << " in layer " << layer2 << " : " << nPtsInLayer2 << "\n";
        }
	*/
        for (Int_t point2 = 0; point2 < nPtsInLayer2; ++point2) {

          clsInLayer2 = road.getClustersIdInLayer(layer2)[point2];

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
void Tracker::runForwardInRoad(ROframe& event)
{
  Int_t layerR, layerL, icellR, icellL;
  Int_t iter = 0;
  Bool_t levelChange = kTRUE;

  Road& road = event.getCurrentRoad();

  while (levelChange) {

    levelChange = kFALSE;
    ++iter;

    // R = right, L = left
    for (layerL = 0; layerL < (constants::mft::LayersNumber - 2); ++layerL) {

      for (icellL = 0; icellL < road.getCellsInLayer(layerL).size(); ++icellL) {

        Cell& cellL = road.getCellsInLayer(layerL)[icellL];

        layerR = cellL.getSecondLayerId();

        if (layerR == (constants::mft::LayersNumber - 1)) {
          continue;
        }

        for (icellR = 0; icellR < road.getCellsInLayer(layerR).size(); ++icellR) {

          Cell& cellR = road.getCellsInLayer(layerR)[icellR];

          if ((cellL.getLevel() == cellR.getLevel()) && getCellsConnect(cellL, cellR)) {
            if (iter == 1) {
              road.addRightNeighbourToCell(layerL, icellL, layerR, icellR);
              road.addLeftNeighbourToCell(layerR, icellR, layerL, icellL);
            }
            road.incrementCellLevel(layerR, icellR);
            levelChange = kTRUE;

          } // end matching cells
        }   // end loop cellR
      }     // end loop cellL
    }       // end loop layer

    updateCellStatusInRoad(road);

  } // end while (step)
}

//_________________________________________________________________________________________________
void Tracker::runBackwardInRoad(ROframe& event)
{
  if (mMaxCellLevel == 1) {
    return; // we have only isolated cells
  }

  Bool_t addCellToNewTrack, hasDisk[constants::mft::DisksNumber];

  Int_t lastCellLayer, lastCellId, icell;
  Int_t cellId, layerC, cellIdC, layerRC, cellIdRC, layerL, cellIdL;
  Int_t nPointDisks;
  Float_t deviationPrev, deviation;

  // start layer
  Int_t minLayer = 6;
  Int_t maxLayer = 8;

  Road& road = event.getCurrentRoad();

  Int_t nCells;
  TrackElement trackCells[constants::mft::LayersNumber - 1];
  //std::vector<TrackElement> trackCells;
  //trackCells.reserve(constants::mft::LayersNumber - 1);

  for (Int_t layer = maxLayer; layer >= minLayer; --layer) {

    for (cellId = 0; cellId < road.getCellsInLayer(layer).size(); ++cellId) {

      if (road.isCellUsed(layer, cellId)) {
        continue;
      }
      if (road.getCellLevel(layer, cellId) < (mTrackerConfig.get()->mMinTrackPointsCA - 1)) {
        continue;
      }

      // start a TrackCA
      nCells = 0;
      //trackCells.clear();

      trackCells[nCells].layer = layer;
      trackCells[nCells].idInLayer = cellId;
      nCells++;
      //trackCells.emplace_back(layer, cellId);

      // add cells to the new track
      addCellToNewTrack = kTRUE;
      while (addCellToNewTrack) {

        layerRC = trackCells[nCells - 1].layer;
        cellIdRC = trackCells[nCells - 1].idInLayer;

        const Cell& cellRC = road.getCellsInLayer(layerRC)[cellIdRC];

        addCellToNewTrack = kFALSE;

        // loop over left neighbours
        deviationPrev = o2::constants::math::TwoPI;

        for (Int_t iLN = 0; iLN < cellRC.getNLeftNeighbours(); ++iLN) {

          auto leftNeighbour = cellRC.getLeftNeighbours()[iLN];
          layerL = leftNeighbour.first;
          cellIdL = leftNeighbour.second;

          const Cell& cellL = road.getCellsInLayer(layerL)[cellIdL];

          if (road.isCellUsed(layerL, cellIdL)) {
            continue;
          }
          if (road.getCellLevel(layerL, cellIdL) != (road.getCellLevel(layerRC, cellIdRC) - 1)) {
            continue;
          }

          deviation = getCellDeviation(cellL, cellRC);

          if (deviation < deviationPrev) {

            deviationPrev = deviation;

            if (leftNeighbour != cellRC.getLeftNeighbours().front()) {
              // delete the last added cell
              nCells--;
              //trackCells.pop_back();
            }

            trackCells[nCells].layer = layerL;
            trackCells[nCells].idInLayer = cellIdL;
            nCells++;
            //trackCells.emplace_back(layerL, cellIdL);

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
      const Cell& cellC = event.getCurrentRoad().getCellsInLayer(layerC)[cellIdC];
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

      if (nPointDisks < mTrackerConfig.get()->mMinTrackStationsCA) {
        continue;
      }

      // add a new TrackCA
      event.addTrackCA(road.getRoadId());
      for (icell = 0; icell < nCells; ++icell) {
        layerC = trackCells[icell].layer;
        cellIdC = trackCells[icell].idInLayer;
        addCellToCurrentTrackCA(layerC, cellIdC, event);
        road.setCellUsed(layerC, cellIdC, kTRUE);
        // marked the used clusters
        const Cell& cellC = event.getCurrentRoad().getCellsInLayer(layerC)[cellIdC];
        event.getClustersInLayer(cellC.getFirstLayerId())[cellC.getFirstClusterIndex()].setUsed(true);
        event.getClustersInLayer(cellC.getSecondLayerId())[cellC.getSecondClusterIndex()].setUsed(true);
      }
    } // end loop cells
  }   // end loop start layer
}

//_________________________________________________________________________________________________
void Tracker::updateCellStatusInRoad(Road& road)
{
  for (Int_t layer = 0; layer < (constants::mft::LayersNumber - 1); ++layer) {
    for (Int_t icell = 0; icell < road.getCellsInLayer(layer).size(); ++icell) {
      road.updateCellLevel(layer, icell);
      mMaxCellLevel = std::max(mMaxCellLevel, road.getCellLevel(layer, icell));
    }
  }
}

//_________________________________________________________________________________________________
void Tracker::addCellToCurrentRoad(ROframe& event, const Int_t layer1, const Int_t layer2, const Int_t clsInLayer1, const Int_t clsInLayer2, Int_t& cellId)
{
  Road& road = event.getCurrentRoad();

  Cell& cell = road.addCellInLayer(layer1, layer2, clsInLayer1, clsInLayer2, cellId);

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
void Tracker::addCellToCurrentTrackCA(const Int_t layer1, const Int_t cellId, ROframe& event)
{
  TrackCA& trackCA = event.getCurrentTrackCA();
  Road& road = event.getCurrentRoad();
  const Cell& cell = road.getCellsInLayer(layer1)[cellId];
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
bool Tracker::fitTracks(ROframe& event)
{
  for (auto& track : event.getTracksLTF()) {
    TrackLTF outParam = track;
    mTrackFitter->initTrack(track);
    mTrackFitter->fit(track);
    mTrackFitter->initTrack(outParam, true);
    mTrackFitter->fit(outParam, true);
    track.setOutParam(outParam);
  }
  for (auto& track : event.getTracksCA()) {
    track.sort();
    TrackCA outParam = track;
    mTrackFitter->initTrack(track);
    mTrackFitter->fit(track);
    mTrackFitter->initTrack(outParam, true);
    mTrackFitter->fit(outParam, true);
    track.setOutParam(outParam);
  }

  return true;
}

} // namespace mft
} // namespace o2
