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

Tracker::Tracker(bool useMC) : mUseMC{useMC}
{

  /// Configure track propagation
  LOG(INFO) << "initializing track fitter";
  mTrackFitter = std::make_unique<o2::mft::TrackFitter>();
  mTrackFitter->setBz(mBz);
}

void Tracker::clustersToTracks(ROframe& event, std::ostream& timeBenchmarkOutputStream)
{
  mTracks.clear();
  mTrackLabels.clear();
  findTracks(event);
  fitTracks(event);
}

void Tracker::findTracks(ROframe& event)
{
  //computeCells(event);
  findTracksLTF(event);
  findTracksCA(event);
}

void Tracker::computeCells(ROframe& event)
{
  MCCompLabel mcCompLabel;
  Int_t layer1, layer2;
  Int_t nClsInLayer1, nClsInLayer2;
  Int_t disk1, disk2;
  Int_t binR_proj, binPhi_proj, bin;
  Int_t binIndex, clsMinIndex, clsMaxIndex;
  std::array<Int_t, 3> binsR, binsPhi;
  Int_t cellId = 0;
  for (layer1 = 0; layer1 < (constants::mft::LayersNumber - 1); ++layer1) {
    nClsInLayer1 = event.getClustersInLayer(layer1).size();
    disk1 = layer1 / 2;
    layer2 = layer1 + 1; // only (L+1) tracklets !
    disk2 = layer2 / 2;
    nClsInLayer2 = event.getClustersInLayer(layer2).size();
    for (Int_t clsLayer1 = 0; clsLayer1 < nClsInLayer1; ++clsLayer1) {
      const Cluster& cluster1 = event.getClustersInLayer(layer1).at(clsLayer1);
      // project to next layer and get the bin index in R and Phi
      getRPhiProjectionBin(cluster1, layer1, layer2, binR_proj, binPhi_proj);
      // define the search window in bins x bins (3x3, 5x5, etc.)
      for (Int_t i = 0; i < constants::index_table::LTFinterBinWin; ++i) {
        binsR[i] = binR_proj + (i - constants::index_table::LTFinterBinWin / 2);
        binsPhi[i] = binPhi_proj + (i - constants::index_table::LTFinterBinWin / 2);
      }
      // loop over the bins in the search window
      for (auto binR : binsR) {
        for (auto binPhi : binsPhi) {
          // the global bin index
          bin = constants::index_table::getBinIndex(binR, binPhi);
          if (!getBinClusterRange(event, layer2, bin, clsMinIndex, clsMaxIndex)) {
            continue;
          }
          for (Int_t clsLayer2 = clsMinIndex; clsLayer2 <= clsMaxIndex; ++clsLayer2) {
            event.addCellToLayer(layer1, layer2, clsLayer1, clsLayer2, cellId++);
          } // end clusters bin layer2
        }   // end binPhi
      }     // end binR
    }       // end clusters layer1
  }         // end layers
}

void Tracker::findTracksLTF(ROframe& event)
{
  // find (high momentum) tracks by the Linear Track Finder (LTF) method

  MCCompLabel mcCompLabel;
  Int_t layer1, layer2, nPointDisks;
  Int_t nClsInLayer1, nClsInLayer2, nClsInLayer;
  Int_t binR_proj, binPhi_proj, bin;
  Int_t binIndex, clsMinIndex, clsMaxIndex, clsMinIndexS, clsMaxIndexS;
  Float_t dR, dRmin, dRcut = constants::mft::LTFclsRCut;
  std::vector<Int_t> binsR, binsPhi, binsRS, binsPhiS;
  Bool_t hasDisk[constants::mft::DisksNumber], newPoint, seed = kTRUE;

  binsRS.resize(constants::index_table::LTFseed2BinWin);
  binsPhiS.resize(constants::index_table::LTFseed2BinWin);

  binsR.resize(constants::index_table::LTFinterBinWin);
  binsPhi.resize(constants::index_table::LTFinterBinWin);

  Int_t step = 0;

  layer1 = 0;

  while (seed) {

    if (step == 0) {
      layer2 = constants::mft::LayersNumber - 1;
    } else {
      layer2--;
    }

    step++;

    if (layer2 < layer1 + (constants::mft::MinTrackPoints - 1)) {
      ++layer1;
      if (layer1 > (constants::mft::LayersNumber - (constants::mft::MinTrackPoints - 1))) {
        break;
      }
      step = 0;
      continue;
    }

    nClsInLayer1 = event.getClustersInLayer(layer1).size();
    nClsInLayer2 = event.getClustersInLayer(layer2).size();

    for (Int_t clsLayer1 = 0; clsLayer1 < nClsInLayer1; ++clsLayer1) {
      if (event.isClusterUsed(layer1, clsLayer1)) {
        continue;
      }
      const Cluster& cluster1 = event.getClustersInLayer(layer1)[clsLayer1];

      // project to the second seed layer and get the bin index in R and Phi
      getRPhiProjectionBin(cluster1, layer1, layer2, binR_proj, binPhi_proj);
      // define the search window in bins x bins (3x3, 5x5, etc.)
      for (Int_t i = 0; i < constants::index_table::LTFseed2BinWin; ++i) {
        binsRS[i] = binR_proj + (i - constants::index_table::LTFseed2BinWin / 2);
        binsPhiS[i] = binPhi_proj + (i - constants::index_table::LTFseed2BinWin / 2);
      }
      // loop over the bins in the search window
      for (auto binRS : binsRS) {
        for (auto binPhiS : binsPhiS) {
          // the global bin index
          bin = constants::index_table::getBinIndex(binRS, binPhiS);
          if (!getBinClusterRange(event, layer2, bin, clsMinIndexS, clsMaxIndexS)) {
            continue;
          }
          for (Int_t clsLayer2 = clsMinIndexS; clsLayer2 <= clsMaxIndexS; ++clsLayer2) {
            if (event.isClusterUsed(layer2, clsLayer2)) {
              continue;
            }
            const Cluster& cluster2 = event.getClustersInLayer(layer2)[clsLayer2];

            for (Int_t i = 0; i < (constants::mft::DisksNumber); i++) {
              hasDisk[i] = kFALSE;
            }

            hasDisk[layer1 / 2] = kTRUE;
            hasDisk[layer2 / 2] = kTRUE;

            // start a track LTF
            event.addTrackLTF();

            // add the first seed-point
            mcCompLabel = mUseMC ? event.getClusterLabels(layer1, cluster1.clusterId) : MCCompLabel();
            newPoint = kTRUE;
            event.getCurrentTrackLTF().setPoint(cluster1.getX(), cluster1.getY(), cluster1.getZ(), layer1, clsLayer1, mcCompLabel, newPoint);

            for (Int_t layer = (layer1 + 1); layer <= (layer2 - 1); ++layer) {

              nClsInLayer = event.getClustersInLayer(layer).size();

              newPoint = kTRUE;

              // project to the intermediate layer and get the bin index in R and Phi
              getRPhiProjectionBin(cluster1, layer1, layer, binR_proj, binPhi_proj);

              // define the search window in bins x bins (3x3, 5x5, etc.)
              for (Int_t i = 0; i < constants::index_table::LTFinterBinWin; ++i) {
                binsR[i] = binR_proj + (i - constants::index_table::LTFinterBinWin / 2);
                binsPhi[i] = binPhi_proj + (i - constants::index_table::LTFinterBinWin / 2);
              }
              // loop over the bins in the search window
              dRmin = dRcut;
              for (auto binR : binsR) {
                for (auto binPhi : binsPhi) {
                  // the global bin index
                  bin = constants::index_table::getBinIndex(binR, binPhi);
                  if (!getBinClusterRange(event, layer, bin, clsMinIndex, clsMaxIndex)) {
                    continue;
                  }
                  for (Int_t clsLayer = clsMinIndex; clsLayer <= clsMaxIndex; ++clsLayer) {
                    if (event.isClusterUsed(layer, clsLayer)) {
                      continue;
                    }
                    const Cluster& cluster = event.getClustersInLayer(layer)[clsLayer];

                    dR = getDistanceToSeed(cluster1, cluster2, cluster);
                    // retain the closest point within a radius dRcut
                    if (dR >= dRmin) {
                      continue;
                    }
                    dRmin = dR;

                    hasDisk[layer / 2] = kTRUE;
                    mcCompLabel = mUseMC ? event.getClusterLabels(layer, cluster.clusterId) : MCCompLabel();
                    event.getCurrentTrackLTF().setPoint(cluster.getX(), cluster.getY(), cluster.getZ(), layer, clsLayer, mcCompLabel, newPoint);
                  } // end clusters bin intermediate layer
                }   // end intermediate layers
              }     // end binPhi
            }       // end binR

            // add the second seed-point
            mcCompLabel = mUseMC ? event.getClusterLabels(layer2, cluster2.clusterId) : MCCompLabel();
            newPoint = kTRUE;
            event.getCurrentTrackLTF().setPoint(cluster2.getX(), cluster2.getY(), cluster2.getZ(), layer2, clsLayer2, mcCompLabel, newPoint);

            // keep only tracks fulfilling the minimum length condition
            if (event.getCurrentTrackLTF().getNPoints() < constants::mft::MinTrackPoints) {
              event.removeCurrentTrackLTF();
              continue;
            }
            nPointDisks = 0;
            for (Int_t disk = 0; disk < (constants::mft::DisksNumber); ++disk) {
              if (hasDisk[disk])
                ++nPointDisks;
            }
            if (nPointDisks < constants::mft::MinTrackPoints) {
              event.removeCurrentTrackLTF();
              continue;
            }

            // mark the used clusters
            //Int_t lay, layMin = 10, layMax = -1;
            for (Int_t point = 0; point < event.getCurrentTrackLTF().getNPoints(); ++point) {
              event.markUsedCluster(event.getCurrentTrackLTF().getLayers()[point], event.getCurrentTrackLTF().getClustersId()[point]);
              //lay = event.getCurrentTrackLTF().getLayers()[point];
              //layMin = (lay < layMin) ? lay : layMin;
              //layMax = (lay > layMax) ? lay : layMax;
            }
          } // end seed clusters bin layer2
        }   // end binPhi
      }     // end binR
    }       // end clusters layer1

  } // end seeding
}

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
  Int_t layer2Max[4] = {9, 9, 9, 9};

  MCCompLabel mcCompLabel;
  Int_t roadId, nPointDisks;
  Int_t nClsInLayer1, nClsInLayer2, nClsInLayer;
  Int_t binR_proj, binPhi_proj, bin;
  Int_t binIndex, clsMinIndex, clsMaxIndex, clsMinIndexS, clsMaxIndexS;
  Float_t dR, dRcut = constants::mft::ROADclsRCut;
  std::vector<Int_t> binsR, binsPhi, binsRS, binsPhiS;
  Bool_t hasDisk[constants::mft::DisksNumber], newPoint;

  binsRS.resize(constants::index_table::LTFseed2BinWin);
  binsPhiS.resize(constants::index_table::LTFseed2BinWin);

  binsR.resize(constants::index_table::LTFinterBinWin);
  binsPhi.resize(constants::index_table::LTFinterBinWin);

  roadId = 0;

  for (Int_t layer1 = layer1Min; layer1 <= layer1Max; ++layer1) {

    nClsInLayer1 = event.getClustersInLayer(layer1).size();

    for (Int_t layer2 = layer2Max[layer1]; layer2 >= layer2Min[layer1]; --layer2) {

      nClsInLayer2 = event.getClustersInLayer(layer2).size();

      for (Int_t clsLayer1 = 0; clsLayer1 < nClsInLayer1; ++clsLayer1) {

        if (event.isClusterUsed(layer1, clsLayer1)) {
          continue;
        }
        const Cluster& cluster1 = event.getClustersInLayer(layer1)[clsLayer1];

        // project to the second seed layer and get the bin index in R and Phi
        getRPhiProjectionBin(cluster1, layer1, layer2, binR_proj, binPhi_proj);
        // define the search window in bins x bins (3x3, 5x5, etc.)
        for (Int_t i = 0; i < constants::index_table::LTFseed2BinWin; ++i) {
          binsRS[i] = binR_proj + (i - constants::index_table::LTFseed2BinWin / 2);
          binsPhiS[i] = binPhi_proj + (i - constants::index_table::LTFseed2BinWin / 2);
        }

        // loop over the bins in the search window
        for (auto binRS : binsRS) {
          for (auto binPhiS : binsPhiS) {
            // the global bin index
            bin = constants::index_table::getBinIndex(binRS, binPhiS);
            if (!getBinClusterRange(event, layer2, bin, clsMinIndexS, clsMaxIndexS)) {
              continue;
            }
            for (Int_t clsLayer2 = clsMinIndexS; clsLayer2 <= clsMaxIndexS; ++clsLayer2) {
              if (event.isClusterUsed(layer2, clsLayer2)) {
                continue;
              }
              const Cluster& cluster2 = event.getClustersInLayer(layer2)[clsLayer2];

              for (Int_t i = 0; i < (constants::mft::DisksNumber); i++) {
                hasDisk[i] = kFALSE;
              }

              hasDisk[layer1 / 2] = kTRUE;
              hasDisk[layer2 / 2] = kTRUE;

              // start a road
              event.addRoad();

              // add the 1st/2nd road points
              mcCompLabel = mUseMC ? event.getClusterLabels(layer1, cluster1.clusterId) : MCCompLabel();
              newPoint = kTRUE;
              event.getCurrentRoad().setPoint(cluster1.getX(), cluster1.getY(), cluster1.getZ(), layer1, clsLayer1, mcCompLabel, newPoint);

              for (Int_t layer = (layer1 + 1); layer <= (layer2 - 1); ++layer) {

                nClsInLayer = event.getClustersInLayer(layer).size();

                // project to the intermediate layer and get the bin index in R and Phi
                getRPhiProjectionBin(cluster1, layer1, layer, binR_proj, binPhi_proj);
                // define the search window in bins x bins (3x3, 5x5, etc.)
                for (Int_t i = 0; i < constants::index_table::LTFinterBinWin; ++i) {
                  binsR[i] = binR_proj + (i - constants::index_table::LTFinterBinWin / 2);
                  binsPhi[i] = binPhi_proj + (i - constants::index_table::LTFinterBinWin / 2);
                }

                // loop over the bins in the search window
                for (auto binR : binsR) {
                  for (auto binPhi : binsPhi) {
                    // the global bin index
                    bin = constants::index_table::getBinIndex(binR, binPhi);
                    if (!getBinClusterRange(event, layer, bin, clsMinIndex, clsMaxIndex)) {
                      continue;
                    }
                    for (Int_t clsLayer = clsMinIndex; clsLayer <= clsMaxIndex; ++clsLayer) {
                      if (event.isClusterUsed(layer, clsLayer)) {
                        continue;
                      }
                      const Cluster& cluster = event.getClustersInLayer(layer)[clsLayer];

                      dR = getDistanceToSeed(cluster1, cluster2, cluster);
                      // add all points within a radius dRcut
                      if (dR >= dRcut) {
                        continue;
                      }
                      hasDisk[layer / 2] = kTRUE;
                      mcCompLabel = mUseMC ? event.getClusterLabels(layer, cluster.clusterId) : MCCompLabel();
                      newPoint = kTRUE;
                      event.getCurrentRoad().setPoint(cluster.getX(), cluster.getY(), cluster.getZ(), layer, clsLayer, mcCompLabel, newPoint);

                    } // end clusters bin intermediate layer
                  }   // end intermediate layers
                }     // end binPhi
              }       // end binR

              // add the second seed-point
              mcCompLabel = mUseMC ? event.getClusterLabels(layer2, cluster2.clusterId) : MCCompLabel();
              newPoint = kTRUE;
              event.getCurrentRoad().setPoint(cluster2.getX(), cluster2.getY(), cluster2.getZ(), layer2, clsLayer2, mcCompLabel, newPoint);

              // keep only roads fulfilling the minimum length condition
              if (event.getCurrentRoad().getNPoints() < constants::mft::MinTrackPoints) {
                event.removeCurrentRoad();
                continue;
              }
              nPointDisks = 0;
              for (Int_t disk = 0; disk < (constants::mft::DisksNumber); ++disk) {
                if (hasDisk[disk])
                  ++nPointDisks;
              }
              if (nPointDisks < constants::mft::MinTrackPoints) {
                event.removeCurrentRoad();
                continue;
              }
              event.getCurrentRoad().setNDisks(nPointDisks);
              event.getCurrentRoad().setRoadId(roadId);
              ++roadId;

              computeCellsInRoad(event.getCurrentRoad());
              runForwardInRoad(event);
              runBackwardInRoad(event);

            } // end clusters bin layer2
          }   // end binPhiS
        }     // end binRS
      }       // end clusters in layer1
    }         // end layer2
  }           // end layer1
}

void Tracker::computeCellsInRoad(Road& road)
{
  Int_t layer1, layer1min, layer1max, layer2, layer2min, layer2max;
  Int_t nPtsInLayer1, nPtsInLayer2;
  Int_t clsLayer1, clsLayer2;
  Bool_t noCell;

  road.getLength(layer1min, layer1max);
  --layer1max;

  Int_t cellId = 0;
  for (layer1 = layer1min; layer1 <= layer1max; ++layer1) {
    layer2min = layer1 + 1;
    layer2max = std::min(layer1 + (constants::mft::DisksNumber - isDiskFace(layer1)), constants::mft::LayersNumber - 1);
    nPtsInLayer1 = road.getNPointsInLayer(layer1);
    for (Int_t point1 = 0; point1 < nPtsInLayer1; ++point1) {
      clsLayer1 = road.getClustersIdInLayer(layer1)[point1];
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
          clsLayer2 = road.getClustersIdInLayer(layer2)[point2];
          noCell = kFALSE;
          // create a cell
          road.addCellToLayer(layer1, layer2, clsLayer1, clsLayer2, cellId++);
        } // end points in layer2
        ++layer2;
      } // end while(noCell && (layer2 <= layer2max))
    }   // end points in layer1
  }     // end layer1
}

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
        const Cell& cellL = road.getCellsInLayer(layerL)[icellL];

        if (cellL.getLevel() == 0) {
          continue;
        }

        layerR = cellL.getSecondLayerId();
        if (layerR >= (constants::mft::LayersNumber - 1)) {
          continue;
        }

        for (icellR = 0; icellR < road.getCellsInLayer(layerR).size(); ++icellR) {
          const Cell& cellR = road.getCellsInLayer(layerR)[icellR];

          if (cellR.getLevel() == 0) {
            continue;
          }
          if ((cellL.getLevel() == cellR.getLevel()) && getCellsConnect(event, cellL, cellR)) {
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

void Tracker::runBackwardInRoad(ROframe& event)
{
  if (mMaxCellLevel == 1)
    return; // we have only isolated cells

  Bool_t addCellToNewTrack, hasDisk[constants::mft::DisksNumber];

  Int_t iSelectChisquare, iSelectDeviation, lastCellLayer, lastCellId;
  Int_t icell, layerC, cellIdC, nPointDisks;
  Float_t chisquarePrev, deviationPrev, deviation, chisquare;

  // start layer
  Int_t minLayer = 6;
  Int_t maxLayer = 8;

  Road& road = event.getCurrentRoad();

  for (Int_t layer = maxLayer; layer >= minLayer; --layer) {

    for (icell = 0; icell < road.getCellsInLayer(layer).size(); ++icell) {
      if (road.getCellLevel(layer, icell) == 0) {
        continue;
      }
      if (road.isCellUsed(layer, icell)) {
        continue;
      }
      if (road.getCellLevel(layer, icell) < (constants::mft::MinTrackPoints - 1)) {
        continue;
      }

      // start a track CA
      event.addTrackCA();
      event.getCurrentTrackCA().setRoadId(road.getRoadId());
      event.getCurrentTrackCA().setCA();
      if (addCellToCurrentTrackCA(layer, icell, event)) {
        road.setCellUsed(layer, icell, kTRUE);
      }

      // add cells to new track
      addCellToNewTrack = kTRUE;
      while (addCellToNewTrack) {
        Int_t layerRC = event.getCurrentTrackCA().getCellsLayer()[event.getCurrentTrackCA().getNCells() - 1];
        Int_t cellIdRC = event.getCurrentTrackCA().getCellsId()[event.getCurrentTrackCA().getNCells() - 1];
        const Cell& cellRC = road.getCellsInLayer(layerRC)[cellIdRC];
        addCellToNewTrack = kFALSE;

        // find the left neighbor giving the smalles chisquare
        iSelectChisquare = 0;
        chisquarePrev = 0.;

        // ... or

        // find the left neighbor giving the smallest deviation
        iSelectDeviation = 0;
        deviationPrev = -1.;

        // loop over left neighbours
        deviationPrev = o2::constants::math::TwoPI;
        chisquarePrev = 1.E5;

        for (Int_t iLN = 0; iLN < cellRC.getNLeftNeighbours(); ++iLN) {
          auto leftNeighbour = cellRC.getLeftNeighbours()[iLN];
          Int_t layerL = leftNeighbour.first;
          Int_t cellIdL = leftNeighbour.second;

          const Cell& cellL = road.getCellsInLayer(layerL)[cellIdL];

          if (road.getCellLevel(layerL, cellIdL) == 0) {
            continue;
          }
          if (road.isCellUsed(layerL, cellIdL)) {
            continue;
          }
          if (road.getCellLevel(layerL, cellIdL) != (road.getCellLevel(layerRC, cellIdRC) - 1)) {
            continue;
          }
          /*
          // ... smallest deviation
          deviation = getCellDeviation(event, cellL, cellRC);
          if (deviation < deviationPrev) {
            deviationPrev = deviation;
	    if (leftNeighbour != cellRC.getLeftNeighbours().front()) {
	      event.getCurrentTrackCA().removeLastCell(lastCellLayer, lastCellId);
              road.setCellUsed(lastCellLayer, lastCellId, kFALSE);
	      road.setCellLevel(lastCellLayer, lastCellId, 1);
	    }
	    if (addCellToCurrentTrackCA(layerL, cellIdL, event)) {
	      addCellToNewTrack = kTRUE;
              road.setCellUsed(layerL, cellIdL, kTRUE);
            }
          }
	  */
          // ... smallest chisquare
          chisquare = getCellChisquare(event, cellL);
          if (chisquare > 0.0 && chisquare < chisquarePrev) {
            chisquarePrev = chisquare;

            if (leftNeighbour != cellRC.getLeftNeighbours().front()) {
              event.getCurrentTrackCA().removeLastCell(lastCellLayer, lastCellId);
              road.setCellUsed(lastCellLayer, lastCellId, kFALSE);
              road.setCellLevel(lastCellLayer, lastCellId, 1);
            }
            if (addCellToCurrentTrackCA(layerL, cellIdL, event)) {
              addCellToNewTrack = kTRUE;
              road.setCellUsed(layerL, cellIdL, kTRUE);
            } else {
              //LOG(INFO) << "***** Failed to add cell to the current CA track! *****\n";
            }
          }

        } // end loop left neighbour
      }   // end  while(addCellToNewTrack)

      // check the track length
      for (Int_t i = 0; i < (constants::mft::DisksNumber); ++i) {
        hasDisk[i] = kFALSE;
      }
      for (icell = 0; icell < event.getCurrentTrackCA().getNCells(); ++icell) {
        layerC = event.getCurrentTrackCA().getCellsLayer()[icell];
        cellIdC = event.getCurrentTrackCA().getCellsId()[icell];
        const Cell& cellC = road.getCellsInLayer(layerC)[cellIdC];
        hasDisk[cellC.getFirstLayerId() / 2] = kTRUE;
        hasDisk[cellC.getSecondLayerId() / 2] = kTRUE;
      }
      nPointDisks = 0;
      for (Int_t disk = 0; disk < (constants::mft::DisksNumber); ++disk) {
        if (hasDisk[disk]) {
          ++nPointDisks;
        }
      }
      if (nPointDisks < constants::mft::MinTrackPoints) {
        for (icell = 0; icell < event.getCurrentTrackCA().getNCells(); ++icell) {
          layerC = event.getCurrentTrackCA().getCellsLayer()[icell];
          cellIdC = event.getCurrentTrackCA().getCellsId()[icell];
          road.setCellUsed(layerC, cellIdC, kFALSE);
          road.setCellLevel(layerC, cellIdC, 1);
        }
        event.removeCurrentTrackCA();
        continue;
      }
      // marked the used clusters
      for (icell = 0; icell < event.getCurrentTrackCA().getNCells(); ++icell) {
        layerC = event.getCurrentTrackCA().getCellsLayer()[icell];
        cellIdC = event.getCurrentTrackCA().getCellsId()[icell];
        const Cell& cellC = event.getCurrentRoad().getCellsInLayer(layerC)[cellIdC];
        event.markUsedCluster(cellC.getFirstLayerId(), cellC.getFirstClusterIndex());
        event.markUsedCluster(cellC.getSecondLayerId(), cellC.getSecondClusterIndex());
      }
    } // end loop cells
  }   // end loop start layer
}

void Tracker::updateCellStatusInRoad(Road& road)
{
  for (Int_t layer = 0; layer < (constants::mft::LayersNumber - 1); ++layer) {
    for (Int_t icell = 0; icell < road.getCellsInLayer(layer).size(); ++icell) {
      road.updateCellLevel(layer, icell);
      mMaxCellLevel = std::max(mMaxCellLevel, road.getCellLevel(layer, icell));
    }
  }
}

const Float_t Tracker::getCellChisquare(ROframe& event, const Cell& cell) const
{
  // returns the new chisquare of the previous cells plus the new one
  TrackCA& trackCA = event.getCurrentTrackCA();
  const Int_t layer2 = cell.getSecondLayerId();
  const Int_t cls2Id = cell.getSecondClusterIndex();
  const Cluster& cluster2 = event.getClustersInLayer(layer2)[cls2Id];

  Float_t x[constants::mft::MaxTrackPoints], y[constants::mft::MaxTrackPoints], z[constants::mft::MaxTrackPoints], err[constants::mft::MaxTrackPoints];
  Int_t point;

  for (point = 0; point < trackCA.getNPoints(); ++point) {
    x[point] = trackCA.getXCoordinates()[point];
    y[point] = trackCA.getYCoordinates()[point];
    z[point] = trackCA.getZCoordinates()[point];
    err[point] = constants::mft::Resolution; // FIXME
  }
  x[point] = cluster2.getX();
  y[point] = cluster2.getY();
  z[point] = cluster2.getZ();
  err[point] = constants::mft::Resolution; // FIXME

  // linear regression in the plane z:x
  // x = zxApar * z + zxBpar
  Float_t zxApar, zxBpar, zxAparErr, zxBparErr, chisqZX = 0.0;
  if (!LinearRegression(trackCA.getNPoints() + 1, z, x, err, zxApar, zxAparErr, zxBpar, zxBparErr, chisqZX)) {
    return -1.0;
  }
  // linear regression in the plane z:y
  // y = zyApar * z + zyBpar
  Float_t zyApar, zyBpar, zyAparErr, zyBparErr, chisqZY = 0.0;
  if (!LinearRegression(trackCA.getNPoints() + 1, z, y, err, zyApar, zyAparErr, zyBpar, zyBparErr, chisqZY)) {
    return -1.0;
  }

  Int_t nDegFree = 2 * (trackCA.getNPoints() + 1) - 4;
  return (chisqZX + chisqZY) / (Float_t)nDegFree;
}

const Bool_t Tracker::addCellToCurrentTrackCA(const Int_t layer1, const Int_t cellId, ROframe& event)
{
  TrackCA& trackCA = event.getCurrentTrackCA();
  Road& road = event.getCurrentRoad();
  const Cell& cell = road.getCellsInLayer(layer1)[cellId];
  const Int_t layer2 = cell.getSecondLayerId();
  const Int_t cls1Id = cell.getFirstClusterIndex();
  const Int_t cls2Id = cell.getSecondClusterIndex();

  const Cluster& cluster1 = event.getClustersInLayer(layer1)[cls1Id];
  const Cluster& cluster2 = event.getClustersInLayer(layer2)[cls2Id];

  if (trackCA.getNPoints() > 0) {
    const Float_t xLast = trackCA.getXCoordinates()[trackCA.getNPoints() - 1];
    const Float_t yLast = trackCA.getYCoordinates()[trackCA.getNPoints() - 1];
    Float_t dx = xLast - cluster2.getX();
    Float_t dy = yLast - cluster2.getY();
    Float_t dr = std::sqrt(dx * dx + dy * dy);
    if (dr > constants::mft::Resolution) {
      return kFALSE;
    }
  }

  MCCompLabel mcCompLabel1 = mUseMC ? event.getClusterLabels(layer1, cluster1.clusterId) : MCCompLabel();
  MCCompLabel mcCompLabel2 = mUseMC ? event.getClusterLabels(layer2, cluster2.clusterId) : MCCompLabel();

  Bool_t newPoint;

  if (trackCA.getNPoints() == 0) {
    newPoint = kTRUE;
    trackCA.setPoint(cluster2.getX(), cluster2.getY(), cluster2.getZ(), layer2, cls2Id, mcCompLabel2, newPoint);
  }

  newPoint = kTRUE;
  trackCA.setPoint(cluster1.getX(), cluster1.getY(), cluster1.getZ(), layer1, cls1Id, mcCompLabel1, newPoint);

  trackCA.addCell(layer1, cellId);

  // update the chisquare
  if (trackCA.getNPoints() == 2) {

    trackCA.setChiSquareZX(0.0);
    trackCA.setChiSquareZY(0.0);
    return kTRUE;
  }

  Float_t x[constants::mft::MaxTrackPoints], y[constants::mft::MaxTrackPoints], z[constants::mft::MaxTrackPoints], err[constants::mft::MaxTrackPoints];
  for (Int_t point = 0; point < trackCA.getNPoints(); ++point) {
    x[point] = trackCA.getXCoordinates()[point];
    y[point] = trackCA.getYCoordinates()[point];
    z[point] = trackCA.getZCoordinates()[point];
    err[point] = constants::mft::Resolution; // FIXME
  }

  // linear regression in the plane z:x
  // x = zxApar * z + zxBpar
  Float_t zxApar, zxBpar, zxAparErr, zxBparErr, chisqZX = 0.0;
  if (LinearRegression(trackCA.getNPoints(), z, x, err, zxApar, zxAparErr, zxBpar, zxBparErr, chisqZX)) {
    trackCA.setChiSquareZX(chisqZX);
  } else {
    return kFALSE;
  }

  // linear regression in the plane z:y
  // y = zyApar * z + zyBpar
  Float_t zyApar, zyBpar, zyAparErr, zyBparErr, chisqZY = 0.0;
  if (LinearRegression(trackCA.getNPoints(), z, y, err, zyApar, zyAparErr, zyBpar, zyBparErr, chisqZY)) {
    trackCA.setChiSquareZY(chisqZY);
  } else {
    return kFALSE;
  }

  return kTRUE;
}

const Bool_t Tracker::LinearRegression(Int_t npoints, Float_t* x, Float_t* y, Float_t* yerr, Float_t& apar, Float_t& aparerr, Float_t& bpar, Float_t& bparerr, Float_t& chisq, Int_t skippoint) const
{
  // y = apar * x + bpar

  // work with only part of the points
  Float_t xCl[constants::mft::MaxTrackPoints], yCl[constants::mft::MaxTrackPoints], yClErr[constants::mft::MaxTrackPoints];
  Int_t ipoints = 0;
  for (Int_t i = 0; i < npoints; ++i) {
    if (i == skippoint) {
      continue;
    }
    xCl[ipoints] = x[i];
    yCl[ipoints] = y[i];
    yClErr[ipoints] = yerr[i];
    ipoints++;
  }

  // calculate the regression parameters
  Float_t S1, SXY, SX, SY, SXX, SsXY, SsXX, SsYY, Xm, Ym, s, delta, difx;
  S1 = SXY = SX = SY = SXX = 0.0;
  SsXX = SsYY = SsXY = Xm = Ym = 0.0;
  difx = 0.;
  for (Int_t i = 0; i < ipoints; ++i) {
    S1 += 1.0 / (yClErr[i] * yClErr[i]);
    SXY += xCl[i] * yCl[i] / (yClErr[i] * yClErr[i]);
    SX += xCl[i] / (yClErr[i] * yClErr[i]);
    SY += yCl[i] / (yClErr[i] * yClErr[i]);
    SXX += xCl[i] * xCl[i] / (yClErr[i] * yClErr[i]);
    if (i > 0)
      difx += TMath::Abs(xCl[i] - xCl[i - 1]);
    Xm += xCl[i];
    Ym += yCl[i];
    SsXX += xCl[i] * xCl[i];
    SsYY += yCl[i] * yCl[i];
    SsXY += xCl[i] * yCl[i];
  }
  delta = SXX * S1 - SX * SX;
  if (delta == 0.) {
    return kFALSE;
  }
  apar = (SXY * S1 - SX * SY) / delta;
  bpar = (SY * SXX - SX * SXY) / delta;

  // calculate the chisquare
  chisq = 0.0;
  for (Int_t i = 0; i < ipoints; ++i) {
    chisq += (yCl[i] - (apar * xCl[i] + bpar)) * (yCl[i] - (apar * xCl[i] + bpar)) / (yerr[i] * yerr[i]);
  }

  // calculate the errors of the regression parameters
  Ym /= (Float_t)ipoints;
  Xm /= (Float_t)ipoints;
  SsYY -= (Float_t)ipoints * (Ym * Ym);
  SsXX -= (Float_t)ipoints * (Xm * Xm);
  SsXY -= (Float_t)ipoints * (Ym * Xm);
  Float_t eps = 1.E-24;
  if ((ipoints > 2) && (TMath::Abs(difx) > eps) && ((SsYY - (SsXY * SsXY) / SsXX) > 0.0)) {
    s = std::sqrt((SsYY - (SsXY * SsXY) / SsXX) / (ipoints - 2));
    bparerr = s * std::sqrt(1. / (Float_t)ipoints + (Xm * Xm) / SsXX);
    aparerr = s / std::sqrt(SsXX);
  } else {
    bparerr = 0.;
    aparerr = 0.;
  }

  return kTRUE;
}

bool Tracker::fitTracks(ROframe& event)
{
  for (auto& track : event.getTracksLTF()) {
    TrackLTF outParam = track;
    mTrackFitter->initTrack(track);
    mTrackFitter->fit(track);
    mTrackFitter->initTrack(outParam, true);
    mTrackFitter->fit(outParam, true);
    track.SetOutParam(outParam);
  }
  for (auto& track : event.getTracksCA()) {
    track.sort();
    mTrackFitter->fit(track);
  }

  return true;
}

} // namespace mft
} // namespace o2
