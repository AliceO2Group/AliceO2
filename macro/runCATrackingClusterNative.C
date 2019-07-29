// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <fstream>
#include <iostream>
#include <vector>
#include "TSystem.h"

#include "TChain.h"
#include "TClonesArray.h"
#include "TFile.h"
#include "TROOT.h"
#include "TTree.h"

#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/Helpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsTPC/Constants.h"
#include "TPCReconstruction/GPUCATracking.h"
#include "GPUO2InterfaceConfiguration.h"
#include "DataFormatsTPC/TrackTPC.h"
#else
#pragma cling load("libO2TPCReconstruction")
#pragma cling load("libO2DataFormatsTPC")
#endif

using namespace o2::gpu;
using namespace o2::tpc;
using namespace o2::dataformats;
using namespace std;

using MCLabelContainer = MCTruthContainer<o2::MCCompLabel>;

#if !defined(__CLING__) || defined(__ROOTCLING__) // Disable in interpreted mode due to missing rootmaps

// This is a prototype of a macro to test running the HLT O2 CA Tracking library on a root input file containg
// TClonesArray of clusters.
// It wraps the GPUCATracking class
int runCATrackingClusterNative(TString inputFile, TString outputFile)
{
  if (inputFile.EqualTo("") || outputFile.EqualTo("")) {
    printf("Filename missing\n");
    return (1);
  }
  GPUCATracking tracker;

  // Just some default options to keep the macro running for now
  // Should be deprecated anyway in favor of the TPC workflow
  GPUO2InterfaceConfiguration config;
  config.configEvent.continuousMaxTimeBin = 0.023 * 5e6;
  config.configReconstruction.NWays = 3;
  config.configReconstruction.NWaysOuter = true;
  config.configReconstruction.SearchWindowDZDR = 2.5f;
  if (tracker.initialize(config)) {
    printf("Error initializing tracker\n");
    return (0);
  }

  std::vector<ClusterNativeContainer> cont;
  std::vector<MCLabelContainer> contMC;
  std::unique_ptr<ClusterNative[]> clusterBuffer;
  MCLabelContainer clusterMCBuffer;
  bool doMC = true;

  TFile fin(inputFile);
  for (int i = 0; i < Constants::MAXSECTOR; i++) {
    for (int j = 0; j < Constants::MAXGLOBALPADROW; j++) {
      TString contName = Form("clusters_sector_%d_row_%d", i, j);
      TObject* tmp = fin.FindObjectAny(contName);
      if (tmp == nullptr) {
        printf("Error reading clusters %s\n", contName.Data());
      } else {
        cont.emplace_back(std::move(*reinterpret_cast<ClusterNativeContainer*>(tmp)));
        tmp = fin.FindObjectAny(Form("clustersMCTruth_sector_%d_row_%d", i, j));

        if (tmp == nullptr) {
          printf("Error, clustersMCTruth missing or clusters and clustersMCtruth out of sync! Disabling MC data\n");
          doMC = false;
        } else {
          contMC.emplace_back(std::move(*reinterpret_cast<MCLabelContainer*>(tmp)));
        }
      }
    }
  }
  fin.Close();

  std::unique_ptr<ClusterNativeAccess> clusters =
    ClusterNativeHelper::createClusterNativeIndex(clusterBuffer, cont, doMC ? &clusterMCBuffer : nullptr, doMC ? &contMC : nullptr);

  vector<TrackTPC> tracks;
  MCLabelContainer tracksMC;

  TFile fout(outputFile, "recreate");
  TTree tout("events", "events");
  tout.Branch("Tracks", &tracks);
  tout.Branch("TracksMCTruth", &tracksMC);

  printf("Processing time frame\n");
  GPUO2InterfaceIOPtrs ptrs;
  ptrs.clusters = clusters.get();
  ptrs.outputTracks = &tracks;
  ptrs.outputTracksMCTruth = doMC ? &tracksMC : nullptr;
  if (tracker.runTracking(&ptrs) == 0) {
    printf("\tFound %d tracks\n", (int)tracks.size());
  } else {
    printf("\tError during tracking\n");
  }

  float artificialVDrift = tracker.getPseudoVDrift();
  float tfReferenceLength = tracker.getTFReferenceLength();

  // partial printout of 100 tracks
  int step = tracks.size() / 100;
  step = step < 1 ? 1 : step;
  for (unsigned int i = 0; i < tracks.size(); i += step) {
    // Loop over clusters
    for (int j = tracks[i].getNClusterReferences() - 1; j >= 0; j--) {
      // Get cluster references
      uint8_t sector, row;
      uint32_t clusterIndexInRow;
      tracks[i].getClusterReference(j, sector, row, clusterIndexInRow);
      const ClusterNative& cl = tracks[i].getCluster(j, *clusters, sector, row);
      const ClusterNative& clLast = tracks[i].getCluster(0, *clusters);
      // RS: TODO: account for possible A/C merged tracks
      float sideFactor = tracks[i].hasASideClustersOnly() ? -1.f : 1.f;
      printf(
        "Track %d: Side %s Estimated timeVertex: %f, num clusters %d, innermost cluster: sector %d, row %d, "
        "ClusterTime %f, TrackParam X %f Z %f --> T %f, LastClusterT: %f\n",
        i, tracks[i].hasBothSidesClusters() ? "AC" : (tracks[i].hasASideClusters() ? "A" : "C"), tracks[i].getTime0(),
        tracks[i].getNClusterReferences(), (int)sector, (int)row, cl.getTime(), tracks[i].getX(), tracks[i].getZ(),
        tracks[i].getTime0() - sideFactor * tracks[i].getZ() / artificialVDrift, clLast.getTime());
      break; // Reduce output in this example code
    }
  }

  tout.Fill();
  fout.Write();
  fout.Close();

  tracker.deinitialize();
  return (0);
}

#endif
