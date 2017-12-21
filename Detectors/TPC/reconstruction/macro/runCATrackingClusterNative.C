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
#include <vector>
#include <fstream>
#include <iostream>
#include "TSystem.h"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TClonesArray.h"

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCReconstruction/TPCCATracking.h"
#include "TPCReconstruction/TrackTPC.h"
#include "DetectorsBase/Track.h"
#include "TPCBase/Constants.h" 
#else
#pragma cling load("libTPCReconstruction")
#pragma cling load("libDataFormatsTPC")
#endif

using namespace o2;
using namespace o2::TPC;
using namespace o2::DataFormat::TPC;
using namespace o2::dataformats;
using namespace std;

using MCLabelContainer = MCTruthContainer<MCCompLabel>;

//This is a prototype of a macro to test running the HLT O2 CA Tracking library on a root input file containg TClonesArray of clusters.
//It wraps the TPCCATracking class, forwwarding all parameters, which are passed as options.
int runCATrackingClusterNative(TString inputFile, TString outputFile, TString options="") {
  if (inputFile.EqualTo("") || outputFile.EqualTo("")) {printf("Filename missing\n");return(1);}
  TPCCATracking tracker;
  
  if (tracker.initialize(options.Data())) {
    printf("Error initializing tracker\n");
    return(0);
  }

  std::vector<ClusterNativeContainer> cont;
  std::vector<MCLabelContainer> contMC;
  bool doMC = true;

  TFile fin(inputFile);
  for (int i = 0;i < Constants::MAXSECTOR;i++)
  {
    for (int j = 0;j < Constants::MAXGLOBALPADROW;j++)
    {
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

  std::unique_ptr<ClusterNativeAccessFullTPC> clusters = TPCClusterFormatHelper::accessNativeContainerArray(cont, doMC ? &contMC : nullptr);

  vector<TrackTPC> tracks;
  MCLabelContainer tracksMC;

  TFile fout(outputFile, "recreate");
  TTree tout("events","events");
  tout.Branch("Tracks", &tracks);
  tout.Branch("TracksMCTruth", &tracksMC);

  printf("Processing time frame\n");
  if (tracker.runTracking(*clusters, &tracks, doMC ? &tracksMC : nullptr) == 0)     {
    printf("\tFound %d tracks\n", (int) tracks.size());
  } else {
    printf("\tError during tracking\n");
  }
  
  float artificialVDrift = tracker.getPseudoVDrift();
  float tfReferenceLength = tracker.getTFReferenceLength();
  unsigned int nTracksASide = tracker.getNTracksASide();
  for (unsigned int i = 0;i < tracks.size();i++)
  {
    bool isASide = i < nTracksASide;
    if (isASide != (tracks[i].getSide() == Side::A)) printf("Incorrect sorting\n");
    //Loop over clusters
    for (int j = tracks[i].getNClusterReferences() - 1;j >= 0;j--)
    {
      //Get cluster references
      uint8_t sector, row;
      uint32_t clusterIndexInRow;
      tracks[i].getClusterReference(j, sector, row, clusterIndexInRow);
      const ClusterNative& cl = tracks[i].getCluster(j, *clusters, sector, row);
      const ClusterNative& clLast = tracks[i].getCluster(0, *clusters);
      float sideFactor = tracks[i].getSide() == Side::A ? -1.f : 1.f;
      
      printf("Track %d: Side %c Estimated timeVertex: %f, num clusters %d, innermost cluster: sector %d, row %d, ClusterTime %f, TrackParam X %f Z %f --> T %f, LastClusterZ %f --> T %f (T from cluster itself %f)\n",
        i, tracks[i].getSide() == Side::A ? 'A' : 'C', tracks[i].getTime0(), tracks[i].getNClusterReferences(), (int) sector, (int) row, cl.getTime(), tracks[i].getX(),
        tracks[i].getZ(), tracks[i].getTime0() - sideFactor * tracks[i].getZ() / artificialVDrift,
        tracks[i].getLastClusterZ(), tracks[i].getTime0() - sideFactor * tracks[i].getLastClusterZ() / artificialVDrift, clLast.getTime());
      break; //Reduce output in this example code
    }
  }
  
  tout.Fill();
  fout.Write();
  fout.Close();
  
  tracker.deinitialize();
  return(0);
}
