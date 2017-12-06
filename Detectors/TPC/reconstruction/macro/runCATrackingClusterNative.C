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

#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCReconstruction/TPCCATracking.h"
#include "TPCReconstruction/TrackTPC.h"
#include "DetectorsBase/Track.h"
#include "TPCBase/Constants.h" 
#endif

using namespace o2::TPC;
using namespace o2::DataFormat::TPC;
using namespace std;

//This is a prototype of a macro to test running the HLT O2 CA Tracking library on a root input file containg TClonesArray of clusters.
//It wraps the TPCCATracking class, forwwarding all parameters, which are passed as options.
void runCATrackingClusterNative(TString inputFile, TString outputFile, TString options="") {
  gSystem->Load("libTPCReconstruction.so");
  TPCCATracking tracker;
  vector<TrackTPC> tracks;
  if (tracker.initialize(options.Data())) {
    printf("Error initializing tracker\n");
    return;
  }

  std::vector<ClusterNativeContainer> cont;

  TFile fin(inputFile);
  for (int i = 0;i < Constants::MAXSECTOR;i++)
  {
    for (int j = 0;j < Constants::MAXGLOBALPADROW;j++)
    {
      TString contName = Form("clusters_sector_%d_row_%d", i, j);
      TObject* tmp = fin.FindObjectAny(contName);
      if (tmp == nullptr) printf("Error reading clusters %s\n", contName.Data());
      else cont.push_back(*(ClusterNativeContainer*) tmp);
    }
  }
  fin.Close();

  std::unique_ptr<ClusterNativeAccessFullTPC> clusters = TPCClusterFormatHelper::accessNativeContainerArray(cont);

  TFile fout(outputFile, "recreate");
  TTree tout("events","events");
  tout.Branch("Tracks", &tracks);

  tracks.clear();
  printf("Processing time frame\n");
  if (tracker.runTracking(*clusters, &tracks) == 0)     {
    printf("\tFound %d tracks\n", (int) tracks.size());
  } else {
    printf("\tError during tracking\n");
  }
  tout.Fill();
  fout.Write();
  fout.Close();
  
  tracker.deinitialize();
}
