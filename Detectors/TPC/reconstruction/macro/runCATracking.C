// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <vector>
#include <fstream>
#include <iostream>
#include "TSystem.h"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TClonesArray.h"

#include "TPCBase/Defs.h"
#include "TPCBase/CRU.h"
#include "TPCBase/Sector.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCSimulation/Cluster.h"
#include "TPCReconstruction/TPCCATracking.h"
#include "TPCReconstruction/SyncPatternMonitor.h"
#include "TPCReconstruction/TrackTPC.h"
#include "DetectorsBase/Track.h"

using namespace o2::TPC;
using namespace std;

void runCATracking(TString filename, TString outputFile, TString options, Int_t nmaxEvent=-1, Int_t startEvent=0) {
  gSystem->Load("libTPCReconstruction.so");
  TPCCATracking tracker;
  vector<TrackTPC> tracks;
  if (tracker.initialize(options.Data())) {
    printf("Error initializing tracker\n");
    return;
  }

  // ===| input chain initialisation |==========================================
  TChain c("cbmsim");
  c.AddFile(filename);

  TClonesArray *clusters=0x0;
  c.SetBranchAddress("TPCClusterHW", &clusters);

  // ===| output tree |=========================================================
  TFile fout(outputFile, "recreate");
  TTree tout("events","events");
  tout.Branch("Tracks", &tracks);

  // ===| event ranges |========================================================
  const Int_t nentries = c.GetEntries();
  const Int_t start = startEvent>nentries?0:startEvent;
  const Int_t max   = nmaxEvent>0 ? TMath::Min(nmaxEvent, nentries-startEvent) : nentries;

  // ===| loop over events |====================================================
  for (Int_t iEvent=0; iEvent<max; ++iEvent)   {
    c.GetEntry(start+iEvent);

    printf("Processing event %d with %d clusters\n", iEvent, clusters->GetEntries());
    if (!clusters->GetEntries()) continue;

    tracks.clear();
    if (tracker.runTracking(clusters, &tracks) == 0)     {
      printf("\tFound %d tracks\n", (int) tracks.size());
    } else {
      printf("\tError during tracking\n");
    }
    tout.Fill();
  }
  fout.Write();
  fout.Close();

  tracker.deinitialize();
}
