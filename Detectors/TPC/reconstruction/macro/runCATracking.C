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

#include "TPCReconstruction/Cluster.h"
#include "TPCReconstruction/TPCCATracking.h"
#include "TPCReconstruction/TrackTPC.h"
#include "DetectorsBase/Track.h"
#endif

using namespace std;
using namespace o2::TPC;

//This is a prototype of a macro to test running the HLT O2 CA Tracking library on a root input file containg TClonesArray of clusters.
//It wraps the TPCCATracking class, forwwarding all parameters, which are passed as options.
int runCATracking(TString filename="", TString outputFile="", TString options="", bool mergeChain = false, int nmaxEvent=-1, int startEvent=0) {
  if (filename.EqualTo("") || outputFile.EqualTo("")) {printf("Filename missing\n");return(1);}
  TPCCATracking tracker;
  vector<TrackTPC> tracks;
  if (tracker.initialize(options.Data())) {
    printf("Error initializing tracker\n");
    return(1);
  }

  // ===| input chain initialisation |==========================================
  TChain c("o2sim");
  c.AddFile(filename);

  // ===| output tree |=========================================================
  TFile fout(outputFile, "recreate");
  TTree tout("events","events");
  tout.Branch("Tracks", &tracks);

  if (mergeChain) {
    tracks.clear();
    printf("Processing full TChain of clusters at once\n");
    if (tracker.runTracking(&c, &tracks) == 0)     {
      printf("\tFound %d tracks\n", (int) tracks.size());
    } else {
      printf("\tError during tracking\n");
    }
    tout.Fill();
  } else {
    std::vector<o2::TPC::Cluster>* clusters=nullptr;
    c.SetBranchAddress("TPCClusterHW", &clusters);

    // ===| event ranges |========================================================
    const int nentries = c.GetEntries();
    const int start = startEvent>nentries?0:startEvent;
    const int max   = nmaxEvent>0 ? TMath::Min(nmaxEvent, nentries-startEvent) : nentries;

    // ===| loop over events |====================================================
    for (int iEvent=0; iEvent<max; ++iEvent)   {
      c.GetEntry(start+iEvent);

      printf("Processing event %d with %zu clusters\n", iEvent, clusters->size());
      if (!clusters->size()) continue;

      tracks.clear();
      if (tracker.runTracking(clusters, &tracks) == 0)     {
        printf("\tFound %d tracks\n", (int) tracks.size());
      } else {
        printf("\tError during tracking\n");
      }
      tout.Fill();
    }
  }
  fout.Write();
  fout.Close();

  tracker.deinitialize();
  return(0);
}
