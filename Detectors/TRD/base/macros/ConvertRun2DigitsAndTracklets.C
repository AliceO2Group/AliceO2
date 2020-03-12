#if !defined(__CINT__) || defined(__MAKECINT__)
#include <TClonesArray.h>
#include <TEveLine.h>
#include <TEveManager.h>
#include <TEveElement.h>

#include <AliRunLoader.h>
#include <AliLoader.h>
#include <AliDataLoader.h>
#include <AliTreeLoader.h>
#include <AliTRDarrayADC.h>
#include <AliTRDtrackletWord.h>
#include <AliTRDtrackletMCM.h>

#include <iostream>

#include "TH1F.h"
#include "AliTRDtrackletMCM.h"
//#include "TRDDataFormat/TriggerRecord.h"
#include "TRDBase/Digit.h"
#include "TRDBase/Tracklet.h"
#endif

using namespace o2;
using namespace trd;
using namespace std;

void ConvertRun2DigitsAndTracklets()
{

  TH1F* slope = new TH1F("slope", "mSlope", 800, -40, 40);
  TH1F* offset = new TH1F("offset", "mSlope", 1000, -400, 400);
  TH1F* residuals = new TH1F("residual", "mSlope", 1000, -400, 400);

  unsigned int trackletscount = 0;
  unsigned int digitscount = 0;
  std::vector<o2::trd::Digit> run3digits;
  std::vector<o2::trd::Tracklet> run3tracklets;
  //o2::dataformats::MCTruthContainer<o2::trd::MCLabel> mMCLabels, *mPMCLabels = &mMCLabels;
  //std::vector<o2::trd::TriggerRecord> mTriggerRecords, *mPTriggerRecords = &mTriggerRecords;

  AliRunLoader* rl = AliRunLoader::Open("galice.root");
  for (int j = 0; j < rl->GetNumberOfEvents(); j++) {
    double eventtime = 12 * j;
    rl->GetEvent(j);
    cout << "Event Number : " << rl->GetEventNumber() << endl;
    //so run2 event is now effectively 1 interaction. Is this correct, not sure, but does not matter so long as its consistant for run3 trapsim.

    //build arbitrary triggerrecord.
    //get the tracklets for this event.
    AliLoader* loader = rl ? rl->GetLoader("TRDLoader") : nullptr;
    AliDataLoader* dl = loader ? loader->GetDataLoader("tracklets") : nullptr;
    if (!dl) {
      cout << "Error getting tracklets loader" << endl;
    } else {
      dl->Load();
      TTree* trackletTree = dl->Tree();
      if (trackletTree) {
        TBranch* trackletBranch = nullptr;
        if ((trackletBranch = trackletTree->GetBranch("mcmtrklbranch"))) {
          AliTRDtrackletMCM* tracklet = nullptr;
          trackletBranch->SetAddress(&tracklet);
          for (int i = 0; i < trackletBranch->GetEntries(); i++) {
            //        cout << "Entry : " << i << endl;
            trackletBranch->GetEntry(i);
            if (tracklet) {
              trackletscount++;
              //do something with the tracklet.
              cout << tracklet->GetOffset() << endl;
              slope->Fill(tracklet->GetSlope());
              offset->Fill(tracklet->GetOffset());
              int clusters = tracklet->GetNClusters();
              float* residual;
              residual = tracklet->GetResiduals();
              for (int i = 0; i < clusters; i++)
                residuals->Fill(residual[i]);
              run3tracklets.push_back(o2::trd::Tracklet(tracklet->GetTrackletWord(), tracklet->GetHCId(), tracklet->GetROB(), tracklet->GetMCM()));
              int newtrackpos = run3tracklets.size() - 1;
              run3tracklets[newtrackpos].setNHits(tracklet->GetNHits());
              run3tracklets[newtrackpos].setNHits0(tracklet->GetNHits0());
              run3tracklets[newtrackpos].setNHits1(tracklet->GetNHits1());
              run3tracklets[newtrackpos].setQ0(tracklet->GetQ0());
              run3tracklets[newtrackpos].setQ1(tracklet->GetQ1());
              run3tracklets[newtrackpos].setSlope(tracklet->GetSlope());
              run3tracklets[newtrackpos].setOffset(tracklet->GetOffset());
              run3tracklets[newtrackpos].setError(tracklet->GetError());
            }
          }
        }
      }
    } //end of else dl.
    cout << " now for digits" << endl;
    // get the digits for this eventA
    // Link TRD digits
    rl->LoadDigits("TRD");
    TTree* tD = rl->GetTreeD("TRD", kFALSE);
    if (!tD) {
      Error("trd_digits", "Missing digits tree");
      return NULL;
    }
    AliTRDdigitsManager dm;
    dm.ReadDigits(tD);
    for (int i = 0; i < 540; i++) {
      AliTRDarrayADC* digits;
      digits = dm.GetDigits(i);
      digits->Expand();
      AliTRDSignalIndex* indexes = dm.GetIndexes(i);
      if (!indexes->IsAllocated())
        dm.BuildIndexes(i);
      int row, col;
      while (indexes->NextRCIndex(row, col)) {
        ArrayADC adctimes;
        if (digits->GetNtime() > 30)
          cout << "----!!! --- number of times is greater than 30" << endl;
        for (int adc = 0; adc < digits->GetNtime(); adc++) {
          adctimes[adc] = digits->GetData(row, col, adc);
        }
        run3digits.push_back(o2::trd::Digit(i, row, col, adctimes, 0, eventtime)); //triggertime);
        digitscount++;
      }
    }
  }

  // now write eveything out
  // need to keep the blank branches to keep the digireader happy, its this or change TRDDigitReaderSpec
  TFile* digitsfile = new TFile("trddigits.root", "RECREATE");
  TTree* digittree = new TTree("o2sim", "run2 digits");
  std::vector<o2::trd::Digit>* run3pdigits = &run3digits;
  digittree->Branch("TRDDigit", &run3pdigits);
  //digittree->Branch("LABELS",&mPMCLabels);
  //digittree->Branch("TRGRDIG",&mPTriggerRecords);
  digittree->Fill();
  cout << " run3digits is : " << run3digits.size() << endl;
  digittree->Write();
  delete digittree;
  delete digitsfile;

  TFile* trackletfile = new TFile("trdtrackletsrun2.root", "RECREATE");
  TTree* tracklettree = new TTree("o2sim", "run2 tracklets");
  std::vector<o2::trd::Tracklet>* run3ptracklets = &run3tracklets;
  tracklettree->Branch("Tracklet", &run3ptracklets);
  tracklettree->Fill();
  cout << " run3tracklets is : " << run3tracklets.size() << endl;
  tracklettree->Write();
  tracklettree->Print();
  delete tracklettree;
  delete trackletfile;
  slope->Draw();
  cout << "Digits:" << digitscount << " and Tracklets:" << trackletscount << endl;
}
