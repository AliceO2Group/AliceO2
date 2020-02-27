#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/HitType.h"
//#include "SimulationDataFormat/MCEventHeader.h"
//#include "DataFormatsParameters/GRPObject.h"
#include <TH2F.h>
#include <TTree.h>
#include <TFile.h>
#include <gsl/span>
#include <vector>

void readFT0digits()
{

  // Create histograms
  TDirectory* cwd = gDirectory;
  gDirectory = 0x0;

  TH2F* hMultDig = new TH2F("hMultDig", "amplitude  ", 210, 0, 210, 200, 0, 1000);
  TH2F* hTimeDig = new TH2F("hTimeDig", "Time", 210, 0, 210, 100, -100, 100);
  TH2F* hTimeDigMultA = new TH2F("hTimeDigMultA", "Time vs amplitude", 200, 0, 100, 100, -100, 100);
  TH2F* hTimeDigMultC = new TH2F("hTimeDigMultC", "Time vs amplitude", 200, 0, 100, 100, -100, 100);
  TH1F* hNchA = new TH1F("hNchA", "FT0-A", 100, 0, 100);
  TH1F* hNchC = new TH1F("hNchC", "FT0-C", 100, 0, 100);

  gDirectory = cwd;

  TFile* fdig = TFile::Open("ft0digits.root");
  std::cout << " Open digits file " << std::endl;
  TTree* digTree = (TTree*)fdig->Get("o2sim");
  std::vector<o2::ft0::Digit> digitsBC, *ft0BCDataPtr = &digitsBC;
  std::vector<o2::ft0::ChannelData> digitsCh, *ft0ChDataPtr = &digitsCh;

  digTree->SetBranchAddress("FT0DIGITSBC", &ft0BCDataPtr);
  digTree->SetBranchAddress("FT0DIGITSCH", &ft0ChDataPtr);

  float cfd[208], amp[208];
  for (int ient = 0; ient < digTree->GetEntries(); ient++) {
    digTree->GetEntry(ient);

    int nbc = digitsBC.size();
    std::cout << "Entry " << ient << " : " << nbc << " BCs stored" << std::endl;
    for (int ibc = 0; ibc < nbc; ibc++) {
      auto& bcd = digitsBC[ibc];
      for (int ii = 0; ii < 208; ii++) {
        cfd[ii] = amp[ii] = 0;
      }

      int nmcpa = 0;
      int nmcpc = 0;
      auto channels = bcd.getBunchChannelData(digitsCh);
      int nch = channels.size();
      for (int ich = 0; ich < nch; ich++) {
        if (nch) {
          channels[ich].print();
          Int_t mcp = channels[ich].ChId;
          cfd[mcp] = float(channels[ich].CFDTime);
          amp[mcp] = float(channels[ich].QTCAmpl);
          hMultDig->Fill(Float_t(mcp), amp[mcp]);
          hTimeDig->Fill(Float_t(mcp), cfd[mcp]);
          if (mcp < 96) {
            nmcpa++;
            hTimeDigMultA->Fill(amp[mcp], cfd[mcp]);
          } else {
            hTimeDigMultC->Fill(amp[mcp], cfd[mcp]);
            nmcpc++;
          }
        }
      }
      hNchA->Fill(nmcpa);
      hNchC->Fill(nmcpc);
    }
  }
  TFile* Hfile = new TFile("digitsFT0.root", "RECREATE");
  printf("Writting histograms to root file \n");
  Hfile->cd();
  //Create a canvas, set the view range, show histograms
  //  TCanvas *c1 = new TCanvas("c1","Alice T0 Time ",400,10,600,600);
  hTimeDig->Write();
  hTimeDigMultA->Write();
  hTimeDigMultC->Write();
  hMultDig->Write();
  hNchA->Write();
  hNchC->Write();

} // end of macro
#endif
