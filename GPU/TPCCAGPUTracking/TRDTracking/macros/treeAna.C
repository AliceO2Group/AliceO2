//
// example macro to analyze the TRD tracking output tree
// 
// in a root session with the tree TRDhlt.root do:
/*
 .L treeAna.C
 InitTree()
 Efficiency()
 PlotChi2()
*/

TTree * tree = 0;

void SetLogX(TH1 *h)
{
   TAxis *ax = h->GetXaxis();
   Int_t nBins = ax->GetNbins();

   Axis_t from = ax->GetXmin();
   Axis_t to = ax->GetXmax();
   Axis_t width = (TMath::Log10(to) - TMath::Log10(from)) / nBins;

   Axis_t *newBins = new Axis_t[nBins+1];

   for (Int_t i=0; i<=nBins; i++) {
     newBins[i] = TMath::Power(10, TMath::Log10(from) + i * width);
   }

   ax->Set(nBins, newBins);

   delete newBins;
}

void InitTree() {
  TFile *f = TFile::Open("TRDhlt.root");
  tree = (TTree*)f->Get("tracksFinal");
  // aliases
  tree->SetAlias("correct0", "(update.fElements[0]>60)");
  tree->SetAlias("correct1", "(update.fElements[1]>60)");
  tree->SetAlias("correct2", "(update.fElements[2]>60)");
  tree->SetAlias("correct3", "(update.fElements[3]>60)");
  tree->SetAlias("correct4", "(update.fElements[4]>60)");
  tree->SetAlias("correct5", "(update.fElements[5]>60)");
  tree->SetAlias("related0", "(update.fElements[0]<60 && update.fElements[0]>4)");
  tree->SetAlias("related1", "(update.fElements[1]<60 && update.fElements[1]>4)");
  tree->SetAlias("related2", "(update.fElements[2]<60 && update.fElements[2]>4)");
  tree->SetAlias("related3", "(update.fElements[3]<60 && update.fElements[3]>4)");
  tree->SetAlias("related4", "(update.fElements[4]<60 && update.fElements[4]>4)");
  tree->SetAlias("related5", "(update.fElements[5]<60 && update.fElements[5]>4)");
  tree->SetAlias("fake0", "(update.fElements[0]<4 && update.fElements[0]>0)");
  tree->SetAlias("fake1", "(update.fElements[1]<4 && update.fElements[1]>0)");
  tree->SetAlias("fake2", "(update.fElements[2]<4 && update.fElements[2]>0)");
  tree->SetAlias("fake3", "(update.fElements[3]<4 && update.fElements[3]>0)");
  tree->SetAlias("fake4", "(update.fElements[4]<4 && update.fElements[4]>0)");
  tree->SetAlias("fake5", "(update.fElements[5]<4 && update.fElements[5]>0)");
}

void Efficiency() {

  const float ptCut = 1.; // cut on 1/pt

  double nCorrectUpdatesTot = 0;
  double nRelatedUpdatesTot = 0;
  double nFakeUpdatesTot = 0;
  double nUpdatesAvailable = 0;
  double nUpdatesOffline = 0;
  double nUpdatesMax = 0;

  TCanvas *cTmp = new TCanvas("cTmp", "cTmp");
  for (int iLy=0; iLy<6; iLy++) {
    TString cutCorrect = TString::Format("abs(track.fP[4])<%f && correct%i && findable.Sum() > -0.5", ptCut, iLy);
    TString cutRelated = TString::Format("abs(track.fP[4])<%f && related%i && findable.Sum() > -0.5", ptCut, iLy, iLy);
    TString cutFake = TString::Format("abs(track.fP[4])<%f && fake%i && findable.Sum() > -0.5", ptCut, iLy, iLy);
    TString cutOffline = TString::Format("abs(track.fP[4])<%f && nTrackletsOffline==%i && findable.Sum()>-0.5", ptCut, iLy+1);
    TString cutAvailable = TString::Format("abs(track.fP[4])<%f && matchAvailable.fElements[%i]>0 && findable.Sum()>-0.5", ptCut, iLy);

    double nCorrectUpdatesCurr = tree->Draw("nTracklets", cutCorrect);
    double nRelatedUpdatesCurr = tree->Draw("nTracklets", cutRelated);
    double nFakeUpdatesCurr = tree->Draw("nTracklets", cutFake);
    double nUpdatesOfflineCurr = tree->Draw("nTrackletsOffline", cutOffline);
    double nUpdatesAvailableCurr = tree->Draw("nTracklets", cutAvailable);

    nCorrectUpdatesTot += nCorrectUpdatesCurr;
    nRelatedUpdatesTot += nRelatedUpdatesCurr;
    nFakeUpdatesTot += nFakeUpdatesCurr;
    nUpdatesOffline += ((iLy+1) * nUpdatesOfflineCurr);
    nUpdatesAvailable += nUpdatesAvailableCurr;
  }
  nUpdatesMax = 6. * (tree->Draw("nTracklets", TString::Format("abs(track.fP[4])<%f && findable.Sum()>-0.5", ptCut)));
  cTmp->Close();

  printf("-------\n");
  printf("ptCut = %.2f GeV/c\n", 1./ptCut);
  printf("-------\n");
  printf("number of correct updates: %i\n", nCorrectUpdatesTot);
  printf("number of related updates: %i\n", nRelatedUpdatesTot);
  printf("number of fake updates: %i\n", nFakeUpdatesTot);
  printf("number of updates in total: %i\n", nCorrectUpdatesTot+nFakeUpdatesTot+nRelatedUpdatesTot);
  printf("-------\n");
  printf("number of tracks times 6 (max. number of updates): %i\n", nUpdatesMax);
  printf("-------\n");
  printf("efficiency (correct updates/max. number of updates): %f\n", nCorrectUpdatesTot/nUpdatesMax);
  printf("fake ratio (fake updates/all updates): %f\n", nFakeUpdatesTot/(nCorrectUpdatesTot+nFakeUpdatesTot+nRelatedUpdatesTot));
  printf("related ratio (related updates/all updates): %f\n", nRelatedUpdatesTot/(nCorrectUpdatesTot+nFakeUpdatesTot+nRelatedUpdatesTot));
  printf("-------\n");
  printf("number of attached offline tracklets: %i\n", nUpdatesOffline);
  printf("offline efficiency: %f\n", nUpdatesOffline/nUpdatesMax);
  printf("-------\n");
  printf("available correct updates in HLT: %i -> max. efficiency = %f\n", nUpdatesAvailable, nUpdatesAvailable/nUpdatesMax);
  printf("-------\n");
}

void PlotChi2() {
  
  const int nBins = 50;
  const double xStart = 0.1;
  const double xEnd = 20;
  TH1F *hChi2Corr[6];
  TH1F *hChi2Rel[6];
  TH1F *hChi2Fake[6];
  for (int iLy=0; iLy<6; iLy++) {
    const char *histNameCorr = TString::Format("hChi2CorrL%i", iLy);
    const char *histNameRel = TString::Format("hChi2RelL%i", iLy);
    const char *histNameFake = TString::Format("hChi2FakeL%i", iLy);
    hChi2Corr[iLy] = new TH1F(histNameCorr, ";#chi^{2};counts", nBins, xStart, xEnd);
    hChi2Rel[iLy] = new TH1F(histNameRel, ";#chi^{2};counts", nBins, xStart, xEnd);
    hChi2Fake[iLy] = new TH1F(histNameFake, ";#chi^{2};counts", nBins, xStart, xEnd);

    SetLogX(hChi2Corr[iLy]);
    SetLogX(hChi2Rel[iLy]);
    SetLogX(hChi2Fake[iLy]);
  }
  TH1F *hChi2CorrAll = new TH1F("chi2CorrAll", ";#chi^{2};counts", nBins, xStart, xEnd);
  TH1F *hChi2RelAll = new TH1F("chi2RelAll", ";#chi^{2};counts", nBins, xStart, xEnd);
  TH1F *hChi2FakeAll = new TH1F("chi2FakeAll", ";#chi^{2};counts", nBins, xStart, xEnd);

  SetLogX(hChi2CorrAll);
  SetLogX(hChi2RelAll);
  SetLogX(hChi2FakeAll);

  TCanvas *cTmp = new TCanvas("cTmp", "cTmp");
  tree->Draw("chi2Update.fElements[0]>>hChi2CorrL0", "correct0");
  tree->Draw("chi2Update.fElements[1]>>hChi2CorrL1", "correct1");
  tree->Draw("chi2Update.fElements[2]>>hChi2CorrL2", "correct2");
  tree->Draw("chi2Update.fElements[3]>>hChi2CorrL3", "correct3");
  tree->Draw("chi2Update.fElements[4]>>hChi2CorrL4", "correct4");
  tree->Draw("chi2Update.fElements[5]>>hChi2CorrL5", "correct5");
  tree->Draw("chi2Update.fElements[0]>>hChi2RelL0", "related0");
  tree->Draw("chi2Update.fElements[1]>>hChi2RelL1", "related1");
  tree->Draw("chi2Update.fElements[2]>>hChi2RelL2", "related2");
  tree->Draw("chi2Update.fElements[3]>>hChi2RelL3", "related3");
  tree->Draw("chi2Update.fElements[4]>>hChi2RelL4", "related4");
  tree->Draw("chi2Update.fElements[5]>>hChi2RelL5", "related5");
  tree->Draw("chi2Update.fElements[0]>>hChi2FakeL0", "fake0");
  tree->Draw("chi2Update.fElements[1]>>hChi2FakeL1", "fake1");
  tree->Draw("chi2Update.fElements[2]>>hChi2FakeL2", "fake2");
  tree->Draw("chi2Update.fElements[3]>>hChi2FakeL3", "fake3");
  tree->Draw("chi2Update.fElements[4]>>hChi2FakeL4", "fake4");
  tree->Draw("chi2Update.fElements[5]>>hChi2FakeL5", "fake5");
  cTmp->Close();

  hChi2CorrAll->Add(hChi2Corr[0]);
  hChi2CorrAll->Add(hChi2Corr[1]);
  hChi2CorrAll->Add(hChi2Corr[2]);
  hChi2CorrAll->Add(hChi2Corr[3]);
  hChi2CorrAll->Add(hChi2Corr[4]);
  hChi2CorrAll->Add(hChi2Corr[5]);
  hChi2RelAll->Add(hChi2Rel[0]);
  hChi2RelAll->Add(hChi2Rel[1]);
  hChi2RelAll->Add(hChi2Rel[2]);
  hChi2RelAll->Add(hChi2Rel[3]);
  hChi2RelAll->Add(hChi2Rel[4]);
  hChi2RelAll->Add(hChi2Rel[5]);
  hChi2FakeAll->Add(hChi2Fake[0]);
  hChi2FakeAll->Add(hChi2Fake[1]);
  hChi2FakeAll->Add(hChi2Fake[2]);
  hChi2FakeAll->Add(hChi2Fake[3]);
  hChi2FakeAll->Add(hChi2Fake[4]);
  hChi2FakeAll->Add(hChi2Fake[5]);

  hChi2CorrAll->SetLineWidth(2);
  hChi2RelAll->SetLineWidth(2);
  hChi2FakeAll->SetLineWidth(2);
  hChi2CorrAll->SetLineColor(kGreen);
  hChi2RelAll->SetLineColor(kBlue);
  hChi2FakeAll->SetLineColor(kRed);

  TCanvas *c1 = new TCanvas("c1", "c1");
  c1->SetLogx();
  hChi2CorrAll->Draw();
  hChi2RelAll->Draw("same");
  hChi2FakeAll->Draw("same");

  TLegend *leg = new TLegend(.1, .7, .3, .9);
  leg->AddEntry(hChi2CorrAll, "correct", "l");
  leg->AddEntry(hChi2RelAll, "related", "l");
  leg->AddEntry(hChi2FakeAll, "fake", "l");
  leg->Draw();

}
