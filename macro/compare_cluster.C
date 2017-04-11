#if !defined(__CINT__) || defined(__MAKECINT__)
  #include <TSystem.h>
//  #include <TString.h>
//  #include <TStopwatch.h>
//  #include <TGeoManager.h>
//
//  #include "FairRunSim.h"
//  #include "FairRuntimeDb.h"
//  #include "FairPrimaryGenerator.h"
//  #include "FairBoxGenerator.h"
//  #include "FairParRootFileIo.h"
//
//  #include "DetectorsPassive/Cave.h"
//
//  #include "ITSBase/GeometryTGeo.h"
//  #include "ITSBase/SegmentationPixel.h"
//  #include "ITSSimulation/Detector.h"
//
//  #include "TPCSimulation/Detector.h"
//
  #include "TStyle.h"
  #include "TFile.h"
//  #include "TTree.h"
  #include "TCanvas.h"
  #include "TLegend.h"
  #include "TH1.h"
  #include "TF1.h"
  #include "TPaveStats.h"
  #include "TLatex.h"
  #include "TTreeReader.h"
  #include "TPCSimulation/Cluster.h"
#endif

#include <signal.h>
#include <boost/functional/hash.hpp>
#include <unordered_set>
#include <utility>

#define DIFF_THRESHOLD 0.00001

bool stop_process = false;

void signal_handler(int signal)
{
  switch (signal) {
    case SIGHUP:
    case SIGINT:
    case SIGTERM:
      stop_process = true;
      std::cout << "Received signal: " << strsignal(signal) << ", stopping process..." << std::endl;
      break;
    default:
      std::cout << "Received unhandled signal: " <<  strsignal(signal) << std::endl;
  }
}

struct range {
  int bins;
  double min;
  double max;
};

namespace std {
  template <> struct hash<o2::TPC::Cluster>
    {
      typedef o2::TPC::Cluster     argument_type;
      typedef std::size_t               result_type;

      result_type operator()(const argument_type& t) const
      {
        std::size_t val { 0 };
        boost::hash_combine(val,t.getCRU());
        boost::hash_combine(val,t.getRow());
        boost::hash_combine(val,t.getQ());
        boost::hash_combine(val,t.getQmax());
        boost::hash_combine(val,((float)((int)(t.getPadMean()*10)))/10);
        boost::hash_combine(val,((float)((int)(t.getTimeMean()*10)))/10);
//        boost::hash_combine(val,((float)((int)(t.getPadSigma()*1000)))/1000);
//        boost::hash_combine(val,((float)((int)(t.getTimeSigma()*1000)))/1000);

        return val;
      }
    };
};

void compare_cluster(Int_t nEvents = 10, std::string mcEngine = "TGeant3")
{

  std::string filename = "AliceO2_" + mcEngine + ".clusters_" + std::to_string(nEvents) + "_event.root";

  signal(SIGHUP, signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGINT, signal_handler);

  gStyle->SetOptFit(1111);
  gStyle->SetOptStat(1);
  std::cout << filename << std::endl;

  // open input file
  TFile *file = TFile::Open(filename.c_str());
  if (file == NULL) {
    std::cout << "ERROR: Can't open file " << filename << std::endl;
    return;
  }
  
  std::string treeName = "cbmsim";

  std::vector<std::string> toPlot;
  toPlot.push_back("mCRU");
  toPlot.push_back("mRow");
  toPlot.push_back("mQ");
  toPlot.push_back("mQmax");
  toPlot.push_back("mPadMean");
  toPlot.push_back("mPadSigma");
  toPlot.push_back("mTimeMean");
  toPlot.push_back("mTimeSigma");

  std::vector<struct range> plotRanges;
  plotRanges.push_back({360,-0.5,359.5});
  plotRanges.push_back({18,-0.5,17.5});
  plotRanges.push_back({80,0,8000});
  plotRanges.push_back({260,-0.5,1039.5});
  plotRanges.push_back({138+2+2,-0.5,138+2+2-0.5});
  plotRanges.push_back({200,0,2});
  plotRanges.push_back({300,-0.5,299.5});
  plotRanges.push_back({200,0,2});

  std::vector<std::string> xAxis;
  xAxis.push_back("CRU number");
  xAxis.push_back("row number");
  xAxis.push_back("Q_{tot} [ADC counts]");
  xAxis.push_back("Q_{max} [ADC counts]");
  xAxis.push_back("pad number");
  xAxis.push_back("#sigma_{p}");
  xAxis.push_back("timebin");
  xAxis.push_back("#sigma_{t}");


  TTree* tree = (TTree*)file->Get(treeName.c_str());
  if (tree == NULL) {
    std::cout << "ERROR: can't get tree " << treeName << std::endl;
    return;
  }

  TCanvas *c = new TCanvas("test","test");
  TLegend *leg = new TLegend(0.7,0.8,0.9,0.9);
  TH1D* hist1 = new TH1D;
  TH1D* hist2 = new TH1D;
  TF1 *f1 = new TF1 ("f1","gaus(0)+gaus(3)",0,2);
  f1->SetParNames("Constant_0","Mean_0","Sigma_0","Constant_1","Mean_1","Sigma_1");
  TF1 *f2 = new TF1 ("f2","gaus(0)+gaus(3)",0,2);
  f2->SetParNames("Constant_0","Mean_0","Sigma_0","Constant_1","Mean_1","Sigma_1");

  int count = 0;
  for (std::vector<std::string>::iterator it = toPlot.begin(); it != toPlot.end(); it++) {
    std::cout << "Plotting " << *it << std::endl;

    c->cd();
    c->SetLogy(1);
    std::string hist1name = "hist_" + *it + "_1";
    std::string draw1 = "TPC_HW_Cluster." + *it + " >> " + hist1name;

    tree->Draw(draw1.c_str(),"");
    hist1 = (TH1D*)gPad->GetPrimitive(hist1name.c_str());
    hist1->SetLineColor(2);
    hist1->GetXaxis()->SetTitle(xAxis[count++].c_str());
    hist1->GetYaxis()->SetTitle("#clusters");
    hist1->GetXaxis()->SetDecimals(kTRUE);
    hist1->GetYaxis()->SetDecimals(kTRUE);
//    hist1->SetStats(kFALSE);
//    hist1->SetFillStyle(3345);
//    hist1->SetFillColor(2);

    TPaveStats* stat1;
    TList *listOfLines;
    if (it->find("Sigma") != std::string::npos) {
      c->SetLogy(0);
      gStyle->SetOptStat(0);
      hist1->SetStats(kTRUE);
//      if(it->find("PadSigma") != std::string::npos) {
//        f1->SetParameters(hist1->GetEntries()/40,0.4,0.1,hist1->GetEntries()/40,0.8,0.3);
//        hist1->Fit("f1");
//        hist1->GetFunction("f1")->SetLineColor(2);
//      } else {
        hist1->Fit("gaus");
        hist1->GetFunction("gaus")->SetLineColor(2);
//      }
      c->Update();
      stat1 = (TPaveStats*) hist1->GetListOfFunctions()->FindObject("stats");
      listOfLines = stat1->GetListOfLines();

      stat1->SetX1NDC(0.7);
      stat1->SetX2NDC(0.9);
      stat1->SetY1NDC(0.4);
      stat1->SetY2NDC(0.8);
      stat1->SetTextColor(2);
      c->Modified();
    }
 
    std::string hist2name = "hist_" + *it + "_2";
    std::string draw2 = "TPC_Cluster." + *it + " >> " + hist2name;
    tree->Draw(draw2.c_str(),"","same");
    hist2 = (TH1D*)gPad->GetPrimitive(hist2name.c_str());
    hist2->SetLineColor(4);
//    hist2->SetFillStyle(3354);
//    hist2->SetFillColor(4);

    c->Modified();
    int length[7] =   {5,6,8,5,5,6,0};
    int length_2[7] = {4,6,7,4,5,6,0};
    if (it->find("Sigma") != std::string::npos) {
      gStyle->SetOptStat(0);

      TF1 *f; 
//      if(it->find("PadSigma") != std::string::npos) {
//        f2->SetParameters(hist2->GetEntries()/40,0.4,0.1,hist2->GetEntries()/40,0.8,0.3);
//        hist2->Fit("f2");
//        hist2->GetFunction("f2")->SetLineColor(4);
//        f = f2;
//      } else {
        hist2->Fit("gaus");
        hist2->GetFunction("gaus")->SetLineColor(4);
        f = hist2->GetFunction("gaus");
//      }
      gPad->Update();
      c->Update();

      stat1->SetName("mystats");
//      stat1->InsertLine();
        std::string text = "#chi^{2} / ndf = "; text += std::to_string(f->GetChisquare()).substr(0,std::to_string(f->GetChisquare()).find(".")+2); text += " / "; text += std::to_string(f->GetNDF());
        TLatex *myt = new TLatex(0,0,text.c_str());
        myt->SetTextFont(42);
        myt->SetTextSize(0.02);
        myt->SetTextColor(4);
        listOfLines->Add(myt);
        hist1->SetStats(0);
        c->Modified();

      for (int i = 0; i < f->GetNpar(); i++){
        text = f->GetParName(i); text += " = "; text += std::to_string(f->GetParameter(i)).substr(0,length[i]); text += " #pm "; text += std::to_string(f->GetParError(i)).substr(0,length_2[i]);
        myt = new TLatex(0,0,text.c_str());
        myt->SetTextFont(42);
        myt->SetTextSize(0.02);
        myt->SetTextColor(4);
        listOfLines->Add(myt);
        hist1->SetStats(0);
        c->Modified();
      }
    }


    leg->AddEntry(hist1,"HW-CF","l");
    leg->AddEntry(hist2,"Box-CF","l");
    leg->Draw();
    std::string saveString = filename.substr(0,filename.find(".root")) + "_" + *it+".pdf";

    c->SaveAs(saveString.c_str());

    leg->Clear();
    c->Clear();
    hist1->Clear();
    hist2->Clear();
  }
  delete f1;
  delete f2;
  delete hist1;
  delete hist2;
  delete leg;
  delete c;

  std::cout << std::endl;
  std::cout << "#######################" << std::endl;
  std::cout << "#### Plotting done ####" << std::endl;
  std::cout << "#######################" << std::endl;
  std::cout << std::endl;

  TH1D* hBoth[toPlot.size()];
  TH1D* hBoxOnly[toPlot.size()];
  TH1D* hHwOnly[toPlot.size()];

  TH1D* hDiffPadMean = new TH1D("hPadMeanDiff","hist_diff_PadMean",200,-0.0001,0.0001);
  hDiffPadMean->GetXaxis()->SetTitle("padMean_{box}-padMean_{hw}");
  hDiffPadMean->GetYaxis()->SetTitle("#clusters");
  hDiffPadMean->GetXaxis()->SetDecimals(kTRUE); 
  hDiffPadMean->GetYaxis()->SetDecimals(kTRUE);
  TH1D* hDiffTimeMean = new TH1D("hTimeMeanDiff","hist_diff_TimeMean",200,-0.0001,0.0001);
  hDiffTimeMean->GetXaxis()->SetTitle("timeMean_{box}-timeMean_{hw}");
  hDiffTimeMean->GetYaxis()->SetTitle("#clusters");
  hDiffTimeMean->GetXaxis()->SetDecimals(kTRUE); 
  hDiffTimeMean->GetYaxis()->SetDecimals(kTRUE);

  for (int i = 0; i < toPlot.size(); i++) {
    std::string titleBoth = "hist_"; titleBoth += toPlot[i]; titleBoth += "_both";
    std::string titleBox = "hist_";  titleBox += toPlot[i];  titleBox += "_box";
    std::string titleHw = "hist_";   titleHw += toPlot[i];   titleHw += "_hw";

    std::string nameBoth = "h_"; nameBoth += toPlot[i];
    std::string nameBox = "hB_"; nameBox += toPlot[i];
    std::string nameHw = "hH_";  nameHw += toPlot[i];

    hBoth[i]    = new TH1D(nameBoth.c_str(),titleBoth.c_str(),plotRanges[i].bins, plotRanges[i].min,plotRanges[i].max);
    hBoth[i]->GetXaxis()->SetTitle(xAxis[i].c_str());
    hBoth[i]->GetYaxis()->SetTitle("#clusters");
    hBoth[i]->GetXaxis()->SetDecimals(kTRUE); 
    hBoth[i]->GetYaxis()->SetDecimals(kTRUE);
    hBoxOnly[i] = new TH1D(nameBox.c_str(), titleBox.c_str(), plotRanges[i].bins, plotRanges[i].min,plotRanges[i].max);
    hBoxOnly[i]->GetXaxis()->SetTitle(xAxis[i].c_str());
    hBoxOnly[i]->GetYaxis()->SetTitle("#clusters");
    hBoxOnly[i]->GetXaxis()->SetDecimals(kTRUE); 
    hBoxOnly[i]->GetYaxis()->SetDecimals(kTRUE);
    hHwOnly[i]  = new TH1D(nameHw.c_str(),  titleHw.c_str(),  plotRanges[i].bins, plotRanges[i].min,plotRanges[i].max);
    hHwOnly[i]->GetXaxis()->SetTitle(xAxis[i].c_str());
    hHwOnly[i]->GetYaxis()->SetTitle("#clusters");
    hHwOnly[i]->GetXaxis()->SetDecimals(kTRUE); 
    hHwOnly[i]->GetYaxis()->SetDecimals(kTRUE);
  }

  TTreeReader myReader(treeName.c_str(), file);
  TTreeReaderValue<TClonesArray> bClusters(myReader, "TPC_Cluster");
  TTreeReaderValue<TClonesArray> hClusters(myReader, "TPC_HW_Cluster");

//  std::set<int> duplicateBcluster;
//  std::set<int> duplicateHcluster;
  while (myReader.Next()) {
//    duplicateBcluster.clear();
//    duplicateHcluster.clear();
    std::cout << myReader.GetCurrentEntry()+1 << " / " << myReader.GetEntries(false) << std::endl;
    int nBclusters = bClusters->GetEntries();
    int nHclusters = hClusters->GetEntries();
    std::cout << nBclusters << " Box clusters and " << nHclusters << " HW clusters" << std::endl;

    // remove duplicate clusters in both arrays
    std::unordered_set<o2::TPC::Cluster> uniqueBoxClusters;
    for (int i = 0; i < bClusters->GetEntries(); i++){
//      if (i%100 == 0) std::cout << "Checked " << i << " clusters of " << bClusters->GetEntries() << " for duplicates"  << std::endl;
//      if (duplicateBcluster.find(i) != duplicateBcluster.end()) continue;
      o2::TPC::Cluster* bCluster = dynamic_cast<o2::TPC::Cluster*>(bClusters->At(i));
      std::pair<std::unordered_set<o2::TPC::Cluster>::iterator,bool> ret = uniqueBoxClusters.insert(*bCluster);
      if (ret.second == false) {
        std::cout << *(ret.first) << std::endl << "is very similar to which will be removed" << std::endl;
        std::cout << *bCluster << std::endl << std::endl;
      }
//      for (int j = i+1; j < bClusters->GetEntries(); j++){
//        if (stop_process) return;
//        o2::TPC::Cluster* bCluster2 = dynamic_cast<o2::TPC::Cluster*>(bClusters->At(j));
//
//        if(bCluster->getRow() != bCluster2->getRow()) continue;
//        if(bCluster->getCRU() != bCluster2->getCRU()) continue;
//        if(bCluster->getQmax() != bCluster2->getQmax()) continue;
//        if(bCluster->getQ() != bCluster2->getQ()) continue;
//        if(abs(bCluster->getPadMean() - bCluster2->getPadMean()) > DIFF_THRESHOLD) continue;
//        if(abs(bCluster->getTimeMean() - bCluster2->getTimeMean()) > DIFF_THRESHOLD) continue;
//
//          duplicateBcluster.insert(j);
//          bCluster->Print(std::cout); std::cout << std::endl << "\tis equal to" << std::endl; bCluster2->Print(std::cout); std::cout << std::endl;
////        bCluster->Print(std::cout); std::cout << " is equal to" << std::endl;
////        lastCluster->Print(std::cout); std::cout << std::endl;
////          continue;
//      }
    }
    std::cout << "\t\tRemoved " <<  bClusters->GetEntries() - uniqueBoxClusters.size() << " duplicated Box Clusters" << std::endl << std::endl << std::endl;

    std::unordered_set<o2::TPC::Cluster> uniqueHwClusters;
    for (int i = 0; i < hClusters->GetEntries(); i++){
//      if (i%100 == 0) std::cout << "Checked " << i << " clusters of " << hClusters->GetEntries() << " for duplicates"  << std::endl;
//      if (duplicateHcluster.find(i) != duplicateHcluster.end()) continue;
      o2::TPC::Cluster* hCluster = dynamic_cast<o2::TPC::Cluster*>(hClusters->At(i));
      std::pair<std::unordered_set<o2::TPC::Cluster>::iterator,bool> ret = uniqueHwClusters.insert(*hCluster);
      if (ret.second == false) {
        std::cout << *(ret.first) << std::endl << "is very similar to which will be removed" << std::endl;
        std::cout << *hCluster << std::endl << std::endl;
      }
//      for (int j = i+1; j < hClusters->GetEntries(); j++){
//        if (stop_process) return;
//        o2::TPC::Cluster* hCluster2 = dynamic_cast<o2::TPC::Cluster*>(hClusters->At(j));
//
//        if(hCluster->getRow() != hCluster2->getRow()) continue;
//        if(hCluster->getCRU() != hCluster2->getCRU()) continue;
//        if(hCluster->getQmax() != hCluster2->getQmax()) continue;
//        if(hCluster->getQ() != hCluster2->getQ()) continue;
//        if(abs(hCluster->getPadMean() - hCluster2->getPadMean()) > DIFF_THRESHOLD) continue;
//        if(abs(hCluster->getTimeMean() - hCluster2->getTimeMean()) > DIFF_THRESHOLD) continue;
//
//          duplicateHcluster.insert(j);
//          hCluster->Print(std::cout); std::cout << std::endl << "\tis equal to" << std::endl; hCluster2->Print(std::cout); std::cout << std::endl;
////        hCluster->Print(std::cout); std::cout << " is equal to" << std::endl;
////        lastCluster->Print(std::cout); std::cout << std::endl;
////        continue;
//      }
    }
    std::cout << "\t\tRemoved " <<  hClusters->GetEntries() - uniqueHwClusters.size() << " duplicated HW Clusters" << std::endl << std::endl << std::endl;

//    nBclusters = bClusters->GetEntries();
//    nHclusters = hClusters->GetEntries();

    std::unordered_set<o2::TPC::Cluster> hwClusterFound;
    for (o2::TPC::Cluster bc : uniqueBoxClusters) {
      bool hwFound = false;
      for (o2::TPC::Cluster hc : uniqueHwClusters) {
        if (bc.sim(hc)){
          hDiffPadMean->Fill(bc.getPadMean() - hc.getPadMean());
          hDiffTimeMean->Fill(bc.getTimeMean() - hc.getTimeMean());
        }

//        if (bc == hc) {
        if (bc.sim(hc)){
          if (hwFound == false) {
            hBoth[0]->Fill(bc.getCRU());
            hBoth[1]->Fill(bc.getRow());
            hBoth[2]->Fill(bc.getQ());
            hBoth[3]->Fill(bc.getQmax());
            hBoth[4]->Fill(bc.getPadMean());
            hBoth[5]->Fill(bc.getPadSigma());
            hBoth[6]->Fill(bc.getTimeMean());
            hBoth[7]->Fill(bc.getTimeSigma());
            hwFound = true;
            hwClusterFound.insert(hc);
          } else {
            std::cout << "HW Cluster was already found..." << std::endl;
            std::cout << hc << std::endl;
          }
        }
      }
      if (hwFound == false) {
          hBoxOnly[0]->Fill(bc.getCRU());
          hBoxOnly[1]->Fill(bc.getRow());
          hBoxOnly[2]->Fill(bc.getQ());
          hBoxOnly[3]->Fill(bc.getQmax());
          hBoxOnly[4]->Fill(bc.getPadMean());
          hBoxOnly[5]->Fill(bc.getPadSigma());
          hBoxOnly[6]->Fill(bc.getTimeMean());
          hBoxOnly[7]->Fill(bc.getTimeSigma());
          std::cout << "Found only by Box CF: ";
          std::cout << bc << std::endl;
      }
    }
    for (o2::TPC::Cluster hc : uniqueHwClusters) {
      if (hwClusterFound.find(hc) != hwClusterFound.end()) continue;

      hHwOnly[0]->Fill(hc.getCRU());
      hHwOnly[1]->Fill(hc.getRow());
      hHwOnly[2]->Fill(hc.getQ());
      hHwOnly[3]->Fill(hc.getQmax());
      hHwOnly[4]->Fill(hc.getPadMean());
      hHwOnly[5]->Fill(hc.getPadSigma());
      hHwOnly[6]->Fill(hc.getTimeMean());
      hHwOnly[7]->Fill(hc.getTimeSigma());
      std::cout << "Found only by HW CF: ";
      std::cout << hc << std::endl;
    }

    std::cout << std::endl;
    if (stop_process) break;
  }

//    for (int i = 0; i < nBclusters; i++){
//      if (duplicateBcluster.find(i) != duplicateBcluster.end()) continue;
//      bool hwFound = false;
//      o2::TPC::Cluster* bCluster = dynamic_cast<o2::TPC::Cluster*>(bClusters->At(i));
////      std::cout << "BoxCluster - "; bCluster->Print(std::cout); std::cout << std::endl;
//    
//      for (int j = 0; j < nHclusters; j++){
//        if (duplicateHcluster.find(j) != duplicateHcluster.end()) continue;
//        o2::TPC::Cluster* hwCluster = dynamic_cast<o2::TPC::Cluster*>(hClusters->At(j));
//
//        if(bCluster->getRow() != hwCluster->getRow()) continue;
//        if(bCluster->getCRU() != hwCluster->getCRU()) continue;
//        if(bCluster->getQmax() != hwCluster->getQmax()) continue;
//        if(bCluster->getQ() != hwCluster->getQ()) continue;
//        if(abs(bCluster->getPadMean() - hwCluster->getPadMean()) > DIFF_THRESHOLD) continue;
//        if(abs(bCluster->getTimeMean() - hwCluster->getTimeMean()) > DIFF_THRESHOLD) continue;
//
//          if (hwFound == false) {
//            hBoth[0]->Fill(bCluster->getCRU());
//            hBoth[1]->Fill(bCluster->getRow());
//            hBoth[2]->Fill(bCluster->getQ());
//            hBoth[3]->Fill(bCluster->getQmax());
//            hBoth[4]->Fill(bCluster->getPadMean());
//            hBoth[5]->Fill(bCluster->getPadSigma());
//            hBoth[6]->Fill(bCluster->getTimeMean());
//            hBoth[7]->Fill(bCluster->getTimeSigma());
//            hwFound = true;
//            hwClusterFound.insert(j);
//          } else {
//            std::cout << "HW Cluster was already found..." << std::endl;
//          }
//
//      }
//      if (hwFound == false) {
//          hBoxOnly[0]->Fill(bCluster->getCRU());
//          hBoxOnly[1]->Fill(bCluster->getRow());
//          hBoxOnly[2]->Fill(bCluster->getQ());
//          hBoxOnly[3]->Fill(bCluster->getQmax());
//          hBoxOnly[4]->Fill(bCluster->getPadMean());
//          hBoxOnly[5]->Fill(bCluster->getPadSigma());
//          hBoxOnly[6]->Fill(bCluster->getTimeMean());
//          hBoxOnly[7]->Fill(bCluster->getTimeSigma());
//          std::cout << "Found only by Box CF: ";
//          bCluster->Print(std::cout); std::cout << std::endl;
//      }
//    }
//    for (int j = 0; j < nHclusters; j++){
//      if (duplicateHcluster.find(j) != duplicateHcluster.end()) continue;
//      if (hwClusterFound.find(j) != hwClusterFound.end()) {
//        continue;
//      }
//      o2::TPC::Cluster* hwCluster = dynamic_cast<o2::TPC::Cluster*>(hClusters->At(j));
//      hHwOnly[0]->Fill(hwCluster->getCRU());
//      hHwOnly[1]->Fill(hwCluster->getRow());
//      hHwOnly[2]->Fill(hwCluster->getQ());
//      hHwOnly[3]->Fill(hwCluster->getQmax());
//      hHwOnly[4]->Fill(hwCluster->getPadMean());
//      hHwOnly[5]->Fill(hwCluster->getPadSigma());
//      hHwOnly[6]->Fill(hwCluster->getTimeMean());
//      hHwOnly[7]->Fill(hwCluster->getTimeSigma());
//      std::cout << "Found only by HW CF: ";
//      hwCluster->Print(std::cout); std::cout << std::endl;
//    }
//
//    if (stop_process) break;
//  }

  gStyle->SetOptStat(1);
  c = new TCanvas("test","test");
  std::string saveName;
  for (int i = 0; i < 8; i++) {
    c->cd();
    if (i == 5 || i == 7) { c->SetLogy(0); } else { c->SetLogy(1); }
    hBoth[i]->Draw();
    saveName = "morePlots/"; saveName += hBoth[i]->GetTitle();
    saveName.insert(saveName.find_last_of("_"), "_" + std::to_string(myReader.GetEntries(false)) + "_events"); saveName += ".pdf";
    c->SaveAs(saveName.c_str());
    c->Clear();

    if (i == 5 || i == 7) { c->SetLogy(0); } else { c->SetLogy(1); }
    hBoxOnly[i]->Draw();
    saveName = "morePlots/"; saveName += hBoxOnly[i]->GetTitle();
    saveName.insert(saveName.find_last_of("_"), "_" + std::to_string(myReader.GetEntries(false)) + "_events"); saveName += ".pdf";
    c->SaveAs(saveName.c_str());
    c->Clear();

    if (i == 5 || i == 7) { c->SetLogy(0); } else { c->SetLogy(1); }
    hHwOnly[i]->Draw();
    saveName = "morePlots/"; saveName += hHwOnly[i]->GetTitle();
    saveName.insert(saveName.find_last_of("_"), "_" + std::to_string(myReader.GetEntries(false)) + "_events"); saveName += ".pdf";
    c->SaveAs(saveName.c_str());
    c->Clear();
  }

  c->SetLogy(0);
  hDiffPadMean->Draw();
  saveName = "morePlots/"; saveName += hDiffPadMean->GetTitle();
  saveName += "_"; saveName += std::to_string(myReader.GetEntries(false)); saveName += "_events"; saveName += ".pdf";
  c->SaveAs(saveName.c_str());
  c->Clear();

  hDiffTimeMean->Draw();
  saveName = "morePlots/"; saveName += hDiffTimeMean->GetTitle();
  saveName += "_"; saveName += std::to_string(myReader.GetEntries(false)); saveName += "_events"; saveName += ".pdf";
  c->SaveAs(saveName.c_str());
  c->Clear();
  delete c;

  std::cout << hBoth[0]->GetEntries()       << " clusters found by both CFs" << std::endl;
  std::cout << hBoxOnly[0]->GetEntries()    << " clusters found only by Box CF" << std::endl;
  std::cout << hHwOnly[0]->GetEntries()     << " clusters found only by HW CF" << std::endl;


  file->Close();
  return;
}
