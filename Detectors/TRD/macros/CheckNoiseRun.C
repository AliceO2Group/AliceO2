// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CheckNoiseRun.C
/// \brief Simple macro to check TRD noise runs using CCDB object

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <RDataFrame.h>

#include <map>
#include <string>
#include <ostream>
#include <fairlogger/Logger.h>
#include "TRDBase/Calibrations.h"
#include "DataFormatsTRD/HelperMethods.h"

#endif

using namespace ROOT;

struct ChannelStatusClass {
  TString label;
  int color;
  float minMean;
  float maxMean;
  float minRMS;
  float maxRMS;

  void draw(int color)
  {
    TBox box;
    box.SetFillStyle(0);
    box.SetLineColor(color);
    box.DrawBox(minMean, minRMS, maxMean, maxRMS);

    TText txt;
    txt.SetTextAlign(13);
    txt.SetTextSize(0.04);
    txt.SetTextColor(color);
    txt.DrawText(minMean, maxRMS, label);
  }
};

class ChannelStatusClassifier : public std::vector<ChannelStatusClass>
{

 public:
  ChannelStatusClassifier(vector<ChannelStatusClass> classes = {
                            {"Good", kGreen + 1, 9.0, 10.2, 0.7, 1.8},
                            {"LowNoise", kRed, 9.0, 10.2, 0.0, 0.7},
                            {"Noisy", kRed, 9.0, 10.2, 1.8, 7.0},
                            {"HighMeanRMS", kRed, 8.0, 25.0, 7.0, 30.0},
                            {"HighMean", kRed, 10.5, 25.0, 2.0, 6.0},
                            {"HighBaseline", kRed, 10.5, 520.0, 0.0, 10.0},
                            {"VeryHighNoise", kRed, 0.0, 200.0, 30.0, 180.0},
                            {"Ugly1", kRed, 200., 350., 15.0, 45.0},
                            {"Ugly2", kRed, 200., 350., 45.0, 70.0},
                            {"Ugly3", kRed, 350., 550., 10.0, 25.0},
                            {"Ugly4", kRed, 350., 550., 25.0, 60.0}}) : vector<ChannelStatusClass>(classes)
  {
  }

  // the actual classification method
  int classify(o2::trd::ChannelInfo& c) { return classify(c.getEntries(), c.getMean(), c.getRMS()); }
  int classify(uint64_t nentries, float mean, float rms);

  ROOT::RDF::RNode AddToRDF(ROOT::RDF::RNode df, std::string out = "class", ROOT::RDF::ColumnNames_t in = {"nentries", "mean", "rms"})
  {
    return df.Define(
      out, [this](uint32_t n, float m, float r) { return this->classify(n, m, r); }, in);
  }

  // methods to display classification criteria and results
  TH1* prepareHistogram(TH1* hist);
  void setAxisLabels(TAxis* axis);
};

int ChannelStatusClassifier::classify(uint64_t nentries, float mean, float rms)
{
  if (nentries == 0) {
    return -2;
  }
  if (mean == 10.0 && rms == 0.0) {
    return -1;
  }

  for (int i = 0; i < size(); i++) {
    auto c = this->at(i);
    if (mean > c.minMean && mean < c.maxMean && rms > c.minRMS && rms < c.maxRMS) {
      return i;
    }
  }

  return size(); // default: class not found
}

void ChannelStatusClassifier::setAxisLabels(TAxis* axis)
{
  axis->SetBinLabel(1, "Missing");
  axis->SetBinLabel(2, "Masked");
  axis->SetBinLabel(axis->GetNbins(), "Other");
  for (int i = 0; i < size(); i++) {
    axis->SetBinLabel(3 + i, this->at(i).label);
  }
}

TH1* ChannelStatusClassifier::prepareHistogram(TH1* hist)
{
  setAxisLabels(hist->GetXaxis());

  hist->SetFillColor(kBlue + 1);
  hist->SetBarWidth(0.8);
  hist->SetBarOffset(0.1);
  hist->SetStats(0);

  return hist;
}

o2::trd::ChannelInfoContainer* LoadNoiseCalObject(const long timestamp, char* url = 0)
{
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  // ccdbmgr.clearCache();
  // ccdbmgr.setURL("http://localhost:8080");
  // ccdbmgr.setURL("http://ccdb-test.cern.ch:8080");
  if (url != 0) {
    ccdbmgr.setURL(url);
  }
  ccdbmgr.setTimestamp(timestamp);
  auto calobject = ccdbmgr.get<o2::trd::ChannelInfoContainer>("TRD/Calib/ChannelStatus");
  if (!calobject) {
    LOG(fatal) << "No chamber calibrations returned from CCDB for TRD calibrations";
  }
  return calobject;
}

o2::trd::ChannelInfoContainer* LoadNoiseCalObject(const TString filename)
{
  TFile* infile = new TFile(filename);
  o2::trd::ChannelInfoContainer* calobject;
  infile->GetObject("ccdb_object", calobject);
  return calobject;
}

// Dummy loader to use for cal objects that are already in memory
o2::trd::ChannelInfoContainer* LoadNoiseCalObject(o2::trd::ChannelInfoContainer* calobject)
{
  return calobject;
}

template <typename T>
ROOT::RDF::RNode BuildNoiseDF(T id)
{

  // cout << "Creating RDataFrame" << endl;
  auto df1 = ROOT::RDataFrame(o2::trd::constants::NCHANNELSTOTAL)
               .Define("sector", "rdfentry_ / o2::trd::constants::NCHANNELSPERSECTOR")
               .Define("layer", "(rdfentry_ % o2::trd::constants::NCHANNELSPERSECTOR) / o2::trd::constants::NCHANNELSPERLAYER")
               .Define("row", "(rdfentry_ % o2::trd::constants::NCHANNELSPERLAYER) / o2::trd::constants::NCHANNELSPERROW")
               .Define("col", "rdfentry_ % o2::trd::constants::NCHANNELSPERROW");

  auto calobject = LoadNoiseCalObject(id);

  auto df2 = df1
               .Define("nentries", [calobject](ULong64_t i) { return calobject->getChannel(i).getEntries(); }, {"rdfentry_"})
               .Define("mean", [calobject](ULong64_t i) { return calobject->getChannel(i).getMean(); }, {"rdfentry_"})
               .Define("rms", [calobject](ULong64_t i) { return calobject->getChannel(i).getRMS(); }, {"rdfentry_"});

  return df2;
}

template <typename T>
ROOT::RDF::RNode AddToNoiseDF(ROOT::RDF::RNode df, T id, TString postfix = "1")
{
  auto cal = LoadNoiseCalObject(id);

  return df
    .Define("nentries" + postfix, [cal](ULong64_t i) { return cal->getChannel(i).getEntries(); }, {"rdfentry_"})
    .Define("mean" + postfix, [cal](ULong64_t i) { return cal->getChannel(i).getMean(); }, {"rdfentry_"})
    .Define("rms" + postfix, [cal](ULong64_t i) { return cal->getChannel(i).getRMS(); }, {"rdfentry_"});
}

template <typename T>
T* MakePadAndDraw(Double_t xlow, Double_t ylow, Double_t xup, Double_t yup, ROOT::RDF::RResultPtr<T> h, TString opt = "")
{
  gPad->GetCanvas()->cd();
  TPad* pad = new TPad(h->GetName(), h->GetTitle(), xlow, ylow, xup, yup);
  pad->Draw();
  pad->cd();

  T* hh = (T*)h->Clone();
  Double_t scale = 1. / pad->GetHNDC();
  for (auto* ax : {hh->GetXaxis(), hh->GetYaxis()}) {
    ax->SetLabelSize(scale * ax->GetLabelSize());
    ax->SetTitleSize(scale * ax->GetTitleSize());
  }

  hh->Draw(opt);
  pad->Update();
  return hh;
}

// Set the plotting style for the different histograms in the canvas
void SetStyle(TString style)
{
  // default settings
  gStyle->SetLabelSize(0.02, "X");
  gStyle->SetLabelSize(0.02, "Y");
  gStyle->SetTitleSize(0.02, "X");
  gStyle->SetTitleSize(0.02, "Y");

  gStyle->SetOptStat(0);
  gStyle->SetOptLogx(0);
  gStyle->SetOptLogy(0);
  gStyle->SetOptLogz(0);

  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadLeftMargin(0.15);
  gStyle->SetPadTopMargin(0.02);
  gStyle->SetPadBottomMargin(0.15);

  if (style == "MeanRms1D") {
    gStyle->SetOptStat(1);
    gStyle->SetOptLogy(1);
  }

  if (style == "MeanVsRms") {
    gStyle->SetPadRightMargin(0.1);
  }

  if (style == "ClassStats") {
    gStyle->SetOptLogx(1);
    gStyle->SetPadRightMargin(0.1);
    gStyle->SetPadLeftMargin(0.2);
  }

  if (style == "Map") {
    gStyle->SetPadRightMargin(0.1);
    gStyle->SetPadLeftMargin(0.1);
  }

  if (style == "Comparison") {
    gStyle->SetOptStat(1);
    gStyle->SetOptLogz(1);
    gStyle->SetPadRightMargin(0.1);
    gStyle->SetPadLeftMargin(0.1);
    gStyle->SetPadTopMargin(0.1);
    gStyle->SetPadBottomMargin(0.1);
  }
}

template <typename RDF>
TCanvas* MakeClassSummary(RDF df, ChannelStatusClass cls)
{
  // auto mydf = df.Define("sm", "det/30").Define("ly", "det%6");

  TString id = TString("-") + cls.label;
  auto frame = new TH1F("frame" + id, "frame", 10, 0., 5.);
  auto hMean = df.Histo1D({"Mean" + id, ";Mean;# channels", 100, cls.minMean, cls.maxMean}, "mean");
  auto hRMS = df.Histo1D({"RMS" + id, ";RMS;# channels", 100, cls.minRMS, cls.maxRMS}, "rms");
  auto hMeanRMS = df.Histo2D({"hMeanRMS" + id, ";Mean;RMS", 100, cls.minMean, cls.maxMean, 100, cls.minRMS, cls.maxRMS}, "mean", "rms");
  auto hClass = df.Histo1D("class");

  auto hGlobalPos = df.Histo2D({"GlobalPos" + id, ";Sector;Layer", 18, -0.5, 17.5, 6, -0.5, 5.5}, "sector", "layer");
  auto hLayerPos = df.Histo2D({"LayerPos" + id, ";Pad row;ADC channel column", 76, -0.5, 75.5, 168, -0.5, 167.5}, "row", "col");

  auto cnv = new TCanvas("ClassSummary-" + cls.label, "Class Summary - " + cls.label, 1500, 1000);
  TText txt;
  txt.SetTextSize(0.05);
  txt.DrawText(0.02, 0.9, cls.label);
  txt.SetTextSize(0.02);
  txt.DrawText(0.02, 0.85, Form("%.1f < mean < %.1f", cls.minMean, cls.maxMean));
  txt.DrawText(0.02, 0.80, Form("%.1f < rms < %.1f", cls.minRMS, cls.maxRMS));

  SetStyle("MeanRms1D");
  MakePadAndDraw(0.7, 0.3, 1.0, 0.6, hMean);
  MakePadAndDraw(0.7, 0.0, 1.0, 0.3, hRMS);

  SetStyle("MeanVsRms");
  MakePadAndDraw(0.7, 0.6, 1.0, 1.0, hMeanRMS, "colz");

  SetStyle("Map");
  gStyle->SetPadRightMargin(0.05);
  MakePadAndDraw(0.2, 0.65, 0.7, 1.0, hGlobalPos, "col")->DrawClone("text45,same");

  gStyle->SetPadRightMargin(0.1);
  MakePadAndDraw(0.0, 0.0, 0.7, 0.65, hLayerPos, "colz");

  return cnv;
}

TCanvas* MakeRunSummary(ROOT::RDF::RNode df_all, ChannelStatusClassifier& classifier)
{

  auto df = df_all.Filter("class >= 0");

  auto hMeanRms1 = df.Histo2D({"MeanRms1", ";Mean;RMS", 100, 0.0, 1024.0, 100, 0.0, 200.}, "mean", "rms");
  auto hMeanRms2 = df.Histo2D({"MeanRms2", ";Mean;RMS", 100, 0.0, 30.0, 100, 0.0, 30.0}, "mean", "rms");
  auto hMeanRms3 = df.Histo2D({"MeanRms2", ";Mean;RMS", 100, 8.5, 11.0, 100, 0.0, 6.0}, "mean", "rms");

  auto hMean = df.Histo1D({"Mean", ";Mean;#channels", 200, 8.5, 11.0}, "mean");
  auto hRMS = df.Histo1D({"RMS", ";RMS;#channels", 200, 0.4, 3.0}, "rms");

  int nclasses = classifier.size();
  auto hClasses = df_all.Histo1D({"Classes", "", nclasses + 3, -2.5, nclasses + 0.5}, "class");

  auto maxmean = df.Max("mean");
  auto maxrms = df.Max("rms");

  hMeanRms1->GetXaxis()->SetRangeUser(0.0, *maxmean * 1.1);
  hMeanRms1->GetYaxis()->SetRangeUser(0.0, *maxrms * 1.1);

  auto cnv = new TCanvas("NoiseRunSummary", "Noise Run Summary", 1500, 1000);

  SetStyle("MeanRms1D");
  MakePadAndDraw(0.0, 0.5, 0.33, 1.0, hRMS);
  MakePadAndDraw(0.34, 0.5, 0.66, 1.0, hMean);

  SetStyle("ClassStats");
  classifier.prepareHistogram(hClasses.GetPtr());
  MakePadAndDraw(0.67, 0.5, 1.00, 1.0, hClasses, "hbar");

  SetStyle("MeanVsRms");
  auto h2 = MakePadAndDraw(0.00, 0.0, 0.33, 0.5, hMeanRms3, "colz");
  classifier[0].draw(kGreen);
  classifier[1].draw(kRed);
  classifier[2].draw(kRed);

  auto h3 = MakePadAndDraw(0.34, 0.0, 0.66, 0.5, hMeanRms2, "colz");
  classifier[3].draw(kRed);
  classifier[4].draw(kRed);

  auto h4 = MakePadAndDraw(0.67, 0.0, 1.00, 0.5, hMeanRms1, "colz");
  for (int i = 5; i < classifier.size(); i++) {
    classifier[i].draw(kRed);
  }

  return cnv;
}

TCanvas* MakeLayerPlots(ROOT::RDF::RNode df_in, int sector, int layer, ChannelStatusClassifier& classifier)
{

  auto df = df_in.Filter(Form("sector == %d && layer == %d", sector, layer));
  auto df2 = df.Filter("class >= 0");

  TString id = TString::Format("-SM%02dL%d", sector, layer);
  auto hMean = df2.Histo1D({"Mean" + id, ";Mean;# channels", 150, 8.0, 11.0}, "mean");
  auto hRMS = df2.Histo1D({"RMS" + id, ";RMS;# channels", 250, 0.0, 5.0}, "rms");

  int nclasses = classifier.size();
  auto hClasses = df.Histo1D({"Classes" + id, "", nclasses + 3, -2.5, nclasses + 0.5}, "class");

  auto hMeanMap = df.Histo2D({"MeanMap" + id, ";Pad row;ADC channel column", 76, -0.5, 75.5, 168, -0.5, 167.5}, "row", "col", "mean");
  auto hRMSMap = df.Histo2D({"RMSMap" + id, ";Pad row;ADC channel column", 76, -0.5, 75.5, 168, -0.5, 167.5}, "row", "col", "rms");

  auto cnv = new TCanvas("LayerPlots" + id, "Layer Plots " + id, 1500, 1000);

  SetStyle("MeanRms1D");
  MakePadAndDraw(0.0, 0.7, 0.3, 1.0, hMean);
  MakePadAndDraw(0.0, 0.4, 0.3, 0.7, hRMS);

  SetStyle("ClassStats");
  auto h1 = MakePadAndDraw(0.0, 0.0, 0.3, 0.4, hClasses, "hbar");
  classifier.prepareHistogram(h1);

  SetStyle("Map");

  hMeanMap->GetZaxis()->SetRangeUser(8.5, 10.5);
  MakePadAndDraw(0.3, 0.5, 1.0, 1.0, hMeanMap, "colz");

  hRMSMap->GetZaxis()->SetRangeUser(0.3, 2.0);
  hRMSMap->GetZaxis()->SetTitle("RMS");
  MakePadAndDraw(0.3, 0.0, 1.0, 0.5, hRMSMap, "colz");

  return cnv;
}

template <typename T1, typename T2>
void CompareRuns(T1 id1, T2 id2, const char* label1 = "old", const char* label2 = "new")
{
  auto df1 = BuildNoiseDF(id1);
  auto df2 = AddToNoiseDF(df1, id2, "1");

  // auto df1 = BuildNoiseDF(1679064254531);
  // auto df1 = BuildNoiseDF(1679064254529);
  // auto df1 = BuildNoiseDF(1681210184624);
  // auto df2 = AddToNoiseDF(df1, "~/Downloads/o2-trd-ChannelInfoContainer_1680185087230.root", "1");

  ChannelStatusClassifier classifier;
  auto df3 = classifier.AddToRDF(df2);
  auto df = classifier.AddToRDF(df3, "class1", {"nentries1", "mean1", "rms1"});

  auto dfclean = df.Filter("class>=0 && class1>=0");

  auto hr = dfclean.Histo2D({"hr", Form("Comparison - RMS;rms: %s;rms: %s", label1, label2), 100, 0.0, 5.0, 100, 0.0, 5.0}, "rms", "rms1");
  auto hm = dfclean.Histo2D({"hm", Form("Comparison - Mean;mean: %s;mean: %s", label1, label2), 100, 8.5, 11.0, 100, 8.5, 11.0}, "mean", "mean1");

  int nc = classifier.size();
  auto hc = df.Histo2D({"hc", Form("Comparison of channel status classes;channel class: %s;channel class: %s", label1, label2), nc + 3, -2.5, nc + 0.5, nc + 3, -2.5, nc + 0.5}, "class", "class1");

  SetStyle("Comparison");
  new TCanvas(Form("Compare_RMS_%s_%s", label1, label2), Form("Compare_RMS_%s_%s", label1, label2), 1000, 1000);
  hr->DrawClone("colz");

  new TCanvas(Form("Compare_Mean_%s_%s", label1, label2), Form("Compare_Mean_%s_%s", label1, label2), 1000, 1000);
  hm->DrawClone("colz");

  new TCanvas(Form("Compare_Class_%s_%s", label1, label2), Form("Compare_Class_%s_%s", label1, label2), 1500, 1000);
  classifier.setAxisLabels(hc->GetXaxis());
  classifier.setAxisLabels(hc->GetYaxis());
  gPad->SetLeftMargin(0.20);
  gPad->SetBottomMargin(0.15);
  hc->GetXaxis()->SetTitleOffset(1.6);
  hc->SetStats(0);
  hc->DrawClone("colz");
  hc->DrawClone("text,same");
}

void CheckNoiseRun()
{
  cout << "usage:" << endl;
  cout << "   root /path/to/CheckNoiseRun.C" << endl;
  cout << "   root '/path/to/CheckNoiseRun(<timestamp>)'" << endl;
  cout << "   root '/path/to/CheckNoiseRun(\"/path/to/o2-trd-ChannelInfoContainer_<timestamp>.root\")'" << endl;

  cout << endl
       << "Retrieve a list of channel status objects from CCDB:" << endl
       << "> curl  http://alice-ccdb.cern.ch/browse/TRD/Calib/ChannelStatus/" << endl
       << "or from the ROOT prompt: " << endl
       << "root [] .! curl  http://alice-ccdb.cern.ch/browse/TRD/Calib/ChannelStatus/" << endl;

  cout << endl
       << "Suggested commands:" << endl
       << "  run534642 = LoadNoiseCalObject(1681740512656)" << endl
       << "  run534640 = LoadNoiseCalObject(1681739312939)" << endl
       << "  run533031 = LoadNoiseCalObject(1681210184624, \"http://ccdb-test.cern.ch:8080\")" << endl;
}

template <typename T>
void CheckNoiseRun(T id, bool show_classes = false)
{
  auto df1 = BuildNoiseDF(id);
  // auto df1 = BuildNoiseDF(1681210184624);
  // auto df1 = BuildNoiseDF("~/Downloads/o2-trd-ChannelInfoContainer_1680185087230.root");

  ChannelStatusClassifier classifier;
  auto df = classifier.AddToRDF(df1);

  MakeRunSummary(df, classifier);

  if (show_classes) {
    // cout << "Display summary" << endl;
    for (size_t i = 0; i < classifier.size(); i++) {
      MakeClassSummary(df.Filter(Form("class==%zu", i)), classifier[i]);
    }
    MakeClassSummary(df.Filter("class==-2"), {"Missing", kBlue, 0.0, 20.0, 0.0, 2.0});
    MakeClassSummary(df.Filter("class==-1"), {"Masked", kBlue, 9.0, 11.0, -0.1, 0.1});
    MakeClassSummary(df.Filter(Form("class==%zu", classifier.size())), {"Other", kRed, 0.0, 1024.0, 0.0, 1024.0});
  }
}
