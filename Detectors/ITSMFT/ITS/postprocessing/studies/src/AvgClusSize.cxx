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

#include "ITSStudies/AvgClusSize.h"

#include "DataFormatsITS/TrackITS.h"
// #include "Framework/CCDBParamSpec.h"
#include "Framework/Task.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/V0.h"
#include "CommonUtils/TreeStreamRedirector.h"
// #include "ITStracking/IOUtils.h"
#include "DataFormatsParameters/GRPObject.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "ITSBase/GeometryTGeo.h"

#include <TH1F.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TFitResultPtr.h>
#include <TFitResult.h>
#include <TLine.h>
#include <numeric>
#include <TStyle.h>

namespace o2
{
namespace its
{
namespace study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using GTrackID = o2::dataformats::GlobalTrackID;
using V0 = o2::dataformats::V0;
using ITSCluster = o2::BaseCluster<float>;
using mask_t = o2::dataformats::GlobalTrackID::mask_t;
using AvgClusSizeStudy = o2::its::study::AvgClusSizeStudy;
using Track = o2::track::TrackParCov;
using MCLabel = o2::MCCompLabel;

void AvgClusSizeStudy::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);

  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mOutName.c_str(), "recreate");

  mMassSpectrumFull = std::make_unique<THStack>("V0", "V0 mass spectrum; MeV");                                         // auto-set axis ranges
  mMassSpectrumFullNC = std::make_unique<TH1F>("V0_nc", "no cuts; MeV", 100, 1, -1);                                         // auto-set axis ranges
  mMassSpectrumFullC = std::make_unique<TH1F>("V0_c", "cut; MeV", 100, 1, -1);                                               // auto-set axis ranges
  mMassSpectrumK0s = std::make_unique<THStack>("K0s", "'K0' mass spectrum; MeV");                                     // set axis ranges near K0short mass
  mMassSpectrumK0sNC = std::make_unique<TH1F>("K0s_nc", "no cuts; MeV", 15, 475, 525);                                     // set axis ranges near K0short mass
  mMassSpectrumK0sC = std::make_unique<TH1F>("K0s_c", "cut; MeV", 15, 475, 525);                                           // set axis ranges near K0short mass
  mAvgClusSize = std::make_unique<THStack>("avg_clus_size", "Average cluster size per track; pixels / cluster / track"); // auto-set axis ranges
  mAvgClusSizeNC = std::make_unique<TH1F>("avg_clus_size_NC", "no cuts", 20, 1, -1); // auto-set axis ranges
  mAvgClusSizeC = std::make_unique<TH1F>("avg_clus_size_C", "cut", 20, 1, -1);       // auto-set axis ranges
  mCosPA = std::make_unique<TH1F>("CosPA", "cos(PA)", 100, 1, -1);                                                                              // auto-set axis ranges
  mR = std::make_unique<TH1F>("R", "R", 40, 1, -1);                                                                                             // auto-set axis ranges
  mDCA = std::make_unique<TH1F>("DCA", "DCA", 40, 1, -1);                                                                                       // auto-set axis ranges
  mEtaNC = std::make_unique<TH1F>("etaNC", "no cuts", 30, 1, -1);                                                                                       // auto-set axis ranges
  mEtaC = std::make_unique<TH1F>("etaC", "cut", 30, 1, -1);                                                                                       // auto-set axis ranges
  // mAvgClusSizeCEtaBin = std::make_unique<std::vector<TH1F>>();

  mAvgClusSizeCEta = std::make_unique<THStack>("avg_clus_size_eta", "Average cluster size per track; pixels / cluster / track"); // auto-set axis ranges
  double binWidth = (etaMax - etaMin) / (double) etaNBins;
  etaBinEdges = std::vector<double>(etaNBins);
  for (int i = 0; i < etaNBins; i++) {
    etaBinEdges[i] = etaMin + (binWidth * (i + 1));
    mAvgClusSizeCEtaVec.push_back(std::make_unique<TH1F>(Form("avg_clus_size_%i", i), Form("%.2f < #eta < %.2f", etaBinEdges[i] - binWidth, etaBinEdges[i]), 15, 1, -1));
    mAvgClusSizeCEtaVec[i]->SetDirectory(nullptr);
    mAvgClusSizeCEta->Add(mAvgClusSizeCEtaVec[i].get());
  }
  // for (double edge : etaBinEdges) LOGP(info, "BINEDGES bin entry is {}", edge);
  // etaMin{-1.5};
  // etaMax{1.5};
  // etaNBins{5};

  // mMassSpectrumFull->SetDirectory(nullptr);
  mMassSpectrumFullNC->SetDirectory(nullptr);
  mMassSpectrumFullC->SetDirectory(nullptr);
  // mMassSpectrumK0s->SetDirectory(nullptr);
  mMassSpectrumK0sNC->SetDirectory(nullptr);
  mMassSpectrumK0sC->SetDirectory(nullptr);
  mAvgClusSizeNC->SetDirectory(nullptr);
  mAvgClusSizeC->SetDirectory(nullptr);
  mCosPA->SetDirectory(nullptr);
  mR->SetDirectory(nullptr);
  mDCA->SetDirectory(nullptr);
  mEtaNC->SetDirectory(nullptr);
  mEtaC->SetDirectory(nullptr);

  globalNClusters = 0;
  globalNPixels = 0;

  LOGP(important, "Cluster size study initialized.");
}

void AvgClusSizeStudy::run(ProcessingContext& pc)
{
  LOGP(important, "Starting run.");
  // auto geom = o2::its::GeometryTGeo::Instance();
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  LOGP(important, "Collected data.");
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void AvgClusSizeStudy::getClusterSizes(std::vector<int>& clusSizeVec, const gsl::span<const o2::itsmft::CompClusterExt> ITSclus, gsl::span<const unsigned char>::iterator& pattIt, const o2::itsmft::TopologyDictionary* mdict)
{
  for (unsigned int iClus{0}; iClus < ITSclus.size(); ++iClus) {
    auto& clus = ITSclus[iClus];
    auto pattID = clus.getPatternID();
    int npix;
    o2::itsmft::ClusterPattern patt;

    if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict->isGroup(pattID)) {
      patt.acquirePattern(pattIt);
      npix = patt.getNPixels();
    } else {

      npix = mdict->getNpixels(pattID);
      patt = mdict->getPattern(pattID);
    }
    clusSizeVec[iClus] = npix;
  }
}

void AvgClusSizeStudy::loadData(o2::globaltracking::RecoContainer& recoData)
{
  LOGP(important, "About to load data.");
  mInputITSidxs = recoData.getITSTracksClusterRefs();

  auto compClus = recoData.getITSClusters();
  auto clusPatt = recoData.getITSClustersPatterns();
  auto pattIt = clusPatt.begin();
  mInputITSclusters.reserve(compClus.size());
  mInputClusterSizes.resize(compClus.size());
  o2::its::ioutils::convertCompactClusters(compClus, pattIt, mInputITSclusters, mDict);
  auto pattIt2 = clusPatt.begin();
  getClusterSizes(mInputClusterSizes, compClus, pattIt2, mDict);
}

void AvgClusSizeStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  // to get ITS-TPC tracks, we would call recoData.getTPCITSTracks() (line 534) of RecoContainer.h
  // Alternatively, we should just use the track masks to do this in general... but how?
  // I feel like recoData should already contain only the tracks we need, given that we applied the masks at DataRequest time...
  // but clearly that is not the case? or maybe i'm an idiot
  loadData(recoData);
  auto V0s = recoData.getV0s();
  LOGP(important, "LENLEN V0 length {}", V0s.size());
  auto trks = recoData.getITSTracks();
  LOGP(important, "LENLEN trks size {}", trks.size());
  auto labs = recoData.getITSTracksMCLabels();
  LOGP(important, "LENLEN labs size {}", labs.size());
  int totalSize;
  double_t avgClusterSize;
  double_t mass;
  double_t dca;
  o2::its::TrackITS ITStrack;
  double_t cosPA;
  double_t R;
  Track daughter1;
  double_t d1DCA;
  MCLabel V0lab, d1lab, d2lab;
  double_t V0eta;    // d1eta, d2eta;
  double limEta = 5; // we want to scan on this eta to see cluster size

  for (V0 v0 : V0s) {
    // d1eta = v0.getProng(0).getEta();
    // d2eta = v0.getProng(1).getEta();
    V0eta = v0.getEta();
    // d1DCA = daughter1.getDCA();
    dca = v0.getDCA();
    cosPA = v0.getCosPA();

    mass = std::sqrt(v0.calcMass2()) * 1000; // convert mass to MeV
    R = std::sqrt(v0.calcR2());              // gives distance from pvertex to V0 in centimeters (?)
    ITStrack = recoData.getITSTrack(v0.getVertexID());
    V0lab = labs[v0.getVertexID()];
    // if (v0.getVertexID() > 480) LOGP(info, "IDID vertexID is above 480");
    if (v0.getVertexID() > 33020) LOGP(info, "IDID vertexID is above 33020");
    if (V0lab.getTrackID() == v0.getVertexID()) LOGP(info, "IDID tracks match");
    else LOGP(info, "IDID tracks dont match");

    auto firstClus = ITStrack.getFirstClusterEntry();
    auto ncl = ITStrack.getNumberOfClusters();
    LOGP(important, "PIDPID pid is {}", ITStrack.getPID());
    LOGP(important, "PIDPID pid is {}", V0lab);
    totalSize = 0;
    for (int icl = 0; icl < ncl; icl++) {
      totalSize += mInputClusterSizes[mInputITSidxs[firstClus + icl]];
      globalNPixels += mInputClusterSizes[mInputITSidxs[firstClus + icl]];
    }
    globalNClusters += ncl;
    avgClusterSize = (double_t)totalSize / (double_t)ncl;

    mAvgClusSizeNC->Fill(avgClusterSize);
    mCosPA->Fill(cosPA);
    mMassSpectrumFullNC->Fill(mass);
    mMassSpectrumK0sNC->Fill(mass);
    mEtaNC->Fill(V0eta);
    mR->Fill(R);
    mDCA->Fill(dca);
    // innermost layer of ITS lies at 2.3cm, idk where outer layer is :^)
    if (cosPA > 0.995 && R < 5.4 && R > 0.5 && dca < 0.2) {
      mMassSpectrumK0sC->Fill(mass);
      mMassSpectrumFullC->Fill(mass);
      mAvgClusSizeC->Fill(avgClusterSize);
      mEtaC->Fill(V0eta);
      if (V0eta > etaMin && V0eta < etaMax) findEtaBin(V0eta, avgClusterSize, 0);
      // mAvgClusSizeCEtaVec[0]->Fill(avgClusterSize);
    }
  }
  // cut on before the firsrt layer for V0 so we can get a solid 7 cluster track for daughter pions
  // if we don't have enough statistics, we can cut even harsher on cosPA and inject more statistics
  // next TODO: use all cut statistics, see the full range for this cut, and now make the avg cluster size distributions over various eta bins
  // would be nice to print the cut on the graph but oh well (how to get params out of the fit result?)
  // try a fix for the input data and see if it works (making sure the mc input isn't set to ALL but is just its)
  // try seeing what happens with its-tpc matched tracks
}

void AvgClusSizeStudy::findEtaBin(double eta, double clusSize, int i) {
  if (eta < etaBinEdges[i]) {
    mAvgClusSizeCEtaVec[i]->Fill(clusSize);
    // return i;
  }
  else {
    findEtaBin(eta, clusSize, i+1);
  }
}


void AvgClusSizeStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  LOGP(important, "Begin update.");
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
  }
}

void AvgClusSizeStudy::saveHistograms() {

  // mMassSpectrumK0sCut->Sumw2();
  mAvgClusSizeC->SetLineColor(kRed);
  mMassSpectrumFullC->SetLineColor(kRed);
  mMassSpectrumK0sC->SetLineColor(kRed);
  int globalAvgClusSize = (double) globalNPixels / (double) globalNClusters;
  mMassSpectrumFull->Add(mMassSpectrumFullC.get());
  mMassSpectrumFull->Add(mMassSpectrumFullNC.get());
  mMassSpectrumK0s->Add(mMassSpectrumK0sC.get());
  mMassSpectrumK0s->Add(mMassSpectrumK0sNC.get());
  mAvgClusSize->Add(mAvgClusSizeC.get());
  mAvgClusSize->Add(mAvgClusSizeNC.get());

  // mAvgClusSizeCEta->Add(mAvgClusSizeCEtaVec[0].get());

  mDBGOut.reset();
  TFile fout(mOutName.c_str(), "RECREATE");
  fout.WriteTObject(mMassSpectrumFullNC.get());
  fout.WriteTObject(mMassSpectrumFullC.get());
  fout.WriteTObject(mMassSpectrumK0sNC.get());
  fout.WriteTObject(mMassSpectrumK0sC.get());
  fout.WriteTObject(mAvgClusSize.get());
  fout.WriteTObject(mCosPA.get());
  fout.WriteTObject(mR.get());
  fout.WriteTObject(mDCA.get());
  fout.WriteTObject(mEtaNC.get());
  fout.WriteTObject(mEtaC.get());
  LOGP(important, "Stored full mass spectrum histogram {} into {}", mMassSpectrumFullNC->GetName(), mOutName.c_str());
  LOGP(important, "Stored uncut narrow mass spectrum histogram {} into {}", mMassSpectrumK0sNC->GetName(), mOutName.c_str());
  LOGP(important, "Stored cut narrow mass spectrum histogram {} into {}", mMassSpectrumK0sC->GetName(), mOutName.c_str());
  LOGP(important, "Stored cluster size histogram {} into {}", mAvgClusSize->GetName(), mOutName.c_str());
  LOGP(important, "Stored pointing angle histogram {} into {}", mCosPA->GetName(), mOutName.c_str());
  LOGP(important, "Stored R histogram {} into {}", mR->GetName(), mOutName.c_str());
  LOGP(important, "Stored DCA histogram {} into {}", mDCA->GetName(), mOutName.c_str());
  LOGP(important, "Stored eta histogram {} into {}", mEtaNC->GetName(), mOutName.c_str());
  // can i store the mass data in a tree?? check itsoffsstudy.cxx 'itstof'

  gStyle->SetPalette(kRainbow);
  TCanvas* c1 = new TCanvas();
  mMassSpectrumFull->Draw();
  c1->Print("massSpectrumFull.png");
  TCanvas* c2 = new TCanvas();
  mMassSpectrumK0s->Draw("E");
  c2->BuildLegend();
  c2->Print("massSpectrumK0s.png");
  TCanvas* c3 = new TCanvas();
  mAvgClusSize->Draw();
  TLine * globalAvg = new TLine(globalAvgClusSize, 0, globalAvgClusSize, mAvgClusSizeNC->GetMaximum());
  globalAvg->Draw();
  c3->Print("clusSize.png");
  TCanvas* c4 = new TCanvas();
  mCosPA->Draw();
  c4->Print("cosPA.png");
  TCanvas* c6 = new TCanvas();
  mR->Draw();
  c6->Print("mR.png");
  TCanvas* c7 = new TCanvas();
  mDCA->Draw();
  c7->Print("mDCA.png");
  TCanvas* c8 = new TCanvas();
  mEtaNC->Draw();
  c8->Print("mEtaNC.png");
  TCanvas* c9 = new TCanvas();
  mEtaC->Draw();
  c9->Print("mEtaC.png");
  TCanvas* c10 = new TCanvas();
  mAvgClusSizeCEta->Draw("plc L NOSTACK HIST");
  c10->BuildLegend(0.6, 0.6, 0.8, 0.8);
  c10->Print("clusSizeEta.png");
  for (int i = 0; i < etaNBins; i++) {
    mAvgClusSizeCEtaVec[i]->Scale( 1./mAvgClusSizeCEtaVec[i]->Integral("width"));
  }
  TCanvas* c11 = new TCanvas();
  mAvgClusSizeCEta->Draw("plc L NOSTACK HIST");
  mAvgClusSizeCEta->SetTitle("Average cluster size per track (normed)");
  c11->BuildLegend(0.6, 0.6, 0.8, 0.8);
  c11->Print("clusSizeEtaNormed.png");
  // TCanvas* c9 = new TCanvas();
  // mMassSpectrumFullC->Draw();
  // c9->Print("massSpectrumFullC.png");
  // TCanvas* c10 = new TCanvas();
  // mMassSpectrumK0sC->Draw("E");
  // c10->Print("massSpectrumK0sC.png");

  fout.Close();
}

void AvgClusSizeStudy::fitMassSpectrum() {
  TF1* gaus = new TF1("gaus", "gaus", 485, 505);
  TFitResultPtr fit = mMassSpectrumK0sC->Fit("gaus", "S", "", 480, 510);
  fit->Print();
}

void AvgClusSizeStudy::endOfStream(EndOfStreamContext& ec)
{
  fitMassSpectrum();
  // formatHistograms();
  // plotHistograms();
  saveHistograms();
}

void AvgClusSizeStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  LOGP(important, "CLUSIZE ccdb finalise");
  {
    if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      LOGP(important, "TESTING if has gone through, not updating");
      return;
    }
    // o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
    if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
      LOGP(important, "TESTING Cluster dictionary updated");
      setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
      return;
    }
  }
}

DataProcessorSpec getAvgClusSizeStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC)
{
  LOGP(info, "AT THIS POINT USEMC IS {}", useMC);
  LOGP(info, "USEMC333 1st param is {}", srcTracksMask.to_string());
  LOGP(info, "USEMC333 2nd param is {}", srcClustersMask.to_string());
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  // dataRequest->requestTracks(srcTracksMask, useMC);
  dataRequest->requestITSTracks(useMC);
  // dataRequest->requestClusters(srcClustersMask, useMC);
  dataRequest->requestITSClusters(useMC);
  dataRequest->requestSecondaryVertices(useMC);
  LOGP(info, "USEMC {}", useMC);

  // TODO: is this even right???? need to figure out wtf these params mean
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              false,                             // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  LOGP(important, "CLUSIZE datareq done");
  return DataProcessorSpec{
    "its-study-AvgClusSize",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<AvgClusSizeStudy>(dataRequest, ggRequest, useMC)},
    Options{}};
}
} // namespace study
} // namespace its
} // namespace o2