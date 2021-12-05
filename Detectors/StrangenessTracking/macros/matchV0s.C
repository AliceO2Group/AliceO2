#if !defined(CLING) || defined(ROOTCLING)
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITStracking/IOUtils.h"

#include <gsl/gsl>
#include <TLorentzVector.h>
#include "TCanvas.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2D.h"

#include "TMath.h"
#include "TString.h"
#include "TTree.h"
#include "TLegend.h"
#include "CommonDataFormat/RangeReference.h"
#include "DetectorsVertexing/DCAFitterN.h"

#include "HyperTracker.h"

#endif

using GIndex = o2::dataformats::VtxTrackIndex;
using V0 = o2::dataformats::V0;
using MCTrack = o2::MCTrack;
using Cascade = o2::dataformats::Cascade;
using RRef = o2::dataformats::RangeReference<int, int>;
using VBracket = o2::math_utils::Bracket<int>;
using namespace o2::itsmft;
using CompClusterExt = o2::itsmft::CompClusterExt;
using ITSCluster = o2::BaseCluster<float>;
using Vec3 = ROOT::Math::SVector<double, 3>;
using hyperTracker = o2::tracking::hyperTracker;

const int motherPDG = 1010010030;
const int firstDaughterPDG = 1000020030;
const int secondDaughterPDG = -211;

std::vector<std::array<int, 2>> matchV0stoMC(const std::vector<std::vector<o2::MCTrack>>& mcTracksMatrix, std::map<std::string, std::vector<o2::MCCompLabel>*>& map, std::vector<V0>* v0vec);
std::array<int, 2> matchITStracktoMC(const std::vector<std::vector<o2::MCTrack>>& mcTracksMatrix, o2::MCCompLabel ITSlabel);
std::vector<ITSCluster> getTrackClusters(const o2::its::TrackITS& ITStrack, const std::vector<ITSCluster>& ITSClustersArray, std::vector<int>* ITSTrackClusIdx);

void matchV0s()
{
  // Output Histograms
  TH1D* hChi2Sgn = new TH1D("Chi2 Signal", "; #chi^{2}; Counts", 102, -2, 100);
  TH1D* hChi2Bkg = new TH1D("Chi2 background", "; #chi^{2} (90 is default for overflows and not propagated); Counts", 102, -2, 100);
  TH1D* hSigBkg = new TH1D("Hypertracker eff", "; ; Efficiency", 2, 0, 2);
  TH2D* hPtResBef = new TH2D("pT resolution before hypertracking", "; #it{p}_{T}^{gen} (GeV); (#it{p}_{T}^{gen} - #it{p}_{T}^{rec})/#it{p}_{T}^{gen}; Counts", 20, 0, 10, 20, -1, 1);
  TH2D* hPtResAft = new TH2D("pT resolution before hypertracking", "; #it{p}_{T}^{gen} (GeV); (#it{p}_{T}^{gen} - #it{p}_{T}^{rec})/#it{p}_{T}^{gen}; Counts", 20, 0, 10, 20, -1, 1);

  // Files
  auto fMCTracks = TFile::Open("sgn_Kine.root");
  auto fSecondaries = TFile::Open("o2_secondary_vertex.root");
  auto fITSTPC = TFile::Open("o2match_itstpc.root");
  auto fTPCTOF = TFile::Open("o2match_tof_tpc.root");
  auto fITSTPCTOF = TFile::Open("o2match_tof_itstpc.root");

  auto fITS = TFile::Open("o2trac_its.root");
  auto fITSclus = TFile::Open("o2clus_its.root");

  // Geometry
  o2::base::GeometryManager::loadGeometry("");

  // Trees
  auto treeMCTracks = (TTree*)fMCTracks->Get("o2sim");
  auto treeSecondaries = (TTree*)fSecondaries->Get("o2sim");
  auto treeITSTPC = (TTree*)fITSTPC->Get("matchTPCITS");
  auto treeITSTPCTOF = (TTree*)fITSTPCTOF->Get("matchTOF");
  auto treeTPCTOF = (TTree*)fTPCTOF->Get("matchTOF");

  auto treeITS = (TTree*)fITS->Get("o2sim");
  auto treeITSclus = (TTree*)fITSclus->Get("o2sim");

  // Topology dictionary
  o2::itsmft::TopologyDictionary mdict;
  mdict.readFromFile(o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS));

  // Tracks
  std::vector<o2::MCTrack>* MCtracks = nullptr;
  std::vector<V0>* v0vec = nullptr;
  std::vector<o2::its::TrackITS>* ITStracks = nullptr;
  std::vector<int>* ITSTrackClusIdx = nullptr;

  // Clusters
  std::vector<CompClusterExt>* ITSclus = nullptr;
  std::vector<unsigned char>* ITSpatt = nullptr;

  // Labels
  std::vector<o2::MCCompLabel>* labITSvec = nullptr;
  std::vector<o2::MCCompLabel>* labITSTPCvec = nullptr;
  std::vector<o2::MCCompLabel>* labITSTPCTOFvec = nullptr;
  std::vector<o2::MCCompLabel>* labTPCTOFvec = nullptr;

  // Setting branches
  treeSecondaries->SetBranchAddress("V0s", &v0vec);
  treeMCTracks->SetBranchAddress("MCTrack", &MCtracks);
  treeITSTPC->SetBranchAddress("MatchMCTruth", &labITSTPCvec);
  treeTPCTOF->SetBranchAddress("MatchTOFMCTruth", &labTPCTOFvec);
  treeITSTPCTOF->SetBranchAddress("MatchTOFMCTruth", &labITSTPCTOFvec);
  treeITS->SetBranchAddress("ITSTrackMCTruth", &labITSvec);
  treeITS->SetBranchAddress("ITSTrack", &ITStracks);
  treeITS->SetBranchAddress("ITSTrackClusIdx", &ITSTrackClusIdx);

  treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
  treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);

  // define detector map
  std::map<std::string, std::vector<o2::MCCompLabel>*> map{{"ITS", labITSvec}, {"ITS-TPC", labITSTPCvec}, {"TPC-TOF", labTPCTOFvec}, {"ITS-TPC-TOF", labITSTPCTOFvec}};

  // load geometry
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  // fill MC matrix
  std::vector<std::vector<o2::MCTrack>> mcTracksMatrix;
  auto nev = treeMCTracks->GetEntriesFast();
  mcTracksMatrix.resize(nev);
  for (int n = 0; n < nev; n++) { // loop over MC events
    treeMCTracks->GetEvent(n);

    mcTracksMatrix[n].resize(MCtracks->size());
    for (unsigned int mcI{0}; mcI < MCtracks->size(); ++mcI) {
      mcTracksMatrix[n][mcI] = MCtracks->at(mcI);
    }
  }

  treeSecondaries->GetEntry();
  treeITS->GetEntry();
  treeITSclus->GetEntry();
  treeITSTPC->GetEntry();
  treeTPCTOF->GetEntry();
  treeITSTPCTOF->GetEntry();

  std::vector<std::array<int, 2>> V0sMCref = matchV0stoMC(mcTracksMatrix, map, v0vec);

  // convert Comp Clusters into 3D point
  std::vector<ITSCluster> mITSClustersArray;
  mITSClustersArray.reserve((*ITSclus).size());
  gsl::span<const unsigned char> spanPatt{*ITSpatt};
  auto pattIt = spanPatt.begin();
  o2::its::ioutils::convertCompactClusters(*ITSclus, pattIt, mITSClustersArray, mdict);

  // preparing DCA Fitter
  o2::vertexing::DCAFitterN<2> mFitterV0;
  mFitterV0.setBz(-5);

  auto sig_counter = 0.;
  auto bkg_counter = 0.;

  // Starting matching  loop
  for (int frame = 0; frame < treeITS->GetEntriesFast(); frame++) {
    if (!treeITS->GetEvent(frame) || !treeITS->GetEvent(frame))
      continue;
    if (!treeITS->GetEvent(frame)) {
      continue;
    }

    for (unsigned int iTrack{0}; iTrack < labITSvec->size(); ++iTrack) {
      auto lab = labITSvec->at(iTrack);
      auto ITStrackMCref = matchITStracktoMC(mcTracksMatrix, lab);
      auto ITStrack = ITStracks->at(iTrack);

      auto ITSclusters = getTrackClusters(ITStrack, mITSClustersArray, ITSTrackClusIdx);

      for (unsigned int iV0vec = 0; iV0vec < v0vec->size(); iV0vec++) {
        auto& v0MCref = V0sMCref[iV0vec];
        auto& v0 = (*v0vec)[iV0vec];

        if (ITStrackMCref == v0MCref && ITStrackMCref[0] != -1) {

          auto& mcTrack = mcTracksMatrix[v0MCref[0]][v0MCref[1]];
          sig_counter++;
          auto hyperTrack = hyperTracker(ITStrack, v0, ITSclusters, gman, mFitterV0);
          hyperTrack.setBz(-5.);
          // hyperTrack.setNclusMatching(ITSclusters.size());
          hyperTrack.setMaxChi2(40);
          auto chi2 = hyperTrack.getMatchingChi2();
          std::cout << "V0 orig Pt: " << v0.getPt() << ", V0 recr Pt: " << hyperTrack.getV0().getPt() << std::endl;
          hPtResBef->Fill(mcTrack.GetPt(), (mcTrack.GetPt() - hyperTrack.getV0().getPt()) / mcTrack.GetPt());
          std::cout << "ITS track Pt: " << ITStrack.getPt() << std::endl;
          std::cout << "Starting hyperTracking algorithm..." << std::endl;
          auto isAcc = hyperTrack.process();

          std::cout << "After processing hyperTracking algorithm..." << std::endl;
          hPtResAft->Fill(mcTrack.GetPt(), (mcTrack.GetPt() - hyperTrack.getV0().getPt()) / mcTrack.GetPt());

          std::cout << "Is accepted? : " << isAcc << std::endl;

          for (auto i{0}; i < 7; i++) {
            if (ITStrack.isFakeOnLayer(i) && ITStrack.hasHitOnLayer(i))
              std::cout << "Fake clusters on layer: " << i << std::endl;
          }

          std::cout << "------------------------" << std::endl;
          hChi2Sgn->Fill(chi2);
          if (isAcc)
            hSigBkg->Fill(0.5);
        }
        if (ITStrackMCref != v0MCref) {
          if (bkg_counter > 20000)
            continue;
          bkg_counter++;

          auto hyperTrack = hyperTracker(ITStrack, v0, ITSclusters, gman, mFitterV0);
          hyperTrack.setBz(-5.);
          // hyperTrack.setNclusMatching(ITSclusters.size());
          hyperTrack.setMaxChi2(40);

          auto chi2 = hyperTrack.getMatchingChi2();
          if (chi2 > 90 || chi2 == -1)
            chi2 = 90;
          auto isAcc = hyperTrack.process();
          hChi2Bkg->Fill(chi2);
          if (isAcc)
            hSigBkg->Fill(1.5);
        }
      }
    }
  }
  auto outFile = TFile("v0sITSmatch.root", "recreate");
  // efficiency histo
  hSigBkg->SetBinContent(1, hSigBkg->GetBinContent(1) / sig_counter);
  hSigBkg->SetBinContent(2, hSigBkg->GetBinContent(2) / bkg_counter);
  hSigBkg->GetXaxis()->SetBinLabel(1, "Signal");
  hSigBkg->GetXaxis()->SetBinLabel(2, "Background");
  hSigBkg->Write();
  hPtResAft->Write();
  hPtResBef->Write();

  // chi2 histos
  auto* c = new TCanvas("c1", "chi2", 1000, 400);
  hChi2Bkg->SetStats(0);
  hChi2Sgn->SetLineColor(kRed);
  hChi2Sgn->SetLineWidth(2);
  hChi2Bkg->SetLineColor(kBlue);
  hChi2Bkg->SetLineWidth(2);
  c->cd();
  hChi2Bkg->DrawNormalized();
  hChi2Sgn->DrawNormalized("same");
  auto legend = new TLegend(0.55, 0.2, 0.85, 0.4);
  legend->SetMargin(0.10);
  legend->SetTextSize(0.03);

  legend->AddEntry(hChi2Sgn, "V0-ITStrack #chi^{2} for signal");
  legend->AddEntry(hChi2Bkg, "V0-ITStrack #chi^{2} for background");
  legend->Draw();
  c->Write();
  hChi2Sgn->Write();
  hChi2Bkg->Write();
  outFile.Close();
}

std::vector<std::array<int, 2>> matchV0stoMC(const std::vector<std::vector<o2::MCTrack>>& mcTracksMatrix, std::map<std::string, std::vector<o2::MCCompLabel>*>& map, std::vector<V0>* v0vec)
{
  std::vector<std::array<int, 2>> outArray;
  outArray.resize(v0vec->size());
  int count_V0 = 0;
  for (unsigned int iV0vec = 0; iV0vec < v0vec->size(); iV0vec++) {
    std::vector<int> motherIDvec;
    std::vector<int> daughterIDvec;
    std::vector<int> evIDvec;

    outArray[iV0vec] = {-1, -1};
    auto& v0 = (*v0vec)[iV0vec];

    for (unsigned int iV0 = 0; iV0 < 2; iV0++) {
      if (map[v0.getProngID(iV0).getSourceName()]) {
        auto labTrackType = map[v0.getProngID(iV0).getSourceName()];
        auto lab = labTrackType->at(v0.getProngID(iV0).getIndex());

        int trackID, evID, srcID;
        bool fake;
        lab.get(trackID, evID, srcID, fake);
        if (!lab.isNoise() && lab.isValid() && lab.isCorrect() && srcID) {
          auto motherID = mcTracksMatrix[evID][trackID].getMotherTrackId();
          motherIDvec.push_back(motherID);
          daughterIDvec.push_back(trackID);
          evIDvec.push_back(evID);
        }
      }
    }

    if (motherIDvec.size() < 2)
      continue;
    if (motherIDvec[0] != motherIDvec[1] || evIDvec[0] != evIDvec[1])
      continue;

    if (motherIDvec[0] <= 0 || motherIDvec[0] > 10000)
      continue;

    int pdg0 = mcTracksMatrix[evIDvec[0]][daughterIDvec[0]].GetPdgCode();
    int pdg1 = mcTracksMatrix[evIDvec[0]][daughterIDvec[1]].GetPdgCode();

    if (pdg0 != firstDaughterPDG && pdg0 != secondDaughterPDG)
      continue;
    if (pdg1 != firstDaughterPDG && pdg1 != secondDaughterPDG)
      continue;

    // std::cout << "Mother PDG: " << mcTracksMatrix[evIDvec[0]][motherIDvec[0]].GetPt() << std::endl;
    outArray[iV0vec] = {evIDvec[0], motherIDvec[0]};
    count_V0++;
  }
  std::cout << "Number of V0s: " << count_V0 << std::endl;
  return outArray;
}

std::array<int, 2> matchITStracktoMC(const std::vector<std::vector<o2::MCTrack>>& mcTracksMatrix, o2::MCCompLabel ITSlabel)

{
  std::array<int, 2> outArray = {-1, -1};
  int trackID, evID, srcID;
  bool fake;
  ITSlabel.get(trackID, evID, srcID, fake);
  if (!ITSlabel.isNoise() && ITSlabel.isValid() && srcID && mcTracksMatrix[evID][trackID].GetPdgCode() == motherPDG) {
    outArray = {evID, trackID};
    // std::cout << "ITS EvID, track Id : " << evID << "   " << trackID << std::endl;
    // std::cout << "ITS Mother Pt: " << mcTracksMatrix[evID][trackID].GetPt() << std::endl;
  }

  return outArray;
}

std::vector<ITSCluster> getTrackClusters(const o2::its::TrackITS& ITStrack, const std::vector<ITSCluster>& ITSClustersArray, std::vector<int>* ITSTrackClusIdx)
{

  std::vector<ITSCluster> outVec;
  auto firstClus = ITStrack.getFirstClusterEntry();
  auto ncl = ITStrack.getNumberOfClusters();
  for (int icl = 0; icl < ncl; icl++) {
    outVec.push_back(ITSClustersArray[(*ITSTrackClusIdx)[firstClus + icl]]);
  }
  return outVec;
}