#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <string>

#include <TFile.h>
#include <TChain.h>
#include <TH2D.h>
#include <TBranch.h>
#include <TRandom3.h>
#include <TGeoGlobalMagField.h>
#include <vector>
#include <Math/Point3Dfwd.h>
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraitsCPU.h"
#include "ITStracking/Vertexer.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SimulationDataFormat/MCTrack.h"
#include "MathUtils/Cartesian.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/Vertex.h"
#endif

using o2::its::MemoryParameters;
using o2::its::TrackingParameters;
using o2::itsmft::Hit;
using std::string;

constexpr bool kUseSmearing{false};

struct particle {
  int pdg = 0;
  int nLayers = 0;
  float pt;
  float eta;
  float phi;
  float recoPt;
  float recoEta;
  float energyFirst;
  float energyLast;
  int isReco = 0; // 0 = no, 1 = good, 2 = fake
};

float getDetLengthFromEta(const float eta, const float radius)
{
  return 10. * (10. + radius * std::cos(2 * std::atan(std::exp(-eta))));
}

void run_trac_alice3(const string hitsFileName = "o2sim_HitsTRK.root")
{

  TChain mcTree("o2sim");
  mcTree.AddFile("o2sim_Kine.root");
  mcTree.SetBranchStatus("*", 0); //disable all branches
  mcTree.SetBranchStatus("MCTrack*", 1);

  std::vector<o2::MCTrack>* mcArr = nullptr;
  mcTree.SetBranchAddress("MCTrack", &mcArr);

  o2::its::Vertexer vertexer(new o2::its::VertexerTraits());
  TChain itsHits("o2sim");

  itsHits.AddFile(hitsFileName.data());

  o2::its::Tracker tracker(new o2::its::TrackerTraitsCPU());
  tracker.setBz(5.f);
  tracker.setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrTGeo);

  std::uint32_t roFrame;
  std::vector<Hit>* hits = nullptr;
  itsHits.SetBranchAddress("TRKHit", &hits);

  std::vector<TrackingParameters> trackParams(4);
  trackParams[0].NLayers = 10;
  trackParams[0].MinTrackLength = 10;

  std::vector<float> LayerRadii = {1.8f, 2.8f, 3.8f, 8.0f, 20.0f, 25.0f, 40.0f, 55.f, 80.0f, 100.f};
  std::vector<float> LayerZ(10);
  for (int i{0}; i < 10; ++i)
    LayerZ[i] = getDetLengthFromEta(1.44, LayerRadii[i]) + 1.;
  std::vector<float> TrackletMaxDeltaZ = {0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f, 0.5f, 0.5f, 0.5f};
  std::vector<float> CellMaxDCA = {0.05f, 0.04f, 0.05f, 0.2f, 0.4f, 0.5f, 0.5f, 0.5f};
  std::vector<float> CellMaxDeltaZ = {0.2f, 0.4f, 0.5f, 0.6f, 3.0f, 3.0f, 3.0f, 3.0f};
  std::vector<float> NeighbourMaxDeltaCurvature = {0.008f, 0.0025f, 0.003f, 0.0035f, 0.004f, 0.004f, 0.005f};
  std::vector<float> NeighbourMaxDeltaN = {0.002f, 0.0090f, 0.002f, 0.005f, 0.005f, 0.005f, 0.005f};

  trackParams[0].LayerRadii = LayerRadii;
  trackParams[0].LayerZ = LayerZ;
  trackParams[0].TrackletMaxDeltaPhi = 0.3;
  trackParams[0].CellMaxDeltaPhi = 0.15;
  trackParams[0].CellMaxDeltaTanLambda = 0.03;
  trackParams[0].TrackletMaxDeltaZ = TrackletMaxDeltaZ;
  trackParams[0].CellMaxDCA = CellMaxDCA;
  trackParams[0].CellMaxDeltaZ = CellMaxDeltaZ;
  trackParams[0].NeighbourMaxDeltaCurvature = NeighbourMaxDeltaCurvature;
  trackParams[0].NeighbourMaxDeltaN = NeighbourMaxDeltaN;

  std::vector<MemoryParameters> memParams(4);
  std::vector<float> CellsMemoryCoefficients = {2.3208e-08f * 20, 2.104e-08f * 20, 1.6432e-08f * 20, 1.2412e-08f * 20, 1.3543e-08f * 20, 1.5e-08f * 20, 1.6e-08f * 20, 1.7e-08f * 20};
  std::vector<float> TrackletsMemoryCoefficients = {0.0016353f * 1000, 0.0013627f * 1000, 0.000984f * 1000, 0.00078135f * 1000, 0.00057934f * 1000, 0.00052217f * 1000, 0.00052217f * 1000, 0.00052217f * 1000, 0.00052217f * 1000};
  memParams[0].CellsMemoryCoefficients = CellsMemoryCoefficients;
  memParams[0].TrackletsMemoryCoefficients = TrackletsMemoryCoefficients;
  memParams[0].MemoryOffset = 8000;

  for (int i = 1; i < 4; ++i) {
    memParams[i] = memParams[i - 1];
    trackParams[i] = trackParams[i - 1];
    // trackParams[i].MinTrackLength -= 2;
    trackParams[i].TrackletMaxDeltaPhi = trackParams[i].TrackletMaxDeltaPhi * 3 > TMath::Pi() ? TMath::Pi() : trackParams[i].TrackletMaxDeltaPhi * 3;
    trackParams[i].CellMaxDeltaPhi = trackParams[i].CellMaxDeltaPhi * 3 > TMath::Pi() ? TMath::Pi() : trackParams[i].CellMaxDeltaPhi * 3;
    trackParams[i].CellMaxDeltaTanLambda *= 3;
    for (auto& val : trackParams[i].TrackletMaxDeltaZ)
      val *= 3;
    for (auto& val : trackParams[i].CellMaxDCA)
      val *= 3;
    for (auto& val : trackParams[i].CellMaxDeltaZ)
      val *= 3;
    for (auto& val : trackParams[i].NeighbourMaxDeltaCurvature)
      val *= 3;
    for (auto& val : trackParams[i].NeighbourMaxDeltaN)
      val *= 3;
  }

  tracker.setParameters(memParams, trackParams);

  constexpr int nBins = 100;
  constexpr float minPt = 0.01;
  constexpr float maxPt = 10;
  double newBins[nBins + 1];
  newBins[0] = minPt;
  double factor = pow(maxPt / minPt, 1. / nBins);
  for (int i = 1; i <= nBins; i++) {
    newBins[i] = factor * newBins[i - 1];
  }

  TH1D genH("gen", ";#it{p}_{T} (GeV/#it{c});", nBins, newBins);
  TH1D recH("rec", "Efficiency;#it{p}_{T} (GeV/#it{c});", nBins, newBins);
  TH1D fakH("fak", "Fake rate;#it{p}_{T} (GeV/#it{c});", nBins, newBins);
  TH2D deltaPt("deltapt", ";#it{p}_{T} (GeV/#it{c});#Delta#it{p}_{T} (GeV/#it{c});", nBins, newBins, 100, -0.1, 0.1);
  TH2D dcaxy("dcaxy", ";#it{p}_{T} (GeV/#it{c});DCA_{xy} (#mum);", nBins, newBins, 200, -200, 200);
  TH2D dcaz("dcaz", ";#it{p}_{T} (GeV/#it{c});DCA_{z} (#mum);", nBins, newBins, 200, -200, 200);
  TH2D deltaE("deltaE", ";#it{p}_{T} (GeV/#it{c});#DeltaE (MeV);", nBins, newBins, 200, 0, 100);

  ROOT::Math::XYZPointF pos{0.f, 0.f, 0.f};
  const std::array<float, 6> cov{1.e-4, 0., 0., 1.e-4, 0., 1.e-4};
  o2::dataformats::VertexBase vtx(pos, cov);
  o2::dataformats::DCA dca;

  for (int iEvent{0}; iEvent < itsHits.GetEntriesFast(); ++iEvent) {
    std::cout << "*************** Event " << iEvent << " ***************" << std::endl;
    itsHits.GetEntry(iEvent);
    mcTree.GetEvent(iEvent);
    o2::its::ROframe event{iEvent, 10};

    int id{0};
    std::map<int, particle> mapPDG;
    for (auto& hit : *hits) {
      const int layer{hit.GetDetectorID()};
      float xyz[3]{hit.GetX(), hit.GetY(), hit.GetZ()};
      float r{std::hypot(xyz[0], xyz[1])};
      float phi{std::atan2(-xyz[1], -xyz[0]) + o2::its::constants::math::Pi};

      if (kUseSmearing) {
        phi = gRandom->Gaus(phi, std::asin(0.0005f / r));
        xyz[0] = r * std::cos(phi);
        xyz[1] = r * std::sin(phi);
        xyz[2] = gRandom->Gaus(xyz[2], 0.0005f);
      }

      event.addTrackingFrameInfoToLayer(layer, xyz[0], xyz[1], xyz[2], r, phi, std::array<float, 2>{0.f, xyz[2]},
                                        std::array<float, 3>{0.0005f * 0.0005f, 0.f, 0.0005f * 0.0005f});
      event.addClusterToLayer(layer, xyz[0], xyz[1], xyz[2], event.getClustersOnLayer(layer).size());
      event.addClusterLabelToLayer(layer, o2::MCCompLabel(hit.GetTrackID(), iEvent, iEvent, false));
      event.addClusterExternalIndexToLayer(layer, id++);
      if (mapPDG.find(hit.GetTrackID()) == mapPDG.end()) {
        mapPDG[hit.GetTrackID()] = particle();
        mapPDG[hit.GetTrackID()].nLayers |= 1 << layer;
        mapPDG[hit.GetTrackID()].pt = std::hypot(hit.GetPx(), hit.GetPy());
        if (hit.GetTrackID() < mcArr->size()) {
          auto part = mcArr->at(hit.GetTrackID());
          mapPDG[hit.GetTrackID()].energyFirst = part.GetEnergy();
          mapPDG[hit.GetTrackID()].pdg = part.GetPdgCode();
          mapPDG[hit.GetTrackID()].pt = part.GetPt();
          mapPDG[hit.GetTrackID()].eta = part.GetEta();
          mapPDG[hit.GetTrackID()].phi = part.GetPhi();
        }
      } else {
        mapPDG[hit.GetTrackID()].nLayers |= 1 << layer;
        mapPDG[hit.GetTrackID()].energyLast = hit.GetE();
      }
    }
    roFrame = iEvent;

    vertexer.clustersToVertices(event);
    int nPart10layers{0};
    for (auto& part : mapPDG) {
      if (part.second.nLayers == 0x3FF) {
        nPart10layers++;
        genH.Fill(part.second.pt);
        deltaE.Fill(part.second.pt, 1000 * (part.second.energyFirst - part.second.energyLast));
      }
    }
    tracker.clustersToTracks(event);
    auto& tracks = tracker.getTracks();
    auto& tracksLabels = tracker.getTrackLabels();
    std::cout << "**** " << nPart10layers << " particles with 10 layers " << tracks.size() << " tracks" << std::endl;
    int good{0};
    for (unsigned int i{0}; i < tracks.size(); ++i) {
      auto& lab = tracksLabels[i];
      auto& track = tracks[i];
      int trackID = std::abs(lab.getTrackID());
      if (mapPDG.find(trackID) != mapPDG.end()) {
        if (!mapPDG[trackID].pdg)
          std::cout << "strange" << std::endl;
        mapPDG[trackID].isReco = lab.isFake() ? 2 : 1;
        mapPDG[trackID].recoPt = track.getPt();
        mapPDG[trackID].recoEta = track.getEta();
        (lab.isFake() ? fakH : recH).Fill(mapPDG[trackID].pt);
        deltaPt.Fill(mapPDG[trackID].pt, (mapPDG[trackID].pt - mapPDG[trackID].recoPt) / mapPDG[trackID].pt);
        if (!track.propagateToDCA(vtx, tracker.getBz(), &dca)) {
          std::cout << "Propagation failed." << std::endl;
        } else {
          dcaxy.Fill(mapPDG[trackID].pt, dca.getY() * 1.e4);
          dcaz.Fill(mapPDG[trackID].pt, dca.getZ() * 1.e4);
        }
      }
    }
  }

  TH1D dcaxy_res(recH);
  dcaxy_res.Reset();
  dcaxy_res.SetNameTitle("dcaxy_res", ";#it{p}_{T} (GeV/#it{c});#sigma(DCA_{xy}) (#mum);");
  TH1D dcaz_res(recH);
  dcaz_res.Reset();
  dcaz_res.SetNameTitle("dcaz_res", ";#it{p}_{T} (GeV/#it{c});#sigma(DCA_{z}) (#mum);");
  for (int i = 1; i <= nBins; ++i) {
    auto proj = dcaxy.ProjectionY(Form("xy%i", i), i, i);
    dcaxy_res.SetBinContent(i, proj->GetStdDev());
    dcaxy_res.SetBinError(i, proj->GetStdDevError());
    proj = dcaz.ProjectionY(Form("z%i", i), i, i);
    dcaz_res.SetBinContent(i, proj->GetStdDev());
    dcaz_res.SetBinError(i, proj->GetStdDevError());
  }

  TFile output("output.root", "recreate");
  recH.Divide(&genH);
  fakH.Divide(&genH);
  recH.Write();
  fakH.SetLineColor(kRed);
  fakH.Write();
  genH.Write();
  deltaPt.Write();
  dcaxy.Write();
  dcaz.Write();
  dcaz_res.Write();
  dcaxy_res.Write();
  deltaE.Write();
}
