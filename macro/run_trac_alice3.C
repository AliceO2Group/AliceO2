#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <string>

#include <TFile.h>
#include <TChain.h>
#include <TBranch.h>
#include <TRandom3.h>
#include <TGeoGlobalMagField.h>
#include <vector>


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


#endif

using o2::itsmft::Hit;
using std::string;
using o2::its::MemoryParameters;
using o2::its::TrackingParameters;

constexpr bool kUseSmearing{false};

float getDetLengthFromEta(const float eta, const float radius)
{
  return 10. * (10. + radius * std::cos(2 * std::atan(std::exp(-eta))));
}

void run_vertexer(const string hitsFileName = "o2sim_HitsIT4.root")
{
  o2::its::Vertexer vertexer(new o2::its::VertexerTraits());
  TChain itsHits("o2sim");

  itsHits.AddFile(hitsFileName.data());

  o2::its::Tracker tracker(new o2::its::TrackerTraitsCPU());
  tracker.setBz(5.f);


  std::uint32_t roFrame;
  std::vector<Hit>* hits = nullptr;
  itsHits.SetBranchAddress("IT4Hit", &hits);

  std::vector<TrackingParameters> trackParams(1);
  trackParams[0].NLayers = 10;
  trackParams[0].MinTrackLength = 10;
  std::cout << "Trackparams: " << trackParams[0].CellsPerRoad() << std::endl;

  std::vector<float> LayerRadii = {1.8f,2.4f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f};
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
  trackParams[0].TrackletMaxDeltaZ = TrackletMaxDeltaZ;
  trackParams[0].CellMaxDCA = CellMaxDCA;
  trackParams[0].CellMaxDeltaZ = CellMaxDeltaZ;
  trackParams[0].NeighbourMaxDeltaCurvature = NeighbourMaxDeltaCurvature;
  trackParams[0].NeighbourMaxDeltaN = NeighbourMaxDeltaN;

  std::vector<MemoryParameters> memParams(1);
  std::vector<float> CellsMemoryCoefficients = {2.3208e-08f *20, 2.104e-08f*20, 1.6432e-08f*20, 1.2412e-08f*20, 1.3543e-08f*20, 1.5e-08f*20, 1.6e-08f*20, 1.7e-08f*20};
  std::vector<float> TrackletsMemoryCoefficients = {0.0016353f * 10, 0.0013627f * 10, 0.000984f * 10, 0.00078135f * 10, 0.00057934f * 10, 0.00052217f * 10, 0.00052217f * 10, 0.00052217f * 10, 0.00052217f * 10};
  memParams[0].CellsMemoryCoefficients = CellsMemoryCoefficients;
  memParams[0].TrackletsMemoryCoefficients = TrackletsMemoryCoefficients;

  tracker.setParameters(memParams, trackParams);

  for (int iEvent{0}; iEvent < itsHits.GetEntriesFast(); ++iEvent) {
    std::cout << "***** Event " << iEvent << " *****" << std::endl;
    itsHits.GetEntry(iEvent);
    o2::its::ROframe event{iEvent, 10};

    int id{0};
    for (auto& hit : *hits) {
      const int layer{hit.GetDetectorID()};
      float xyz[3]{hit.GetX(), hit.GetY(), hit.GetZ()};
      float r{std::hypot(xyz[0],xyz[1])};
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

    }
    roFrame = iEvent;

    vertexer.clustersToVertices(event);
    tracker.clustersToTracks(event);

  }

}
