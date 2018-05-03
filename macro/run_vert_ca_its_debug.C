#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TNtuple.h>
#include <TH1I.h>
#include <TGeoGlobalMagField.h>
#include <TParticle.h>
#include <string>
#include <array>
#include <vector>
#include <cmath>
#include <sstream>
#include <FairLogger.h>
#include "FairRunAna.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairGeoParSet.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "Field/MagneticField.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/IOUtils.h"
#include "ITSReconstruction/CA/Vertexer.h"
#include "ITSReconstruction/CA/ClusterLines.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "FairMCEventHeader.h"
// #define DEBUG_BUILD
using o2::ITS::CA::Cluster;
using o2::ITS::CA::Line;
using o2::ITS::CA::MathUtils::calculatePhiCoordinate;
using o2::ITS::CA::MathUtils::calculateRCoordinate;
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

void run_vert_ca_its_debug(int startfrom = 0, int nevents = 1000, std::string path = "./",
                           std::string inputClustersITS = "o2clus_its.root",
                           std::string paramfilename = "o2sim_par.root", std::string simfilename = "o2sim.root",
                           std::string outfile = "vertexer_data.root")
{

  o2::ITS::CA::Event event;
  if (path.back() != '/')
    path += '/';

  // Setup Runtime DB
  TFile paramFile(paramfilename.data());
  paramFile.Get("FairGeoParSet");
  auto gman = o2::ITS::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms

  //>>>---------- attach input data --------------->>>
  TChain mcHeaderTree("o2sim");
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());
  mcHeaderTree.AddFile((path + simfilename).data());

  //<<<---------- attach input data ---------------<<<
  // Workaround to obtain orign monte carlo vertex
  TFile* mEventFile = TFile::Open("Kinematics_pbpb_100_novtx.root");

  if (!itsClusters.GetBranch("ITSCluster"))
    LOG(FATAL) << "Did not find ITS clusters branch ITSCluster in the input tree" << FairLogger::endl;

  std::vector<o2::ITSMFT::Cluster>* clusters = nullptr;
  itsClusters.SetBranchAddress("ITSCluster", &clusters);

  if (!itsClusters.GetBranch("ITSClusterMCTruth"))
    LOG(FATAL) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree" << FairLogger::endl;

  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);

  FairMCEventHeader* header = nullptr;
  if (!mcHeaderTree.GetBranch("MCEventHeader."))
    LOG(FATAL) << "Did not find MC event header in the input header file." << FairLogger::endl;

  mcHeaderTree.SetBranchAddress("MCEventHeader.", &header);
  TFile* outputfile = new TFile(outfile.data(), "recreate");

  TTree outTree("o2sim", "Vertexer Vertices");

  std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>>* verticesITS =
    new std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>>;
  std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>>* verticesITSMC =
    new std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>>;
  outTree.Branch("ITSVertices", &verticesITS);
  outTree.Branch("ITSVerticesMC", &verticesITSMC);
  TNtuple* verTupleResiduals =
    new TNtuple("residuals", "residuals", "evtid:id:residualX:residualY:residualZ:contribs:avg_dist");
  TNtuple* verTupleResidualsmc =
    new TNtuple("residuals_mc", "residuals_mc", "evtid:id:residualX:residualY:residualZ:contribs:avg_dist");
  TNtuple* evtDumpFromVtxer = new TNtuple("evtdump", "evtdump", "evt_id:nClusters:effRecTrks:effMCTrks");
  TNtuple* evtDumpFromVtxermc = new TNtuple("evtdump_mc", "evtdump_mc", "evt_id:nClusters");

  int startevent = static_cast<int>(std::max(0, startfrom));
  int endevent = std::min(static_cast<int>(itsClusters.GetEntries()), nevents + startevent);
  std::cout << "running on evt: [" << startevent << ", " << endevent << ")" << std::endl;
  TTree* tree = new TTree();
  for (int iEvent{ startevent }; iEvent < endevent; ++iEvent) {
    std::cout << "evt: " << iEvent << std::endl;
    int good{ 0 }, bad{ 0 }, duplicate{ 0 }, duplicatemc{ 0 }, idx{ 0 }, idx_mc{ 0 };
    std::stringstream treestringstr;
    treestringstr << "Event" << iEvent << "/TreeK";
    tree = (TTree*)mEventFile->Get(treestringstr.str().c_str());
    auto branch = tree->GetBranch("Particles");
    TParticle* primary = new TParticle();
    branch->SetAddress(&primary);
    branch->GetEntry(0); // first primary particle only, needed
    itsClusters.GetEntry(iEvent);
    mcHeaderTree.GetEntry(iEvent);
    o2::ITS::CA::IOUtils::loadEventData(event, clusters, labels);

#ifdef DEBUG_BUILD
    //<<<---------- MC tracklets reconstruction ---------------<<<
    std::cout << "\tFinding vertices on trackID-validated tracklets" << std::endl;
    o2::ITS::CA::Vertexer vertexer_montecarlo(event);
    vertexer_montecarlo.initialise(0.005, 0.002, 0.04, 0.8, 5);
    vertexer_montecarlo.findTracklets(true);
    auto tracklets_montecarlo = vertexer_montecarlo.getTracklets();
    std::cout << "\t\ttracklets found: " << tracklets_montecarlo.size() << std::endl;
    std::vector<int> used_mc_ids{};
    for (auto tracklet : tracklets_montecarlo) {
      bool duplicate_found = false;
      if (tracklet.originID == tracklet.destinID) {
        for (auto id : used_mc_ids) {
          if (id == tracklet.originID) {
            duplicate_found = true;
            ++duplicatemc;
          }
          if (duplicate_found)
            break;
        }
        if (duplicate_found)
          continue;
        used_mc_ids.push_back(tracklet.originID);
      }
    }
    std::cout << "\t\tduplicate trackID-validated tracklets found: " << duplicatemc << std::endl;
    vertexer_montecarlo.findVertices();
    verticesITSMC->swap(vertexer_montecarlo.getVertices());
    auto vertices_montecarlo = vertexer_montecarlo.getLegacyVertices();
    std::cout << "\t\tvertices found: " << vertices_montecarlo.size() << std::endl;

#endif

    //<<<---------- reconstruction ---------------<<<
    std::cout << "\n\tFinding vertices on REC tracklets" << std::endl;
    o2::ITS::CA::Vertexer vertexer(event);
    vertexer.initialise(0.005, 0.002, 0.04, 0.8, 5);
    vertexer.findTracklets();
    auto tracklets = vertexer.getTracklets();
    std::cout << "\t\ttracklets found: " << tracklets.size() << std::endl;

#ifdef DEBUG_BUILD
    std::vector<int> used_ids{};
    for (auto tracklet : tracklets) {
      bool duplicate_found = false;
      if (tracklet.originID == tracklet.destinID) {
        for (auto id : used_ids) {
          if (id == tracklet.originID) {
            duplicate_found = true;
            ++duplicate;
          }
          if (duplicate_found)
            break;
        }
        if (duplicate_found)
          continue;
        used_ids.push_back(tracklet.originID);
        ++good;
      } else {
        ++bad;
      }
    }
    std::cout << "\t\tduplicate reconstructed tracklets found: " << duplicate << std::endl;

    float effRecTrks = (float)good / (float)(tracklets.size());
    float effMCTrks{ (float)good / (float)(tracklets_montecarlo.size()) };

    std::cout << "\t\trecgood/trackletsfound: " << std::setprecision(4) << 100 * effRecTrks << "%" << std::endl;
    std::cout << "\t\trecgood/validatedfound: " << std::setprecision(4) << 100 * effMCTrks << "%" << std::endl;
#endif

    vertexer.findVertices();
    verticesITS->swap(vertexer.getVertices());
    auto vertices = vertexer.getLegacyVertices();
    std::cout << "\t\tvertices found: " << vertices.size() << std::endl;
    evtDumpFromVtxer->Fill(static_cast<float>(iEvent), static_cast<float>(vertexer.mClusters[0].size())
#ifdef DEBUG_BUILD
                                                         ,
                           effRecTrks, effMCTrks
#endif
                           );

#ifdef DEBUG_BUILD
    evtDumpFromVtxermc->Fill(static_cast<float>(iEvent), static_cast<float>(vertexer_montecarlo.mClusters[0].size()));
#endif

    for (auto& vertex : vertices) {
#ifdef DEBUG_BUILD
      float tmpdata[7] = { static_cast<float>(iEvent),
                           static_cast<float>(idx),
                           std::get<0>(vertex)[0] /* - static_cast<float>(primary->Vx()) */,
                           std::get<0>(vertex)[1] /* - static_cast<float>(primary->Vy()) */,
                           std::get<0>(vertex)[2] /* - static_cast<float>(primary->Vz()) */,
                           static_cast<float>(std::get<1>(vertex)),
                           static_cast<float>(std::get<2>(vertex)) };
#else
      float tmpdata[5] = { static_cast<float>(iEvent), static_cast<float>(idx), vertex[0], vertex[1], vertex[2] };
#endif
      verTupleResiduals->Fill(tmpdata);
      ++idx;
    }
#ifdef DEBUG_BUILD
    for (auto& vertex : vertices_montecarlo) {
      float tmpdata[7] = { static_cast<float>(iEvent),
                           static_cast<float>(idx),
                           std::get<0>(vertex)[0] /* - static_cast<float>(primary->Vx()) */,
                           std::get<0>(vertex)[1] /* - static_cast<float>(primary->Vy()) */,
                           std::get<0>(vertex)[2] /* - static_cast<float>(primary->Vz()) */,
                           static_cast<float>(std::get<1>(vertex)),
                           static_cast<float>(std::get<2>(vertex)) };

      verTupleResidualsmc->Fill(tmpdata);
      ++idx_mc;
    }
#endif
    outTree.Fill();
  } // Loop on events;

  verTupleResiduals->Write();
  evtDumpFromVtxer->Write();
#ifdef DEBUG_BUILD
  evtDumpFromVtxermc->Write();
  verTupleResidualsmc->Write();
#endif
  outTree.Write();
  outputfile->Close();
}
#endif
