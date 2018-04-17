#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <string>
#include <array>
#include <vector>
#include <cmath>
#include <sstream>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TNtuple.h>
#include <TGeoGlobalMagField.h>

#include <FairLogger.h>
#include "FairRunAna.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairGeoParSet.h"
#include "FairMCEventHeader.h"

#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
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

using o2::ITS::CA::Cluster;
using o2::ITS::CA::Line;
using o2::ITS::CA::MathUtils::calculatePhiCoordinate;
using o2::ITS::CA::MathUtils::calculateRCoordinate;
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

void run_vert_ca_its(const int inspEvt = -1, bool useMC = false,
                     std::tuple<float, float, float, float, int> initParams = { 0.005, 0.002, 0.04, 0.8, 5 },
                     std::string path = "./", std::string inputClustersITS = "o2clus_its.root",
                     std::string inputGeom = "O2geometry.root", std::string inputGRP = "o2sim_grp.root",
                     std::string paramfilename = "o2sim_par.root", std::string simfilename = "o2sim.root",
                     std::string outfile = "vertexer_data.root")
{

  o2::ITS::CA::Event event;
  if (path.back() != '/')
    path += '/';
  //-------- init geometry and field --------//

  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  if (!grp) {
    LOG(FATAL) << "Cannot run w/o GRP object" << FairLogger::endl;
  }
  o2::Base::GeometryManager::loadGeometry(path + inputGeom, "FAIRGeom");
  o2::Base::Propagator::initFieldFromGRP(grp);
  auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!field) {
    LOG(FATAL) << "Failed to load ma" << FairLogger::endl;
  }
  double origD[3] = { 0., 0., 0. };
  // tracker.setBz(field->getBz(origD));
  bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  if (!isITS) {
    LOG(WARNING) << "ITS is not in the readoute" << FairLogger::endl;
    return;
  }
  bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(INFO) << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode" << FairLogger::endl;

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
  // Output tree
  TTree outTree("o2sim", "Vertexer Vertices");
  std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>>* verticesITS =
    new std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>>;
  outTree.Branch("ITSVertices", &verticesITS);

  // TNtuple* verTupleResiduals =
  // new TNtuple("residuals", "residuals", "evtid:id:residualX:residualY:residualZ:contribs:avg_dist");
  // TNtuple* verTupleResidualsmc =
  // new TNtuple("residuals_mc", "residuals_mc", "evtid:id:residualX:residualY:residualZ:contribs:avg_dist");
  // TNtuple* evtDumpFromVtxer = new TNtuple("evtdump", "evtdump", "evt_id:nClusters:effRecTrks:effMCTrks");
  // TNtuple* evtDumpFromVtxermc = new TNtuple("evtdump_mc", "evtdump_mc", "evt_id:nClusters");

  std::uint32_t roFrame = 0;
  const int stopAt = (inspEvt == -1) ? itsClusters.GetEntries() : inspEvt + 1;
  for (int iEvent = (inspEvt == -1) ? 0 : inspEvt; iEvent < stopAt; ++iEvent) {
    int idx{ 0 };
    itsClusters.GetEntry(iEvent);
    mcHeaderTree.GetEntry(iEvent);

    if (isContITS) {
      int nclLeft = clusters->size();
      while (nclLeft > 0) {
        int nclUsed = o2::ITS::CA::IOUtils::loadROFrameData(roFrame, event, clusters, labels);
        if (nclUsed) {
          cout << "Event " << iEvent << " ROFrame " << roFrame << std::endl;
          o2::ITS::CA::Vertexer vertexer(event);
          vertexer.setROFrame(roFrame);
          vertexer.initialise(initParams);
          vertexer.findTracklets(useMC);
          std::cout << "\ttracklets found: " << vertexer.getTracklets().size() << std::endl;
          vertexer.findVertices();
          // auto vertices = vertexer.getLegacyVertices();
          verticesITS->swap(vertexer.getVertices());
          std::cout << "\tvertices found: " << verticesITS->size() << std::endl;
          // bevtDumpFromVtxer->Fill(static_cast<float>(iEvent), static_cast<float>(vertexer.mClusters[0].size()));
          // for (auto& vertex : vertices) {
          //  float tmpdata[5] = { static_cast<float>(iEvent), static_cast<float>(idx), vertex[0], vertex[1], vertex[2]
          //  };
          //  verTupleResiduals->Fill(tmpdata);
          //}
          nclLeft -= nclUsed;
        }
        roFrame++;
        outTree.Fill();
      }
    } else { // triggered mode
      cout << "Event " << iEvent << std::endl;
      o2::ITS::CA::IOUtils::loadEventData(event, clusters, labels);
      o2::ITS::CA::Vertexer vertexer(event);
      vertexer.setROFrame(roFrame);
      vertexer.initialise(initParams);
      vertexer.findTracklets(useMC);
      std::cout << "\ttracklets found: " << vertexer.getTracklets().size() << std::endl;
      vertexer.findVertices();
      // auto vertices = vertexer.getLegacyVertices();
      verticesITS->swap(vertexer.getVertices());
      std::cout << "\tvertices found: " << verticesITS->size() << std::endl;
      // evtDumpFromVtxer->Fill(static_cast<float>(iEvent), static_cast<float>(vertexer.mClusters[0].size()));
      // for (auto& vertex : vertices) {
      //   float tmpdata[5] = { static_cast<float>(iEvent), static_cast<float>(idx), vertex[0], vertex[1], vertex[2] };
      //   verTupleResiduals->Fill(tmpdata);
      // }
      outTree.Fill();
    }
  } // loop on events
  outTree.Write();
  // verTupleResiduals->Write();
  // evtDumpFromVtxer->Write();
  outputfile->Close();
}
#endif
*/
