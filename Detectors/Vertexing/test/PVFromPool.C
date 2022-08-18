#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "DetectorsVertexing/PVertexer.h"
#include "DetectorsVertexing/PVertexerHelpers.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include <TGeoGlobalMagField.h>
#include <TGeoManager.h>
#include <TStopwatch.h>
#include "Field/MagneticField.h"
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "DetectorsBase/Propagator.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "CCDB/BasicCCDBManager.h"
#include <string>

using namespace o2::vertexing;

// macro to run PVertex finder for the particular TF from the TrackVF pool dumped via PVertexer::dumpPool() method
// Must be run only in compiled mode

void PVFromPool(int run,                       // run number
                const char* poolName,          // filename of the track pool dump
                const std::string& vtopts = "" // additional options for ConfigurableParam objects
)
{
  TFile pf(poolName);
  const auto* pvecPtr = (std::vector<TrackVF>*)pf.GetObjectUnchecked("pool");
  std::vector<PVertex> vertices;
  std::vector<o2::dataformats::VtxTrackIndex> vertexTrackIDs;
  std::vector<V2TRef> v2tRefs;

  auto& cm = o2::ccdb::BasicCCDBManager::instance();
  auto rlim = cm.getRunDuration(run);
  long ts = rlim.first + (rlim.second - rlim.first) / 2;
  cm.getSpecific<TGeoManager>("GLO/Config/GeometryAligned", ts);

  const auto* grpLHCIF = cm.getSpecific<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF", ts);
  const auto* grpBField = cm.getSpecific<o2::parameters::GRPMagField>("GLO/Config/GRPMagField", ts);
  o2::base::Propagator::initFieldFromGRP(grpBField);

  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  alpParams.printKeyValues();

  cm.getSpecific<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>>("ITS/Config/AlpideParam", ts);

  alpParams.printKeyValues();

  float ITSROFrameLengthMUS = alpParams.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3; // ITS ROFrame duration in \mus

  o2::conf::ConfigurableParam::updateFromString(vtopts);

  PVertexerParams::Instance().printKeyValues();

  o2::vertexing::PVertexer pvfinder;
  pvfinder.setBunchFilling(grpLHCIF->getBunchFilling());
  pvfinder.setITSROFrameLength(ITSROFrameLengthMUS);
  pvfinder.init();
  TStopwatch timer;
  pvfinder.processFromExternalPool(*pvecPtr, vertices, vertexTrackIDs, v2tRefs);
  pvfinder.end();
  timer.Stop();

  LOGP(info, "Found {} PVs, Time CPU/Real:{:.3f}/{:.3f} (DBScan: {:.4f}, Finder:{:.4f}, Rej.Debris:{:.4f}, Reattach:{:.4f}) | {} trials for {} TZ-clusters, max.trials: {}, Slowest TZ-cluster: {} ms of mult {}",
       vertices.size(), timer.CpuTime(), timer.RealTime(),
       pvfinder.getTimeDBScan().CpuTime(), pvfinder.getTimeVertexing().CpuTime(), pvfinder.getTimeDebris().CpuTime(), pvfinder.getTimeReAttach().CpuTime(),
       pvfinder.getTotTrials(), pvfinder.getNTZClusters(), pvfinder.getMaxTrialsPerCluster(),
       pvfinder.getLongestClusterTimeMS(), pvfinder.getLongestClusterMult());
}

#endif
