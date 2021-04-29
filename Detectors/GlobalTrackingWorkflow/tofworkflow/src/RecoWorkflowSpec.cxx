// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFWorkflow/RecoWorkflowSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Framework/SerializationMethods.h"
#include "Headers/DataHeader.h"
#include "DataFormatsTOF/Cluster.h"
#include "GlobalTracking/MatchTOF.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include <gsl/span>
#include "TStopwatch.h"

// from FIT
#include "DataFormatsFT0/RecPoints.h"

#include <memory> // for make_shared, make_unique, unique_ptr
#include <vector>

using namespace o2::framework;

namespace o2
{
namespace tof
{

// use the tasking system of DPL
// just need to implement 2 special methods init + run (there is no need to inherit from anything)
class TOFDPLRecoWorkflowTask
{
  using evIdx = o2::dataformats::EvIndex<int, int>;
  using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;

  bool mUseMC = true;
  bool mUseFIT = false;

 public:
  explicit TOFDPLRecoWorkflowTask(bool useMC, bool useFIT) : mUseMC(useMC), mUseFIT(useFIT) {}

  void init(framework::InitContext& ic)
  {
    // nothing special to be set up
    o2::base::GeometryManager::loadGeometry();
    o2::base::Propagator::initFieldFromGRP();
    std::string matLUTPath = ic.options().get<std::string>("material-lut-path");
    std::string matLUTFile = o2::base::NameConf::getMatLUTFileName(matLUTPath);
    if (o2::utils::Str::pathExists(matLUTFile)) {
      auto* lut = o2::base::MatLayerCylSet::loadFromFile(matLUTFile);
      o2::base::Propagator::Instance()->setMatLUT(lut);
      LOG(INFO) << "Loaded material LUT from " << matLUTFile;
    } else {
      LOG(INFO) << "Material LUT " << matLUTFile << " file is absent, only TGeo can be used";
    }

    mTimer.Stop();
    mTimer.Reset();
  }

  void run(framework::ProcessingContext& pc)
  {
    mTimer.Start(false);
    //>>>---------- attach input data --------------->>>
    const auto clustersRO = pc.inputs().get<gsl::span<o2::tof::Cluster>>("tofcluster");
    const auto tracksRO = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("globaltrack");

    if (mUseFIT) {
      // Note: the particular variable will go out of scope, but the span is passed by copy to the
      // worker and the underlying memory is valid throughout the whole computation
      auto recPoints = std::move(pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("fitrecpoints"));
      mMatcher.setFITRecPoints(recPoints);
      LOG(INFO) << "TOF Reco Workflow pulled " << recPoints.size() << " FIT RecPoints";
    }

    // we do a copy of the input but we are looking for a way to avoid it (current problem in conversion form unique_ptr to *)

    gsl::span<const o2::MCCompLabel> itstpclab;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> toflab;
    if (mUseMC) {
      const auto toflabel = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("tofclusterlabel");
      itstpclab = pc.inputs().get<gsl::span<o2::MCCompLabel>>("itstpclabel");
      toflab = std::move(*toflabel);
    }

    mMatcher.run(tracksRO, clustersRO, toflab, itstpclab);

    // in run_match_tof aggiugnere esplicitamente la chiamata a fill del tree (nella classe MatchTOF) e il metodo per leggere i vettori di output

    //...
    // LOG(INFO) << "TOF CLUSTERER : TRANSFORMED " << digits->size()
    //           << " DIGITS TO " << mClustersArray.size() << " CLUSTERS";

    // send matching-info
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHINFOS", 0, Lifetime::Timeframe}, mMatcher.getMatchedTrackVector());
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MCMATCHTOF", 0, Lifetime::Timeframe}, mMatcher.getMatchedTOFLabelsVector());
    }
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CALIBDATA", 0, Lifetime::Timeframe}, mMatcher.getCalibVector());
    mTimer.Stop();
  }

  void endOfStream(EndOfStreamContext& ec)
  {
    LOGF(INFO, "TOF Matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
         mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
  }

 private:
  o2::globaltracking::MatchTOF mMatcher; ///< Cluster finder
  TStopwatch mTimer;
};

o2::framework::DataProcessorSpec getTOFRecoWorkflowSpec(bool useMC, bool useFIT)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  inputs.emplace_back("tofcluster", o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("globaltrack", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("tofclusterlabel", o2::header::gDataOriginTOF, "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("itstpclabel", "GLO", "TPCITS_MC", 0, Lifetime::Timeframe);
  }

  if (useFIT) {
    inputs.emplace_back("fitrecpoints", o2::header::gDataOriginFT0, "RECPOINTS", 0, Lifetime::Timeframe);
  }

  outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHINFOS", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "MCMATCHTOF", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back(o2::header::gDataOriginTOF, "CALIBDATA", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TOFRecoWorkflow",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TOFDPLRecoWorkflowTask>(useMC, useFIT)},
    Options{
      {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // end namespace tof
} // end namespace o2
