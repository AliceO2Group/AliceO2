// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFWorkflow/RecoWorkflowWithTPCSpec.h"
#include "DataFormatsTPC/WorkflowHelper.h"
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
class TOFDPLRecoWorkflowWithTPCTask
{
  using evIdx = o2::dataformats::EvIndex<int, int>;
  using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;

  bool mUseMC = true;
  bool mUseFIT = false;
  bool mDoTPCRefit = false;
  bool mIsCosmics = false;

 public:
  explicit TOFDPLRecoWorkflowWithTPCTask(bool useMC, bool useFIT, bool doTPCRefit, bool iscosmics) : mUseMC(useMC), mUseFIT(useFIT), mDoTPCRefit(doTPCRefit), mIsCosmics(iscosmics) {}

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
    if (mIsCosmics) {
      mMatcher.setCosmics();
    }
    mMatcher.print();

    mTimer.Start(false);
    //>>>---------- attach input data --------------->>>
    const auto clustersRO = pc.inputs().get<gsl::span<o2::tof::Cluster>>("tofcluster");
    const auto tracksRO = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("tracks");

    if (mUseFIT) {
      // Note: the particular variable will go out of scope, but the span is passed by copy to the
      // worker and the underlying memory is valid throughout the whole computation
      auto recPoints = std::move(pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("fitrecpoints"));
      mMatcher.setFITRecPoints(recPoints);
      LOG(INFO) << "TOF Reco WorkflowWithTPC pulled " << recPoints.size() << " FIT RecPoints";
    }

    // we do a copy of the input but we are looking for a way to avoid it (current problem in conversion form unique_ptr to *)

    o2::dataformats::MCTruthContainer<o2::MCCompLabel> toflab;
    gsl::span<const o2::MCCompLabel> tpclab;
    if (mUseMC) {
      const auto toflabel = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("tofclusterlabel");
      tpclab = pc.inputs().get<gsl::span<o2::MCCompLabel>>("tpctracklabel");
      toflab = std::move(*toflabel);
    }

    std::decay_t<decltype(o2::tpc::getWorkflowTPCInput(pc))> inputsTPCclusters;
    if (mDoTPCRefit) {
      mMatcher.setTPCTrackClusIdxInp(pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs"));
      mMatcher.setTPCClustersSharingMap(pc.inputs().get<gsl::span<unsigned char>>("clusTPCshmap"));
      inputsTPCclusters = o2::tpc::getWorkflowTPCInput(pc);
      mMatcher.setTPCClustersInp(&inputsTPCclusters->clusterIndex);
    }

    mMatcher.run(tracksRO, clustersRO, toflab, tpclab);

    auto nmatch = mMatcher.getMatchedTrackVector().size();
    if (mDoTPCRefit) {
      LOG(INFO) << "Refitting " << nmatch << " matched TPC tracks with TOF time info";
    } else {
      LOG(INFO) << "Shifting Z for " << nmatch << " matched TPC tracks according to TOF time info";
    }
    auto& tracksTPCTOF = pc.outputs().make<std::vector<o2::dataformats::TrackTPCTOF>>(OutputRef{"tpctofTracks"}, nmatch);
    mMatcher.makeConstrainedTPCTracks(tracksTPCTOF);

    // in run_match_tof aggiugnere esplicitamente la chiamata a fill del tree (nella classe MatchTOF) e il metodo per leggere i vettori di output

    //...
    // LOG(INFO) << "TOF CLUSTERER : TRANSFORMED " << digits->size()
    //           << " DIGITS TO " << mClustersArray.size() << " CLUSTERS";

    // send matching-info
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHINFOS_TPC", 0, Lifetime::Timeframe}, mMatcher.getMatchedTrackVector());
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MCMATCHTOF_TPC", 0, Lifetime::Timeframe}, mMatcher.getMatchedTOFLabelsVector());
    }
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CALIBDATA_TPC", 0, Lifetime::Timeframe}, mMatcher.getCalibVector());
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

o2::framework::DataProcessorSpec getTOFRecoWorkflowWithTPCSpec(bool useMC, bool useFIT, bool doTPCRefit, bool iscosmics)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  inputs.emplace_back("tofcluster", o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracks", o2::header::gDataOriginTPC, "TRACKS", 0, Lifetime::Timeframe);
  if (doTPCRefit) {
    inputs.emplace_back("trackTPCClRefs", o2::header::gDataOriginTPC, "CLUSREFS", 0, Lifetime::Timeframe);
    inputs.emplace_back("clusTPC", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "CLUSTERNATIVE"}, Lifetime::Timeframe);
    inputs.emplace_back("clusTPCshmap", o2::header::gDataOriginTPC, "CLSHAREDMAP", 0, Lifetime::Timeframe);
  }
  if (useMC) {
    inputs.emplace_back("tofclusterlabel", o2::header::gDataOriginTOF, "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("tpctracklabel", o2::header::gDataOriginTPC, "TRACKSMCLBL", 0, Lifetime::Timeframe);
  }

  if (useFIT) {
    inputs.emplace_back("fitrecpoints", o2::header::gDataOriginFT0, "RECPOINTS", 0, Lifetime::Timeframe);
  }

  outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHINFOS_TPC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"tpctofTracks"}, o2::header::gDataOriginTOF, "TOFTRACKS_TPC", 0, Lifetime::Timeframe);

  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "MCMATCHTOF_TPC", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back(o2::header::gDataOriginTOF, "CALIBDATA_TPC", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TOFRecoWorkflowWithTPC",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TOFDPLRecoWorkflowWithTPCTask>(useMC, useFIT, doTPCRefit, iscosmics)},
    Options{
      {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // end namespace tof
} // end namespace o2
