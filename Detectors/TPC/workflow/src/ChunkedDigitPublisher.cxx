// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @author Sandro Wenzel
/// @since  2021-03-10
/// @brief  Takes TPC digit chunks (such drift times) --> accumulates to digit timeframe format --> publishes

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/DataAllocator.h"
#include "Framework/ControlService.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCSimulation/CommonMode.h"
#include "DetectorsBase/Detector.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

using namespace o2::framework;
using namespace o2::header;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // for the TPC it is useful to take at most half of the available (logical) cores due to memory requirements
  int defaultlanes = std::max(1u, std::thread::hardware_concurrency() / 2);
  std::string laneshelp("Number of tpc processing lanes. A lane is a pipeline of algorithms.");
  workflowOptions.push_back(
    ConfigParamSpec{"tpc-lanes", VariantType::Int, defaultlanes, {laneshelp}});

  // option to disable MC truth
  workflowOptions.push_back(ConfigParamSpec{"disable-mc", o2::framework::VariantType::Bool, false, {"disable  mc-truth"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"
#include "DataFormatsTPC/TPCSectorHeader.h"

using MCTruthContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

namespace o2
{
namespace tpc
{

template <typename T, typename R>
void copyHelper(T const& origin, R& target)
{
  std::copy(origin.begin(), origin.end(), std::back_inserter(target));
}
template <>
void copyHelper<MCTruthContainer>(MCTruthContainer const& origin, MCTruthContainer& target)
{
  target.mergeAtBack(origin);
}

template <typename T>
auto makePublishBuffer(framework::ProcessingContext& pc, int sector)
{
  LOG(INFO) << "PUBLISHING SECTOR " << sector;
  uint64_t activeSectors = 0;
  for (int i = 0; i < 36; ++i) {
    activeSectors |= (uint64_t)0x1 << i;
  }

  o2::tpc::TPCSectorHeader header{sector};
  header.activeSectors = activeSectors;
  return &pc.outputs().make<T>(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(sector), Lifetime::Timeframe,
                                      header});
}

template <>
auto makePublishBuffer<MCTruthContainer>(framework::ProcessingContext& pc, int sector)
{
  return new MCTruthContainer();
}

template <typename T>
void publishBuffer(framework::ProcessingContext& pc, int sector, T* accum)
{
  // nothing by default
}

template <>
void publishBuffer<MCTruthContainer>(framework::ProcessingContext& pc, int sector, MCTruthContainer* accum)
{
  uint64_t activeSectors = 0;
  for (int i = 0; i < 36; ++i) {
    activeSectors |= (uint64_t)0x1 << i;
  }

  LOG(INFO) << "PUBLISHING MC LABELS " << accum->getNElements();
  o2::tpc::TPCSectorHeader header{sector};
  header.activeSectors = activeSectors;
  auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(
    Output{"TPC", "DIGITSMCTR", static_cast<SubSpecificationType>(sector), Lifetime::Timeframe, header});
  accum->flatten_to(sharedlabels);
  delete accum;
}

void publishMergedTimeframes(std::vector<int> const& lanes, bool domctruth, framework::ProcessingContext& pc)
{
  for (auto l : lanes) {
    // TODO: we could put each these blocks into a separate thread !

    LOG(DEBUG) << "MERGING CHUNKED DIGITS FOR LANE " << l;
    // merging the data
    std::stringstream tmp;
    tmp << "tpc_driftime_digits_lane" << l << ".root";
    auto originfile = new TFile(tmp.str().c_str(), "OPEN");
    assert(originfile);

    auto merge = [originfile, &pc](auto data, auto brprefix) {
      auto keyslist = originfile->GetListOfKeys();
      for (int i = 0; i < keyslist->GetEntries(); ++i) {
        auto key = keyslist->At(i);
        auto oldtree = (TTree*)originfile->Get(key->GetName());
        assert(oldtree);
        std::stringstream digitbrname;
        digitbrname << brprefix << key->GetName();
        auto br = oldtree->GetBranch(digitbrname.str().c_str());
        if (!br) {
          continue;
        }
        decltype(data)* chunk = nullptr;
        br->SetAddress(&chunk);

        int sector = atoi(key->GetName());
        auto accum = makePublishBuffer<decltype(data)>(pc, sector);

        for (auto e = 0; e < br->GetEntries(); ++e) {
          br->GetEntry(e);
          copyHelper(*chunk, *accum);
          delete chunk;
          chunk = nullptr;
        }
        br->ResetAddress();
        br->DropBaskets("all");
        delete oldtree;

        // some data (labels are published slightly differently)
        publishBuffer(pc, sector, accum);
      }
    };

    //data definitions
    using DigitsType = std::vector<o2::tpc::Digit>;
    using LabelType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
    merge(DigitsType(), "TPCDigit_");
    if (domctruth) {
      merge(LabelType(), "TPCDigitMCTruth_");
    }
    originfile->Close();
    delete originfile;
  }
}

class Task
{
 public:
  Task(std::vector<int> laneConfig, bool mctruth) : mLanes(laneConfig), mDoMCTruth(mctruth)
  {
  }

  void run(framework::ProcessingContext& pc)
  {
    LOG(INFO) << "Preparing digits (from digit chunks) for reconstruction";

    publishMergedTimeframes(mLanes, mDoMCTruth, pc);

    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    return;
  }

  void init(framework::InitContext& ctx)
  {
  }

 private:
  bool mDoMCTruth = true;
  std::vector<int> mLanes;
};

/// create the processor spec
/// describing a processor aggregating digits for various TPC sectors and writing them to file
/// MC truth information is also aggregated and written out
DataProcessorSpec getSpec(std::vector<int> const& laneConfiguration, bool mctruth, bool publish = true)
{
  //data definitions
  using DigitsOutputType = std::vector<o2::tpc::Digit>;
  using CommonModeOutputType = std::vector<o2::tpc::CommonMode>;

  std::vector<OutputSpec> outputs; // define channel by triple of (origin, type id of data to be sent on this channel, subspecification)
  if (publish) {
    // effectively the input expects one sector per subspecification
    for (int s = 0; s < 36; ++s) {
      OutputLabel binding{std::to_string(s)};
      outputs.emplace_back(/*binding,*/ "TPC", "DIGITS", static_cast<SubSpecificationType>(s), Lifetime::Timeframe);
      if (mctruth) {
        outputs.emplace_back(/*binding,*/ "TPC", "DIGITSMCTR", static_cast<SubSpecificationType>(s), Lifetime::Timeframe);
      }
    }
  }

  return DataProcessorSpec{
    "TPCDigitMerger", {}, outputs, AlgorithmSpec{o2::framework::adaptFromTask<Task>(laneConfiguration, mctruth)}, Options{}};
}

} // end namespace tpc
} // end namespace o2

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  auto numlanes = configcontext.options().get<int>("tpc-lanes");
  bool mctruth = !configcontext.options().get<bool>("disable-mc");

  std::vector<int> lanes(numlanes);
  std::iota(lanes.begin(), lanes.end(), 0);
  specs.emplace_back(o2::tpc::getSpec(lanes, mctruth));
  return specs;
}
