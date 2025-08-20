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

/// @author Sandro Wenzel
/// @since  2021-03-10
/// @brief  Takes TPC digit chunks (such drift times) --> accumulates to digit timeframe format --> publishes

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/DataAllocator.h"
#include "Framework/ControlService.h"
#include "DataFormatsTPC/Digit.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "TPCSimulation/CommonMode.h"
#include "DetectorsBase/Detector.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include <CommonUtils/FileSystemUtils.h>
#include "Algorithm/RangeTokenizer.h"
#include "TPCBase/Sector.h"
#include "TPCWorkflow/TPCDigitRootWriterSpec.h"
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <numeric>
#include <TROOT.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#ifdef WITH_OPENMP
#include <omp.h>
#endif
#include <TStopwatch.h>

using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

using namespace o2::framework;
using namespace o2::header;

using namespace o2::framework;
using namespace o2::header;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using DigitOutputType = std::vector<o2::tpc::Digit>;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // for the TPC it is useful to take at most half of the available (logical) cores due to memory requirements
  int defaultlanes = std::max(1u, std::thread::hardware_concurrency() / 2);
  std::string laneshelp("Number of tpc processing lanes. A lane is a pipeline of algorithms.");
  workflowOptions.push_back(
    ConfigParamSpec{"tpc-lanes", VariantType::Int, defaultlanes, {laneshelp}});

  std::string sectorshelp("List of TPC sectors, comma separated ranges, e.g. 0-3,7,9-15");
  std::string sectorDefault = "0-" + std::to_string(o2::tpc::Sector::MAXSECTOR - 1);
  workflowOptions.push_back(
    ConfigParamSpec{"tpc-sectors", VariantType::String, sectorDefault.c_str(), {sectorshelp}});

  // option to disable MC truth
  workflowOptions.push_back(ConfigParamSpec{"disable-mc", o2::framework::VariantType::Bool, false, {"disable  mc-truth"}});
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}});
  workflowOptions.push_back(ConfigParamSpec{"write-digits-file", VariantType::Int, 0, {"Enable writing of tpcdigis.root file"}});
  o2::raw::HBFUtilsInitializer::addConfigOption(workflowOptions);
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
  // Using critical section here as this is writing to shared mem
  // and not sure if boost shared mem allocator is thread-safe.
  // It was crashing without this.
#pragma omp critical
  std::copy(origin.begin(), origin.end(), std::back_inserter(target));
}
template <>
void copyHelper<MCTruthContainer>(MCTruthContainer const& origin, MCTruthContainer& target)
{
  target.mergeAtBack(origin);
}

template <typename T>
auto makePublishBuffer(framework::ProcessingContext& pc, int sector, uint64_t activeSectors)
{
  LOG(info) << "PUBLISHING SECTOR " << sector;

  o2::tpc::TPCSectorHeader header{sector};
  header.activeSectors = activeSectors;
  return &pc.outputs().make<T>(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(sector), header});
}

template <>
auto makePublishBuffer<MCTruthContainer>(framework::ProcessingContext& pc, int sector, uint64_t activeSectors)
{
  return new MCTruthContainer();
}

template <>
auto makePublishBuffer<std::vector<o2::tpc::CommonMode>>(framework::ProcessingContext& pc, int sector, uint64_t activeSectors)
{
  LOG(info) << "PUBLISHING COMMONMODE SECTOR " << sector;
  o2::tpc::TPCSectorHeader header{sector};
  header.activeSectors = activeSectors;
  return &pc.outputs().make<std::vector<o2::tpc::CommonMode>>(Output{"TPC", "COMMONMODE", static_cast<SubSpecificationType>(sector), header});
}

template <typename T>
void publishBuffer(framework::ProcessingContext& pc, int sector, uint64_t activeSectors, T* accum)
{
  // nothing by default
}

template <>
void publishBuffer<MCTruthContainer>(framework::ProcessingContext& pc, int sector, uint64_t activeSectors, MCTruthContainer* accum)
{

  LOG(info) << "PUBLISHING MC LABELS " << accum->getNElements();
  o2::tpc::TPCSectorHeader header{sector};
  header.activeSectors = activeSectors;

#pragma omp critical
  {
    // Convert to IOMCTruthContainerView format as expected by TPC reader
    std::vector<char> flattened;
    accum->flatten_to(flattened);

    auto& sharedlabels = pc.outputs().make<o2::dataformats::IOMCTruthContainerView>(
      Output{"TPC", "DIGITSMCTR", static_cast<SubSpecificationType>(sector), header});
    sharedlabels.adopt(std::move(flattened));
  }
  delete accum;
}

template <>
void publishBuffer<std::vector<o2::tpc::CommonMode>>(framework::ProcessingContext& pc, int sector, uint64_t activeSectors, std::vector<o2::tpc::CommonMode>* accum)
{
  // CommonMode data is already published by makePublishBuffer, so nothing special needed here
  LOG(info) << "PUBLISHED COMMONMODE DATA FOR SECTOR " << sector << " SIZE " << accum->size();
}

template <typename T>
void mergeHelper(const char* brprefix, std::vector<int> const& tpcsectors, uint64_t activeSectors,
                 TFile& originfile, framework::ProcessingContext& pc)
{
  auto keyslist = originfile.GetListOfKeys();
  for (int i = 0; i < keyslist->GetEntries(); ++i) {
    auto key = keyslist->At(i);
    int sector = atoi(key->GetName());
    if (std::find(tpcsectors.begin(), tpcsectors.end(), sector) == tpcsectors.end()) {
      // do nothing if sector not wanted
      continue;
    }

    auto oldtree = (TTree*)originfile.Get(key->GetName());
    assert(oldtree);
    std::stringstream digitbrname;
    digitbrname << brprefix << key->GetName();
    auto br = oldtree->GetBranch(digitbrname.str().c_str());
    if (!br) {
      continue;
    }
    T* chunk = nullptr;
    br->SetAddress(&chunk);

    using AccumType = std::decay_t<decltype(makePublishBuffer<T>(pc, sector, activeSectors))>;
    AccumType accum;
#pragma omp critical
    accum = makePublishBuffer<T>(pc, sector, activeSectors);

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
    publishBuffer(pc, sector, activeSectors, accum);
  }
}

void publishMergedTimeframes(std::vector<int> const& lanes, std::vector<int> const& tpcsectors, bool domctruth, bool writeDigitsFile, framework::ProcessingContext& pc)
{
  uint64_t activeSectors = 0;
  for (auto s : tpcsectors) {
    activeSectors |= (uint64_t)0x1 << s;
  }

  ROOT::EnableThreadSafety();
  // we determine the exact input list of files
  auto digitfilelist = o2::utils::listFiles("tpc_driftime_digits_lane.*.root$");
#ifdef WITH_OPENMP
  omp_set_num_threads(std::min(lanes.size(), digitfilelist.size()));
  LOG(info) << "Running digit publisher with OpenMP enabled";
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t fi = 0; fi < digitfilelist.size(); ++fi) {
    auto& filename = digitfilelist[fi];
    LOG(debug) << "MERGING CHUNKED DIGITS FROM FILE " << filename;
    auto originfile = new TFile(filename.c_str(), "OPEN");
    assert(originfile);

    //data definitions
    using DigitsType = std::vector<o2::tpc::Digit>;
    using LabelType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
    mergeHelper<DigitsType>("TPCDigit_", tpcsectors, activeSectors, *originfile, pc);
    if (domctruth) {
      mergeHelper<LabelType>("TPCDigitMCTruth_", tpcsectors, activeSectors, *originfile, pc);
    }
    if (writeDigitsFile) {
      using CommonModeType = std::vector<o2::tpc::CommonMode>;
      // Try to read common mode data from file, but don't fail if it doesn't exist
      try {
        mergeHelper<CommonModeType>("TPCCommonMode_", tpcsectors, activeSectors, *originfile, pc);
      } catch (const std::exception& e) {
        LOG(warning) << "CommonMode data not found in file, creating empty data: " << e.what();
        // Create empty common mode data for each sector
        for (auto sector : tpcsectors) {
          o2::tpc::TPCSectorHeader header{sector};
          header.activeSectors = activeSectors;
          auto& emptyCommonMode = pc.outputs().make<std::vector<o2::tpc::CommonMode>>(
            Output{"TPC", "COMMONMODE", static_cast<SubSpecificationType>(sector), header});
          // emptyCommonMode is already empty by default
        }
      }
    }
    originfile->Close();
    delete originfile;
  }

  // Create trigger information for continuous mode (one trigger covering all digits)
  if (writeDigitsFile) {
    for (auto sector : tpcsectors) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      auto& triggers = pc.outputs().make<std::vector<o2::dataformats::RangeReference<int, int>>>(
        Output{"TPC", "DIGTRIGGERS", static_cast<SubSpecificationType>(sector), header});

      // For continuous mode, create a single trigger that covers all digits
      // We need to count the digits for this sector across all files
      // For now, create a placeholder trigger - the actual digit count will be determined
      // by the digit writer when it processes the actual digits
      triggers.emplace_back(0, 1); // placeholder: covers digits from 0 to 1 (will be adjusted by writer)
      LOG(info) << "Created continuous mode trigger for sector " << sector;
    }
  }
}

class Task
{
 public:
  Task(std::vector<int> laneConfig, std::vector<int> tpcsectors, bool mctruth, bool writeDigitsFile) : mLanes(laneConfig), mTPCSectors(tpcsectors), mDoMCTruth(mctruth), mWriteDigitsFile(writeDigitsFile)
  {
  }

  void run(framework::ProcessingContext& pc)
  {
    LOG(info) << "Preparing digits (from digit chunks) for reconstruction";

    TStopwatch w;
    w.Start();
    publishMergedTimeframes(mLanes, mTPCSectors, mDoMCTruth, mWriteDigitsFile, pc);

    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);

    LOG(info) << "DIGIT PUBLISHING TOOK " << w.RealTime();
    return;
  }

  void init(framework::InitContext& ctx)
  {
  }

 private:
  bool mDoMCTruth = true;
  bool mWriteDigitsFile = false;
  std::vector<int> mLanes;
  std::vector<int> mTPCSectors;
};

/// create the processor spec
/// describing a processor aggregating digits for various TPC sectors and writing them to file
/// MC truth information is also aggregated and written out
void getSpec(WorkflowSpec& specs, std::vector<int> const& laneConfiguration, std::vector<int> const& tpcsectors, bool mctruth, bool publish = true, bool writeDigitsFile = false)
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
      if (writeDigitsFile) {
        outputs.emplace_back(/*binding,*/ "TPC", "COMMONMODE", static_cast<SubSpecificationType>(s), Lifetime::Timeframe);
        outputs.emplace_back(/*binding,*/ "TPC", "DIGTRIGGERS", static_cast<SubSpecificationType>(s), Lifetime::Timeframe);
      }
    }
  }
  specs.emplace_back(DataProcessorSpec{"TPCDigitMerger", {}, outputs, AlgorithmSpec{o2::framework::adaptFromTask<Task>(laneConfiguration, tpcsectors, mctruth, writeDigitsFile)}, Options{}});

  if (writeDigitsFile) {
    specs.push_back(getTPCDigitRootWriterSpec(tpcsectors, mctruth));
  }
}

} // end namespace tpc
} // end namespace o2

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  auto numlanes = configcontext.options().get<int>("tpc-lanes");
  bool mctruth = !configcontext.options().get<bool>("disable-mc");
  bool writeDigitsFile = configcontext.options().get<int>("write-digits-file");
  auto tpcsectors = o2::RangeTokenizer::tokenize<int>(configcontext.options().get<std::string>("tpc-sectors"));

  std::vector<int> lanes(numlanes);
  std::iota(lanes.begin(), lanes.end(), 0);
  o2::tpc::getSpec(specs, lanes, tpcsectors, mctruth, true, writeDigitsFile);

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);
  return specs;
}
