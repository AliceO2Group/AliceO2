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

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include "Framework/RootSerializationSupport.h"
#include "TPCDigitizerSpec.h"
#include "TPCSectorDigitization.h"
#include "Framework/ControlService.h"
#include "Framework/ParallelContext.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/DeviceSpec.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include <SimulationDataFormat/IOMCTruthContainerView.h>
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "TPCBaseRecSim/CDBInterface.h"
#include "TPCBase/ParameterGEM.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Detector.h"
#include "TPCSpaceCharge/SpaceCharge.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DetectorsBase/Detector.h"
#include "TPCCalibration/VDriftHelper.h"
#include "CommonDataFormat/RangeReference.h"
#include "SimConfig/DigiParams.h"
#include <TRandom.h>
#include <filesystem>
#include "Framework/CCDBParamSpec.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
using DigiGroupRef = o2::dataformats::RangeReference<int, int>;
using SC = o2::tpc::SpaceCharge<float>;

namespace o2
{
namespace tpc
{

template <typename T>
void copyHelper(T const& origin, T& target)
{
  std::copy(origin.begin(), origin.end(), std::back_inserter(target));
}
template <>
void copyHelper<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>(o2::dataformats::MCTruthContainer<o2::MCCompLabel> const& origin, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& target)
{
  target.mergeAtBack(origin);
}

template <typename T>
void writeToBranchHelper(TTree& tree, const char* name, T* accum)
{
  auto targetbr = o2::base::getOrMakeBranch(tree, name, accum);
  targetbr->Fill();
  targetbr->ResetAddress();
  targetbr->DropBaskets("all");
}
template <>
void writeToBranchHelper<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>(TTree& tree,
                                                                             const char* name, o2::dataformats::MCTruthContainer<o2::MCCompLabel>* accum)
{
  // we convert first of all to IOMCTruthContainer
  std::vector<char> buffer;
  accum->flatten_to(buffer);
  accum->clear_andfreememory();
  o2::dataformats::IOMCTruthContainerView view(buffer);
  auto targetbr = o2::base::getOrMakeBranch(tree, name, &view);
  targetbr->Fill();
  targetbr->ResetAddress();
  targetbr->DropBaskets("all");
}

using namespace o2::base;
class TPCDPLDigitizerTask : public BaseDPLDigitizer
{
 public:
  TPCDPLDigitizerTask(bool internalwriter, int distortionType) : mInternalWriter(internalwriter), BaseDPLDigitizer(InitServices::FIELD | InitServices::GEOM), mDistortionType(distortionType)
  {
  }

  void initDigitizerTask(framework::InitContext& ic) override
  {
    LOG(info) << "Initializing TPC digitization";

    mLaneId = ic.services().get<const o2::framework::DeviceSpec>().rank;

    mWithMCTruth = o2::conf::DigiParams::Instance().mctruth;
    auto triggeredMode = ic.options().get<bool>("TPCtriggered");
    mConditions.init(ic, mDistortionType);

    mDigitizer.setContinuousReadout(!triggeredMode);
    mDigitizer.setDistortionScaleType(mDistortionType);

    if (ic.options().get<bool>("tpc-sector-seeding")) {
      const auto seed = o2::conf::DigiParams::Instance().seed;
      if (seed == 0) {
        LOG(warning) << "TPC sector seeding with DigiParams.seed=0 is not reproducible";
      }
      // with seed 0, gRandom was seeded randomly in the base class
      mDigitizer.setUseSectorSeeding(seed != 0 ? seed : gRandom->Integer(kMaxUInt));
      LOG(info) << "TPC: using sector seeding";
    }

    // we send the GRP data once if the corresponding output channel is available
    // and set the flag to false after
    mWriteGRP = true;

    // clean up (possibly) existing digit files
    if (mInternalWriter) {
      cleanDigitFile();
    }
  }

  void cleanDigitFile()
  {
    // since we update digit files during ordinary processing
    // it is better to remove possibly existing files in the same dir
    std::stringstream tmp;
    tmp << "tpc_driftime_digits_lane" << mLaneId << ".root";
    if (std::filesystem::exists(tmp.str())) {
      std::filesystem::remove(tmp.str());
    }
  }

  void writeToROOTFile()
  {
    if (!mInternalROOTFlushFile) {
      std::stringstream tmp;
      tmp << "tpc_driftime_digits_lane" << mLaneId << ".root";
      mInternalROOTFlushFile = new TFile(tmp.str().c_str(), "UPDATE");
      std::stringstream trname;
      trname << mSector;
      mInternalROOTFlushTTree = new TTree(trname.str().c_str(), "o2sim");
    }
    {
      std::stringstream brname;
      brname << "TPCDigit_" << mSector;
      auto br = o2::base::getOrMakeBranch(*mInternalROOTFlushTTree, brname.str().c_str(), &mFlushBuffers.digits);
      br->Fill();
      br->ResetAddress();
    }
    if (mWithMCTruth) {
      // labels
      std::stringstream brname;
      brname << "TPCDigitMCTruth_" << mSector;
      auto br = o2::base::getOrMakeBranch(*mInternalROOTFlushTTree, brname.str().c_str(), &mFlushBuffers.labels);
      br->Fill();
      br->ResetAddress();
    }
    {
      // common
      std::stringstream brname;
      brname << "TPCCommonMode_" << mSector;
      auto br = o2::base::getOrMakeBranch(*mInternalROOTFlushTTree, brname.str().c_str(), &mFlushBuffers.commonMode);
      br->Fill();
      br->ResetAddress();
    }
  }

  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj)
  {
    mConditions.finaliseCCDB(matcher, obj, mDigitizer);
  }

  void run(framework::ProcessingContext& pc)
  {
    LOG(info) << "Processing TPC digitization";

    mConditions.update(pc, mDigitizer, [this](float vdrift, float tdriftOffset) {
      mDigitizer.setVDrift(vdrift);
      mDigitizer.setTDriftOffset(tdriftOffset);
    });

    for (auto it = pc.inputs().begin(), end = pc.inputs().end(); it != end; ++it) {
      for (auto const& inputref : it.parts()) {
        if (inputref.spec->lifetime == o2::framework::Lifetime::Condition) { // process does not need conditions
          continue;
        }
        process(pc, inputref);
        if (mInternalWriter) {
          mInternalROOTFlushTTree->SetEntries(mFlushCounter);
          mInternalROOTFlushFile->Write("", TObject::kOverwrite);
          mInternalROOTFlushFile->Close();
          // delete mInternalROOTFlushTTree; --> automatically done by ->Close()
          delete mInternalROOTFlushFile;
          mInternalROOTFlushFile = nullptr;
        }
        // TODO: make generic reset method?
        mFlushCounter = 0;
      }
    }
  }

  // process one sector
  void process(framework::ProcessingContext& pc, framework::DataRef const& inputref)
  {
    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>(inputref);
    context->initSimChains(o2::detectors::DetID::TPC, mSimChains);
    auto& irecords = context->getEventRecords();
    // A timeframe holds no collision at all whenever the interaction rate is low enough. Do not
    // return here: the sector still has to produce its usual output, an empty one, so that the
    // digit file has the same shape as an ordinary timeframe that happens to contain nothing.
    // Returning also left the chunked writer without a file and a null tree to flush.
    LOG(info) << "TPC: Processing " << irecords.size() << " collisions";
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(inputref);

    bool isContinuous = mDigitizer.isContinuousReadout();
    // we publish the GRP data once if the output channel is there
    if (mWriteGRP && pc.outputs().isAllowed({"TPC", "ROMode", 0})) {
      auto roMode = isContinuous ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
      LOG(info) << "TPC: Sending ROMode= " << (mDigitizer.isContinuousReadout() ? "Continuous" : "Triggered")
                << " to GRPUpdater from channel " << dh->subSpecification;
      pc.outputs().snapshot(Output{"TPC", "ROMode", 0}, roMode);
    }
    mWriteGRP = false;

    // extract which sector to treat
    auto const* sectorHeader = DataRefUtils::getHeader<TPCSectorHeader*>(inputref);
    if (sectorHeader == nullptr) {
      LOG(error) << "TPC: Sector header missing, skipping processing";
      return;
    }
    auto sector = sectorHeader->sector();
    mSector = sector;
    mListOfSectors.push_back(sector);
    LOG(info) << "TPC: Processing sector " << sector;
    // the active sectors need to be propagated
    uint64_t activeSectors = 0;
    activeSectors = sectorHeader->activeSectors;

    // lambda that creates a DPL owned buffer to accumulate the digits (in shared memory)
    auto makeDigitBuffer = [this, sector, &pc, activeSectors, &dh]() {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      if (mInternalWriter) {
        using ContainerType = std::decay_t<decltype(pc.outputs().make<std::vector<o2::tpc::Digit>>(Output{"", "", 0}))>*;
        return ContainerType(nullptr);
      } else {
        // default case
        return &pc.outputs().make<std::vector<o2::tpc::Digit>>(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(dh->subSpecification), header});
      }
    };
    // lambda that snapshots the common mode vector to be sent out; prepares and attaches header with sector information
    auto snapshotCommonMode = [this, sector, &pc, activeSectors, &dh](std::vector<o2::tpc::CommonMode> const& commonMode) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      if (!mInternalWriter) {
        // note that snapshoting only works with non-const references (to be fixed?)
        pc.outputs().snapshot(Output{"TPC", "COMMONMODE", static_cast<SubSpecificationType>(dh->subSpecification), header},
                              const_cast<std::vector<o2::tpc::CommonMode>&>(commonMode));
      }
    };
    // lambda that snapshots labels to be sent out; prepares and attaches header with sector information
    auto snapshotLabels = [this, &sector, &pc, activeSectors, &dh](o2::dataformats::MCTruthContainer<o2::MCCompLabel> const& labels) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      if (mWithMCTruth) {
        if (!mInternalWriter) {
          auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{"TPC", "DIGITSMCTR", static_cast<SubSpecificationType>(dh->subSpecification), header});
          labels.flatten_to(sharedlabels);
        }
      }
    };
    // lambda that snapshots digits grouping (triggers) to be sent out; prepares and attaches header with sector information
    auto snapshotEvents = [this, sector, &pc, activeSectors, &dh](const std::vector<DigiGroupRef>& events) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      if (!mInternalWriter) {
        LOG(info) << "TPC: Send TRIGGERS for sector " << sector << " channel " << dh->subSpecification << " | size " << events.size();
        pc.outputs().snapshot(Output{"TPC", "DIGTRIGGERS", static_cast<SubSpecificationType>(dh->subSpecification), header},
                              const_cast<std::vector<DigiGroupRef>&>(events));
      }
    };

    auto digitsAccum = makeDigitBuffer();                          // accumulator for digits
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelAccum; // timeframe accumulator for labels
    std::vector<CommonMode> commonModeAccum;
    std::vector<DigiGroupRef> eventAccum;

    // this should not happen any more, legacy condition when the sector variable was used
    // to transport control information
    if (sector < 0) {
      throw std::runtime_error("Legacy control information is not expected any more");
    }

    // the TPCSectorHeader now allows to transport information for more than one sector,
    // e.g. for transporting clusters in one single data block. The digitization is however
    // only on sector level
    if (sector >= TPCSectorHeader::NSectors) {
      throw std::runtime_error("Digitizer can only work on single sectors");
    }

    TStopwatch timer;
    timer.Start();

    auto onFlush = [this, digitsAccum, &labelAccum, &commonModeAccum]() {
      mFlushCounter++;
      if (mInternalWriter) {
        // the natural place to write out this independent datachunk immediately ...
        writeToROOTFile();
      } else {
        // ... or to accumulate and later forward to next DPL proc
        std::copy(mFlushBuffers.digits.begin(), mFlushBuffers.digits.end(), std::back_inserter(*digitsAccum));
        if (mWithMCTruth) {
          labelAccum.mergeAtBack(mFlushBuffers.labels);
        }
        std::copy(mFlushBuffers.commonMode.begin(), mFlushBuffers.commonMode.end(), std::back_inserter(commonModeAccum));
      }
    };
    eventAccum = digitizeSector(mDigitizer, *context, mSimChains, sector, dh->tfCounter, mFlushBuffers, onFlush);

    if (!mInternalWriter) {
      // send out to next stage
      snapshotEvents(eventAccum);
      // snapshotDigits(digitsAccum); --> done automatically
      snapshotCommonMode(commonModeAccum);
      snapshotLabels(labelAccum);
    }

    timer.Stop();
    LOG(info) << "TPC: Digitization took " << timer.CpuTime() << "s";
  }

 private:
  o2::tpc::Digitizer mDigitizer;
  TPCDigitizerConditions mConditions;
  std::vector<TChain*> mSimChains;
  SectorFlushBuffers mFlushBuffers; // output of the latest flush
  std::vector<int> mListOfSectors;  //  a list of sectors treated by this task
  TFile* mInternalROOTFlushFile = nullptr;
  TTree* mInternalROOTFlushTTree = nullptr;
  size_t mFlushCounter = 0;
  int mLaneId = 0; // the id of the current process within the parallel pipeline
  int mSector = 0;
  bool mWriteGRP = false;
  bool mWithMCTruth = true;
  bool mInternalWriter = false;
  int mDistortionType = 0;
};

o2::framework::DataProcessorSpec getTPCDigitizerSpec(int channel, bool writeGRP, bool mctruth, bool internalwriter, int distortionType)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::stringstream id;
  id << "TPCDigitizer" << channel;

  std::vector<OutputSpec> outputs; // define channel by triple of (origin, type id of data to be sent on this channel, subspecification)

  if (!internalwriter) {
    outputs.emplace_back("TPC", "DIGITS", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
    outputs.emplace_back("TPC", "DIGTRIGGERS", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
    if (mctruth) {
      outputs.emplace_back("TPC", "DIGITSMCTR", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
    }
    outputs.emplace_back("TPC", "COMMONMODE", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  }
  if (writeGRP) {
    outputs.emplace_back("TPC", "ROMode", 0, Lifetime::Timeframe);
    LOG(debug) << "TPC: Channel " << channel << " will supply ROMode";
  }

  std::vector<InputSpec> inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}};
  auto options = TPCDigitizerConditions::options();
  options.push_back({"tpc-sector-seeding", VariantType::Bool, false, {"Seed the random numbers of each sector from DigiParams.seed and the sector, so that the digits do not depend on the lane assignment"}});
  return DataProcessorSpec{
    id.str().c_str(),
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCDPLDigitizerTask>(internalwriter, distortionType)},
    options};
}

o2::framework::WorkflowSpec getTPCDigitizerSpec(int nLanes, std::vector<int> const& sectors, bool mctruth, bool internalwriter, int distortionType)
{
  // channel parameter is deprecated in the TPCDigitizer processor, all descendants
  // are initialized not to publish GRP mode, but the channel will be added to the first
  // processor after the pipelines have been created. The processor will decide upon
  // the index in the ParallelContext whether to publish
  WorkflowSpec pipelineTemplate{getTPCDigitizerSpec(0, false, mctruth, internalwriter, distortionType)};
  // override the predefined name, index will be added by parallelPipeline method
  pipelineTemplate[0].name = "TPCDigitizer";
  WorkflowSpec pipelines = parallelPipeline(
    pipelineTemplate, nLanes, [size = sectors.size()]() { return size; }, [&sectors](size_t index) { return sectors[index]; });
  // add the channel for the GRP information to the first processor
  for (auto& spec : pipelines) {
    TPCDigitizerConditions::addInputs(spec.inputs, distortionType); // the same CCDB request for each pipeline
  }
  pipelines[0].outputs.emplace_back("TPC", "ROMode", 0, Lifetime::Timeframe);
  return pipelines;
}

} // end namespace tpc
} // end namespace o2
