// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TPCDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ParallelContext.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"
#include "TSystem.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "TPCBase/CDBInterface.h"
#include "TPCBase/Digit.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Detector.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "CommonDataFormat/RangeReference.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "SimConfig/DigiParams.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
using DigiGroupRef = o2::dataformats::RangeReference<int, int>;

namespace o2
{
namespace tpc
{

std::string getBranchNameLeft(int sector)
{
  std::stringstream branchnamestreamleft;
  branchnamestreamleft << "TPCHitsShiftedSector" << int(o2::tpc::Sector::getLeft(o2::tpc::Sector(sector)));
  return branchnamestreamleft.str();
}

std::string getBranchNameRight(int sector)
{
  std::stringstream branchnamestreamright;
  branchnamestreamright << "TPCHitsShiftedSector" << sector;
  return branchnamestreamright.str();
}

using namespace o2::base;
class TPCDPLDigitizerTask : public BaseDPLDigitizer
{
 public:
  TPCDPLDigitizerTask() : BaseDPLDigitizer(InitServices::FIELD | InitServices::GEOM)
  {
  }

  void initDigitizerTask(framework::InitContext& ic) override
  {
    LOG(INFO) << "Initializing TPC digitization";

    auto useDistortions = ic.options().get<int>("distortionType");
    auto triggeredMode = ic.options().get<bool>("TPCtriggered");

    if (useDistortions > 0) {
      if (useDistortions == 1) {
        LOG(INFO) << "Using realistic space-charge distortions.";
      } else {
        LOG(INFO) << "Using constant space-charge distortions.";
      }
      auto readSpaceChargeString = ic.options().get<std::string>("readSpaceCharge");
      std::vector<std::string> readSpaceCharge;
      std::stringstream ssSpaceCharge(readSpaceChargeString);
      while (ssSpaceCharge.good()) {
        std::string substr;
        getline(ssSpaceCharge, substr, ',');
        readSpaceCharge.push_back(substr);
      }
      if (readSpaceCharge[0].size() != 0) { // use pre-calculated space-charge object
        std::unique_ptr<o2::tpc::SpaceCharge> spaceCharge;
        if (!gSystem->AccessPathName(readSpaceCharge[0].data())) {
          auto fileSC = std::unique_ptr<TFile>(TFile::Open(readSpaceCharge[0].data()));
          if (fileSC->FindKey(readSpaceCharge[1].data())) {
            spaceCharge.reset((o2::tpc::SpaceCharge*)fileSC->Get(readSpaceCharge[1].data()));
          }
        }
        if (spaceCharge.get() != nullptr) {
          LOG(INFO) << "Using pre-calculated space-charge object: " << readSpaceCharge[1].data();
          mDigitizer.setUseSCDistortions(spaceCharge.release());
        } else {
          LOG(ERROR) << "Space-charge object or file not found!";
        }
      } else { // create new space-charge object either with empty TPC or an initial space-charge density provided by histogram
        o2::tpc::SpaceCharge::SCDistortionType distortionType = useDistortions == 2 ? o2::tpc::SpaceCharge::SCDistortionType::SCDistortionsConstant : o2::tpc::SpaceCharge::SCDistortionType::SCDistortionsRealistic;
        auto gridSizeString = ic.options().get<std::string>("gridSize");
        std::vector<int> gridSize;
        std::stringstream ss(gridSizeString);
        while (ss.good()) {
          std::string substr;
          getline(ss, substr, ',');
          gridSize.push_back(std::stoi(substr));
        }
        auto inputHistoString = ic.options().get<std::string>("initialSpaceChargeDensity");
        std::vector<std::string> inputHisto;
        std::stringstream ssHisto(inputHistoString);
        while (ssHisto.good()) {
          std::string substr;
          getline(ssHisto, substr, ',');
          inputHisto.push_back(substr);
        }
        std::unique_ptr<TH3> hisSCDensity;
        if (!gSystem->AccessPathName(inputHisto[0].data())) {
          auto fileSCInput = std::unique_ptr<TFile>(TFile::Open(inputHisto[0].data()));
          if (fileSCInput->FindKey(inputHisto[1].data())) {
            hisSCDensity.reset((TH3*)fileSCInput->Get(inputHisto[1].data()));
            hisSCDensity->SetDirectory(nullptr);
          }
        }
        if (hisSCDensity.get() != nullptr) {
          LOG(INFO) << "TPC: Providing initial space-charge density histogram: " << hisSCDensity->GetName();
          mDigitizer.setUseSCDistortions(distortionType, hisSCDensity.get(), gridSize[0], gridSize[1], gridSize[2]);
        } else {
          if (distortionType == SpaceCharge::SCDistortionType::SCDistortionsConstant) {
            LOG(ERROR) << "Input space-charge density histogram or file not found!";
          }
        }
      }
    }
    mDigitizer.setContinuousReadout(!triggeredMode);

    // we send the GRP data once if the corresponding output channel is available
    // and set the flag to false after
    mWriteGRP = true;
  }

  void run(framework::ProcessingContext& pc)
  {
    LOG(INFO) << "Processing TPC digitization";

    /// For the time being use the defaults for the CDB
    auto& cdb = o2::tpc::CDBInterface::instance();
    cdb.setUseDefaults();
    if (!gSystem->AccessPathName("GainMap.root")) {
      LOG(INFO) << "TPC: Using gain map from 'GainMap.root'";
      cdb.setGainMapFromFile("GainMap.root");
    }

    for (auto it = pc.inputs().begin(), end = pc.inputs().end(); it != end; ++it) {
      for (auto const& inputref : it) {
        process(pc, inputref);
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
    LOG(INFO) << "TPC: Processing " << irecords.size() << " collisions";
    if (irecords.size() == 0) {
      return;
    }
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(inputref);

    bool isContinuous = mDigitizer.isContinuousReadout();
    // we publish the GRP data once if the output channel is there
    if (mWriteGRP && pc.outputs().isAllowed({"TPC", "ROMode", 0})) {
      auto roMode = isContinuous ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
      LOG(INFO) << "TPC: Sending ROMode= " << (mDigitizer.isContinuousReadout() ? "Continuous" : "Triggered")
                << " to GRPUpdater from channel " << dh->subSpecification;
      pc.outputs().snapshot(Output{"TPC", "ROMode", 0, Lifetime::Timeframe}, roMode);
    }
    mWriteGRP = false;

    // extract which sector to treat
    auto const* sectorHeader = DataRefUtils::getHeader<TPCSectorHeader*>(inputref);
    if (sectorHeader == nullptr) {
      LOG(ERROR) << "TPC: Sector header missing, skipping processing";
      return;
    }
    auto sector = sectorHeader->sector;
    LOG(INFO) << "TPC: Processing sector " << sector;
    // the active sectors need to be propagated
    uint64_t activeSectors = 0;
    activeSectors = sectorHeader->activeSectors;

    // lambda that snapshots digits to be sent out; prepares and attaches header with sector information
    auto snapshotDigits = [this, sector, &pc, activeSectors, &dh](std::vector<o2::tpc::Digit> const& digits) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      // note that snapshoting only works with non-const references (to be fixed?)
      pc.outputs().snapshot(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(dh->subSpecification), Lifetime::Timeframe,
                                   header},
                            const_cast<std::vector<o2::tpc::Digit>&>(digits));
    };
    // lambda that snapshots the common mode vector to be sent out; prepares and attaches header with sector information
    auto snapshotCommonMode = [this, sector, &pc, activeSectors, &dh](std::vector<o2::tpc::CommonMode> const& commonMode) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      // note that snapshoting only works with non-const references (to be fixed?)
      pc.outputs().snapshot(Output{"TPC", "COMMONMODE", static_cast<SubSpecificationType>(dh->subSpecification), Lifetime::Timeframe,
                                   header},
                            const_cast<std::vector<o2::tpc::CommonMode>&>(commonMode));
    };
    // lambda that snapshots labels to be sent out; prepares and attaches header with sector information
    auto snapshotLabels = [this, &sector, &pc, activeSectors, &dh](o2::dataformats::MCTruthContainer<o2::MCCompLabel> const& labels) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      pc.outputs().snapshot(Output{"TPC", "DIGITSMCTR", static_cast<SubSpecificationType>(dh->subSpecification),
                                   Lifetime::Timeframe, header},
                            const_cast<o2::dataformats::MCTruthContainer<o2::MCCompLabel>&>(labels));
    };
    // lambda that snapshots digits grouping (triggers) to be sent out; prepares and attaches header with sector information
    auto snapshotEvents = [this, sector, &pc, activeSectors, &dh](const std::vector<DigiGroupRef>& events) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      LOG(INFO) << "TPC: Send TRIGGERS for sector " << sector << " channel " << dh->subSpecification << " | size " << events.size();
      pc.outputs().snapshot(Output{"TPC", "DIGTRIGGERS", static_cast<SubSpecificationType>(dh->subSpecification), Lifetime::Timeframe,
                                   header},
                            const_cast<std::vector<DigiGroupRef>&>(events));
    };

    std::vector<o2::tpc::Digit> digitsAccum;                       // accumulator for digits
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelAccum; // timeframe accumulator for labels
    std::vector<CommonMode> commonModeAccum;
    std::vector<DigiGroupRef> eventAccum;

    // no more tasks can be marked with a negative sector
    if (sector < 0) {
      digitsAccum.clear();
      labelAccum.clear();
      commonModeAccum.clear();
      std::vector<DigiGroupRef> evAccDummy;

      snapshotEvents(evAccDummy);
      snapshotDigits(digitsAccum);
      snapshotCommonMode(commonModeAccum);
      snapshotLabels(labelAccum);

      return;
    }

    mDigitizer.setSector(sector);
    mDigitizer.init();

    auto& eventParts = context->getEventParts();

    auto flushDigitsAndLabels = [this, &digitsAccum, &labelAccum, &commonModeAccum](bool finalFlush = false) {
      // flush previous buffer
      mDigits.clear();
      mLabels.clear();
      mCommonMode.clear();
      mDigitizer.flush(mDigits, mLabels, mCommonMode, finalFlush);
      LOG(INFO) << "TPC: Flushed " << mDigits.size() << " digits, " << mLabels.getNElements() << " labels and " << mCommonMode.size() << " common mode entries";
      std::copy(mDigits.begin(), mDigits.end(), std::back_inserter(digitsAccum));
      labelAccum.mergeAtBack(mLabels);
      std::copy(mCommonMode.begin(), mCommonMode.end(), std::back_inserter(commonModeAccum));
    };

    static SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
    mDigitizer.setStartTime(sampaProcessing.getTimeBinFromTime(irecords[0].timeNS / 1000.f));

    TStopwatch timer;
    timer.Start();

    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < irecords.size(); ++collID) {
      const float eventTime = irecords[collID].timeNS / 1000.f;
      LOG(INFO) << "TPC: Event time " << eventTime << " us";
      mDigitizer.setEventTime(eventTime);
      if (!isContinuous) {
        mDigitizer.setStartTime(sampaProcessing.getTimeBinFromTime(eventTime));
      }
      int startSize = digitsAccum.size();

      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        const int eventID = part.entryID;
        const int sourceID = part.sourceID;

        // get the hits for this event and this source
        std::vector<o2::tpc::HitGroup> hitsLeft;
        std::vector<o2::tpc::HitGroup> hitsRight;
        context->retrieveHits(mSimChains, getBranchNameLeft(sector).c_str(), part.sourceID, part.entryID, &hitsLeft);
        context->retrieveHits(mSimChains, getBranchNameRight(sector).c_str(), part.sourceID, part.entryID, &hitsRight);
        LOG(DEBUG) << "TPC: Found " << hitsLeft.size() << " hit groups left and " << hitsRight.size() << " hit groups right in collision " << collID << " eventID " << part.entryID;

        mDigitizer.process(hitsLeft, eventID, sourceID);
        mDigitizer.process(hitsRight, eventID, sourceID);

        flushDigitsAndLabels();

        if (!isContinuous) {
          eventAccum.emplace_back(startSize, digitsAccum.size() - startSize);
        }
      }
    }

    // final flushing step; getting everything not yet written out
    if (isContinuous) {
      LOG(INFO) << "TPC: Final flush";
      flushDigitsAndLabels(true);
      eventAccum.emplace_back(0, digitsAccum.size()); // all digits are grouped to 1 super-event pseudo-triggered mode
    }

    // send out to next stage
    snapshotEvents(eventAccum);
    snapshotDigits(digitsAccum);
    snapshotCommonMode(commonModeAccum);
    snapshotLabels(labelAccum);

    timer.Stop();
    LOG(INFO) << "TPC: Digitization took " << timer.CpuTime() << "s";
  }

 private:
  o2::tpc::Digitizer mDigitizer;
  std::vector<TChain*> mSimChains;
  std::vector<o2::tpc::Digit> mDigits;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mLabels;
  std::vector<o2::tpc::CommonMode> mCommonMode;
  bool mWriteGRP = false;
};

o2::framework::DataProcessorSpec getTPCDigitizerSpec(int channel, bool writeGRP)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::stringstream id;
  id << "TPCDigitizer" << channel;

  std::vector<OutputSpec> outputs; // define channel by triple of (origin, type id of data to be sent on this channel, subspecification)
  outputs.emplace_back("TPC", "DIGITS", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  outputs.emplace_back("TPC", "DIGTRIGGERS", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  outputs.emplace_back("TPC", "DIGITSMCTR", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  outputs.emplace_back("TPC", "COMMONMODE", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  if (writeGRP) {
    outputs.emplace_back("TPC", "ROMode", 0, Lifetime::Timeframe);
    LOG(DEBUG) << "TPC: Channel " << channel << " will supply ROMode";
  }

  return DataProcessorSpec{
    id.str().c_str(),
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<TPCDPLDigitizerTask>()},
    Options{{"distortionType", VariantType::Int, 0, {"Distortion type to be used. 0 = no distortions (default), 1 = realistic distortions (not implemented yet), 2 = constant distortions"}},
            {"gridSize", VariantType::String, "129,144,129", {"Comma separated list of number of bins in (r,phi,z) for distortion lookup tables (r and z can only be 2**N + 1, N=1,2,3,...)"}},
            {"initialSpaceChargeDensity", VariantType::String, "", {"Path to root file containing TH3 with initial space-charge density and name of the TH3 (comma separated)"}},
            {"readSpaceCharge", VariantType::String, "", {"Path to root file containing pre-calculated space-charge object and name of the object (comma separated)"}},
            {"TPCtriggered", VariantType::Bool, false, {"Impose triggered RO mode (default: continuous)"}}}};
}

o2::framework::WorkflowSpec getTPCDigitizerSpec(int nLanes, std::vector<int> const& sectors)
{
  // channel parameter is deprecated in the TPCDigitizer processor, all descendants
  // are initialized not to publish GRP mode, but the channel will be added to the first
  // processor after the pipelines have been created. The processor will decide upon
  // the index in the ParallelContext whether to publish
  WorkflowSpec pipelineTemplate{getTPCDigitizerSpec(0, false)};
  // override the predefined name, index will be added by parallelPipeline method
  pipelineTemplate[0].name = "TPCDigitizer";
  WorkflowSpec pipelines = parallelPipeline(
    pipelineTemplate, nLanes, [size = sectors.size()]() { return size; }, [&sectors](size_t index) { return sectors[index]; });
  // add the channel for the GRP information to the first processor
  pipelines[0].outputs.emplace_back("TPC", "ROMode", 0, Lifetime::Timeframe);
  return pipelines;
}

} // end namespace tpc
} // end namespace o2
