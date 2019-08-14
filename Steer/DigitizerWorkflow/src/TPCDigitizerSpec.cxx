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
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for RunContext
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
#include "DetectorsBase/GeometryManager.h"
#include "CommonDataFormat/RangeReference.h"
#include "TPCSimulation/SAMPAProcessing.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
using DigiGroupRef = o2::dataformats::RangeReference<int, int>;

// helper function which will be offered as a service
template <typename T>
void retrieveHits(std::vector<TChain*> const& chains,
                  const char* brname,
                  int sourceID,
                  int entryID,
                  std::vector<T>* hits)
{
  auto br = chains[sourceID]->GetBranch(brname);
  if (!br) {
    LOG(ERROR) << "TPC: No branch found";
    return;
  }
  br->SetAddress(&hits);
  br->GetEntry(entryID);
}

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

class TPCDPLDigitizerTask
{
 public:
  TPCDPLDigitizerTask(int channel, bool writeGRP)
  {
    mChannel = channel;
    mWriteGRP = writeGRP;
  };

  void init(framework::InitContext& ic)
  {
    LOG(INFO) << "Initializing TPC digitization";

    auto useDistortions = ic.options().get<int>("distortionType");
    auto gridSizeString = ic.options().get<std::string>("gridSize");
    auto triggeredMode = ic.options().get<bool>("TPCtriggered");

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
    if (useDistortions > 0) {
      o2::tpc::SpaceCharge::SCDistortionType distortionType = useDistortions == 1 ? o2::tpc::SpaceCharge::SCDistortionType::SCDistortionsRealistic : o2::tpc::SpaceCharge::SCDistortionType::SCDistortionsConstant;
      std::unique_ptr<TH3> hisSCDensity;
      if (TString(inputHisto[0].data()).EndsWith(".root") && inputHisto[1].size() != 0) {
        auto fileSCInput = std::unique_ptr<TFile>(TFile::Open(inputHisto[0].data()));
        hisSCDensity.reset((TH3*)fileSCInput->Get(inputHisto[1].data()));
        hisSCDensity->SetDirectory(nullptr);
      }
      if (distortionType == SpaceCharge::SCDistortionType::SCDistortionsConstant) {
        LOG(INFO) << "TPC: Using constant space-charge distortions.";
        if (hisSCDensity == nullptr) {
          LOG(FATAL) << "Constant space-charge distortions require an initial space-charge density histogram. Please provide the path to the root file (O2TPCSCDensityHisFilePath) and the histogram name (O2TPCSCDensityHisName) in your environment variables.";
        }
      }
      if (distortionType == SpaceCharge::SCDistortionType::SCDistortionsRealistic) {
        LOG(INFO) << "TPC: Using realistic space-charge distortions.";
      }
      if (hisSCDensity) {
        LOG(INFO) << "TPC: Providing initial space-charge density histogram: " << hisSCDensity->GetName();
      }

      mDigitizer.enableSCDistortions(distortionType, hisSCDensity.get(), gridSize[0], gridSize[1], gridSize[2]);
    }
    mDigitizer.setContinuousReadout(!triggeredMode);

    // setup the input chain for the hits
    mSimChains.emplace_back(new TChain("o2sim"));

    // add the main (background) file
    mSimChains.back()->AddFile(ic.options().get<std::string>("simFile").c_str());

    // maybe add a particular signal file
    auto signalfilename = ic.options().get<std::string>("simFileS");
    if (signalfilename.size() > 0) {
      mSimChains.emplace_back(new TChain("o2sim"));
      mSimChains.back()->AddFile(signalfilename.c_str());
    }

    if (!gGeoManager) {
      o2::base::GeometryManager::loadGeometry();
    }
  }

  void run(framework::ProcessingContext& pc)
  {
    static int callcounter = 0;
    callcounter++;

    static bool finished = false;
    if (finished) {
      return;
    }
    LOG(INFO) << "Processing TPC digitization";

    /// For the time being use the defaults for the CDB
    auto& cdb = o2::tpc::CDBInterface::instance();
    cdb.setUseDefaults();
    if (!gSystem->AccessPathName("GainMap.root")) {
      LOG(INFO) << "TPC: Using gain map from 'GainMap.root'";
      cdb.setGainMapFromFile("GainMap.root");
    }

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::RunContext*>("collisioncontext");
    auto& irecords = context->getEventRecords();
    LOG(INFO) << "TPC: Processing " << irecords.size() << " collisions";
    if (irecords.size() == 0) {
      return;
    }

    bool isContinuous = mDigitizer.isContinuousReadout();
    if (mWriteGRP) {
      auto roMode = isContinuous ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
      LOG(INFO) << "TPC: Sending ROMode= " << (mDigitizer.isContinuousReadout() ? "Continuous" : "Triggered")
                << " to GRPUpdater from channel " << mChannel;
      pc.outputs().snapshot(Output{"TPC", "ROMode", 0, Lifetime::Timeframe}, roMode);
      mWriteGRP = false;
    }

    // extract which sector to treat
    auto const* sectorHeader = DataRefUtils::getHeader<TPCSectorHeader*>(pc.inputs().get("collisioncontext"));
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
    auto snapshotDigits = [this, sector, &pc, activeSectors](std::vector<o2::tpc::Digit> const& digits) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      // note that snapshoting only works with non-const references (to be fixed?)
      pc.outputs().snapshot(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(mChannel), Lifetime::Timeframe,
                                   header},
                            const_cast<std::vector<o2::tpc::Digit>&>(digits));
    };
    // lambda that snapshots labels to be sent out; prepares and attaches header with sector information
    auto snapshotLabels = [this, &sector, &pc, activeSectors](o2::dataformats::MCTruthContainer<o2::MCCompLabel> const& labels) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      pc.outputs().snapshot(Output{"TPC", "DIGITSMCTR", static_cast<SubSpecificationType>(mChannel),
                                   Lifetime::Timeframe, header},
                            const_cast<o2::dataformats::MCTruthContainer<o2::MCCompLabel>&>(labels));
    };
    // lambda that snapshots digits grouping (triggers) to be sent out; prepares and attaches header with sector information
    auto snapshotEvents = [this, sector, &pc, activeSectors](const std::vector<DigiGroupRef>& events) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      LOG(INFO) << "TPC: Send TRIGGERS for sector " << sector << " channel " << mChannel << " | size " << events.size();
      pc.outputs().snapshot(Output{"TPC", "DIGTRIGGERS", static_cast<SubSpecificationType>(mChannel), Lifetime::Timeframe,
                                   header},
                            const_cast<std::vector<DigiGroupRef>&>(events));
    };

    std::vector<o2::tpc::Digit> digitsAccum;                       // accumulator for digits
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelAccum; // timeframe accumulator for labels
    std::vector<DigiGroupRef> eventAccum;

    // no more tasks can be marked with a negative sector
    if (sector < 0) {
      digitsAccum.clear();
      labelAccum.clear();
      std::vector<DigiGroupRef> evAccDummy;

      snapshotEvents(evAccDummy);
      snapshotDigits(digitsAccum);
      snapshotLabels(labelAccum);

      if (sector == -1) {
        LOG(INFO) << "TPC: Processing done - exit through the gift shop...";
        pc.services().get<ControlService>().readyToQuit(false);
        finished = true;
      }
      return;
    }

    mDigitizer.setSector(sector);
    mDigitizer.init();

    auto& eventParts = context->getEventParts();

    auto flushDigitsAndLabels = [this, &digitsAccum, &labelAccum](bool finalFlush = false) {
      // flush previous buffer
      mDigits.clear();
      mLabels.clear();
      mDigitizer.flush(mDigits, mLabels, finalFlush);
      LOG(INFO) << "TPC: Flushed " << mDigits.size() << " digits and " << mLabels.getNElements() << " labels";
      std::copy(mDigits.begin(), mDigits.end(), std::back_inserter(digitsAccum));
      labelAccum.mergeAtBack(mLabels);
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
        retrieveHits(mSimChains, getBranchNameLeft(sector).c_str(), part.sourceID, part.entryID, &hitsLeft);
        retrieveHits(mSimChains, getBranchNameRight(sector).c_str(), part.sourceID, part.entryID, &hitsRight);
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
    snapshotLabels(labelAccum);

    timer.Stop();
    LOG(INFO) << "TPC: Digitization took " << timer.CpuTime() << "s";
  }

 private:
  o2::tpc::Digitizer mDigitizer;
  std::vector<TChain*> mSimChains;
  std::vector<o2::tpc::Digit> mDigits;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mLabels;
  bool mWriteGRP;
  int mChannel;
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
  if (writeGRP) {
    outputs.emplace_back("TPC", "ROMode", 0, Lifetime::Timeframe);
    LOG(INFO) << "TPC: Channel " << channel << " will supply ROMode";
  }

  return DataProcessorSpec{
    id.str().c_str(),
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<TPCDPLDigitizerTask>(channel, writeGRP)},
    Options{{"simFile", VariantType::String, "o2sim.root", {"Sim (background) input filename"}},
            {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}},
            {"distortionType", VariantType::Int, 0, {"Distortion type to be used. 0 = no distortions (default), 1 = realistic distortions (not implemented yet), 2 = constant distortions"}},
            {"gridSize", VariantType::String, "33,180,33", {"Comma separated list of number of bins in z, phi and r for distortion lookup tables (z and r can only be 2**N + 1, N=1,2,3,...)"}},
            {"initialSpaceChargeDensity", VariantType::String, "", {"Path to root file containing TH3 with initial space-charge density and name of the TH3 (comma separated)"}},
            {"TPCtriggered", VariantType::Bool, false, {"Impose triggered RO mode (default: continuous)"}}}};
}

} // end namespace tpc
} // end namespace o2
