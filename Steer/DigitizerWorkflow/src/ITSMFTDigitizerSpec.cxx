// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITSMFTDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "Steer/HitProcessingManager.h" // for RunContext
#include "ITSMFTBase/Digit.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTSimulation/Digitizer.h"
#include "ITSMFTSimulation/DPLDigitizerParam.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "ITSBase/GeometryTGeo.h"
#include "MFTBase/GeometryTGeo.h"
#include <TGeoManager.h>
#include <TChain.h>
#include <TStopwatch.h>
#include <string>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace itsmft
{

class ITSMFTDPLDigitizerTask
{

 public:
  void init(framework::InitContext& ic)
  {
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
    // init optional QED chain
    auto qedfilename = ic.options().get<std::string>("simFileQED");
    if (qedfilename.size() > 0) {
      mQEDChain.AddFile(qedfilename.c_str());
      LOG(INFO) << "Attach QED Tree: " << mQEDChain.GetEntries();
    }

    setDigitizationOptions(); // set options provided via configKeyValues mechanism
    auto& digipar = mDigitizer.getParams();

    mROMode = digipar.isContinuous() ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
    LOG(INFO) << mID.getName() << " simulated in "
              << ((mROMode == o2::parameters::GRPObject::CONTINUOUS) ? "CONTINUOUS" : "TRIGGERED")
              << " RO mode";

    // make sure that the geometry is loaded (TODO will this be done centrally?)
    if (!gGeoManager) {
      o2::base::GeometryManager::loadGeometry();
    }

    // configure digitizer
    o2::itsmft::GeometryTGeo* geom = nullptr;
    if (mID == o2::detectors::DetID::ITS) {
      geom = o2::its::GeometryTGeo::Instance();
    } else {
      geom = o2::mft::GeometryTGeo::Instance();
    }
    geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::L2G)); // make sure L2G matrices are loaded
    mDigitizer.setGeometry(geom);

    // init digitizer
    mDigitizer.init();
  }

  virtual void setDigitizationOptions() = 0;

  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }
    std::string detStr = mID.getName();
    // read collision context from input
    auto context = pc.inputs().get<o2::steer::RunContext*>("collisioncontext");
    auto& timesview = context->getEventRecords();
    LOG(DEBUG) << "GOT " << timesview.size() << " COLLISSION TIMES";
    // if there is nothing to do ... return
    if (timesview.size() == 0) {
      return;
    }
    TStopwatch timer;
    timer.Start();
    LOG(INFO) << " CALLING ITS DIGITIZATION ";

    mDigitizer.setDigits(&mDigits);
    mDigitizer.setROFRecords(&mROFRecords);
    mDigitizer.setMCLabels(&mLabels);

    // attach optional QED digits branch
    setupQEDChain();

    auto& eventParts = context->getEventParts();
    // loop over all composite collisions given from context (aka loop over all the interaction records)
    for (int collID = 0; collID < timesview.size(); ++collID) {
      auto eventTime = timesview[collID].timeNS;

      if (mQEDChain.GetEntries()) { // QED must be processed before other inputs since done in small time steps
        processQED(eventTime);
      }

      mDigitizer.setEventTime(eventTime);
      mDigitizer.resetEventROFrames(); // to estimate min/max ROF for this collID
      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {

        // get the hits for this event and this source
        mHits.clear();
        retrieveHits(part.sourceID, part.entryID);

        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID
                  << " found " << mHits.size() << " hits ";

        mDigitizer.process(&mHits, part.entryID, part.sourceID); // call actual digitization procedure
      }
      mMC2ROFRecordsAccum.emplace_back(collID, -1, mDigitizer.getEventROFrameMin(), mDigitizer.getEventROFrameMax());
      accumulate();
    }
    // finish digitization ... stream any remaining digits/labels
    if (mQEDChain.GetEntries()) { // fill last slots from QED input
      processQED(mDigitizer.getEndTimeOfROFMax());
    }

    mDigitizer.fillOutputContainer();
    accumulate();

    // here we have all digits and labels and we can send them to consumer (aka snapshot it onto output)
    pc.outputs().snapshot(Output{mOrigin, "DIGITS", 0, Lifetime::Timeframe}, mDigitsAccum);
    pc.outputs().snapshot(Output{mOrigin, "DIGITSROF", 0, Lifetime::Timeframe}, mROFRecordsAccum);
    pc.outputs().snapshot(Output{mOrigin, "DIGITSMC2ROF", 0, Lifetime::Timeframe}, mMC2ROFRecordsAccum);
    pc.outputs().snapshot(Output{mOrigin, "DIGITSMCTR", 0, Lifetime::Timeframe}, mLabelsAccum);
    LOG(INFO) << mID.getName() << ": Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{mOrigin, "ROMode", 0, Lifetime::Timeframe}, mROMode);

    timer.Stop();
    LOG(INFO) << "Digitization took " << timer.CpuTime() << "s";

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(false);

    mFinished = true;
  }

 protected:
  ITSMFTDPLDigitizerTask() = default;

  void processQED(double tMax)
  {
    auto tQEDNext = mLastQEDTimeNS + mQEDEntryTimeBinNS; // timeslice to retrieve
    std::string detStr = mID.getName();
    auto br = mQEDChain.GetBranch((detStr + "Hit").c_str());
    while (tQEDNext < tMax) {
      mLastQEDTimeNS = tQEDNext;      // time used for current QED slot
      tQEDNext += mQEDEntryTimeBinNS; // prepare time for next QED slot
      if (++mLastQEDEntry >= mQEDChain.GetEntries()) {
        mLastQEDEntry = 0; // wrapp if needed
      }
      br->GetEntry(mLastQEDEntry);
      mDigitizer.setEventTime(mLastQEDTimeNS);
      mDigitizer.process(&mHits, mLastQEDEntry, QEDSourceID);
      //
    }
  }

  void setupQEDChain()
  {
    if (!mQEDChain.GetEntries()) {
      return;
    }
    std::string detStr = mID.getName();
    auto qedBranch = mQEDChain.GetBranch((detStr + "Hit").c_str());
    assert(qedBranch != nullptr);
    assert(mQEDEntryTimeBinNS >= 1.0);
    assert(QEDSourceID < o2::MCCompLabel::maxSourceID());
    mLastQEDTimeNS = -mQEDEntryTimeBinNS / 2; // time will be assigned to the middle of the bin
    qedBranch->SetAddress(&mHitsP);
    LOG(INFO) << "Attaching QED ITS hits as sourceID=" << int(QEDSourceID) << ", entry integrates "
              << mQEDEntryTimeBinNS << " ns";
  }

  // helper function which will be offered as a service
  void retrieveHits(int sourceID, int entryID)
  {
    std::string detStr = mID.getName();
    auto br = mSimChains[sourceID]->GetBranch((detStr + "Hit").c_str());
    if (!br) {
      LOG(ERROR) << "No branch " << (detStr + "Hit").c_str() << " found for sourceID=" << sourceID;
      return;
    }
    br->SetAddress(&mHitsP);
    br->GetEntry(entryID);
  }

  void accumulate()
  {
    // accumulate result of single event processing, called after processing every event supplied
    // AND after the final flushing via digitizer::fillOutputContainer
    if (!mDigits.size()) {
      return; // no digits were flushed, nothing to accumulate
    }
    static int fixMC2ROF = 0; // 1st entry in mc2rofRecordsAccum to be fixed for ROFRecordID
    auto ndigAcc = mDigitsAccum.size();
    std::copy(mDigits.begin(), mDigits.end(), std::back_inserter(mDigitsAccum));

    // fix ROFrecords references on ROF entries
    auto nROFRecsOld = mROFRecordsAccum.size();

    for (int i = 0; i < mROFRecords.size(); i++) {
      auto& rof = mROFRecords[i];
      rof.getROFEntry().shiftIndex(ndigAcc);
      rof.print();

      if (mFixMC2ROF < mMC2ROFRecordsAccum.size()) { // fix ROFRecord entry in MC2ROF records
        for (int m2rid = mFixMC2ROF; m2rid < mMC2ROFRecordsAccum.size(); m2rid++) {
          // need to register the ROFRecors entry for MC event starting from this entry
          auto& mc2rof = mMC2ROFRecordsAccum[m2rid];
          if (rof.getROFrame() == mc2rof.minROF) {
            mFixMC2ROF++;
            mc2rof.rofRecordID = nROFRecsOld + i;
            mc2rof.print();
          }
        }
      }
    }
    std::copy(mROFRecords.begin(), mROFRecords.end(), std::back_inserter(mROFRecordsAccum));
    mLabelsAccum.mergeAtBack(mLabels);
    LOG(INFO) << "Added " << mDigits.size() << " digits ";
    // clean containers from already accumulated stuff
    mLabels.clear();
    mDigits.clear();
    mROFRecords.clear();
  }

  bool mFinished = false;
  o2::detectors::DetID mID;
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginInvalid;
  o2::itsmft::Digitizer mDigitizer;
  std::vector<o2::itsmft::Digit> mDigits;
  std::vector<o2::itsmft::Digit> mDigitsAccum;
  std::vector<o2::itsmft::ROFRecord> mROFRecords;
  std::vector<o2::itsmft::ROFRecord> mROFRecordsAccum;
  std::vector<o2::itsmft::Hit> mHits;
  std::vector<o2::itsmft::Hit>* mHitsP = &mHits;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mLabels;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mLabelsAccum;
  std::vector<o2::itsmft::MC2ROFRecord> mMC2ROFRecordsAccum;
  std::vector<TChain*> mSimChains;
  TChain mQEDChain = {"o2sim"};

  double mQEDEntryTimeBinNS = 1000;                                               // time-coverage of single QED tree entry in ns (TODO: make it settable)
  double mLastQEDTimeNS = 0;                                                      // time assingned to last QED entry
  int mLastQEDEntry = -1;                                                         // last used QED entry
  int mFixMC2ROF = 0;                                                             // 1st entry in mc2rofRecordsAccum to be fixed for ROFRecordID
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::PRESENT; // readout mode

  const int QEDSourceID = 99; // unique source ID for the QED (TODO: move it as a const to general class?)
};

//_______________________________________________
class ITSDPLDigitizerTask : public ITSMFTDPLDigitizerTask
{
 public:
  // FIXME: origina should be extractable from the DetID, the problem is 3d party header dependencies
  static constexpr o2::detectors::DetID::ID DETID = o2::detectors::DetID::ITS;
  static constexpr o2::header::DataOrigin DETOR = o2::header::gDataOriginITS;
  ITSDPLDigitizerTask()
  {
    mID = DETID;
    mOrigin = DETOR;
  }
  void setDigitizationOptions() override
  {
    auto& dopt = o2::itsmft::DPLDigitizerParam<DETID>::Instance();
    auto& aopt = o2::itsmft::DPLAlpideParam<DETID>::Instance();
    auto& digipar = mDigitizer.getParams();
    digipar.setContinuous(dopt.continuous);
    digipar.setROFrameLength(aopt.roFrameLength); // RO frame in ns
    digipar.setStrobeDelay(aopt.strobeDelay);     // Strobe delay wrt beginning of the RO frame, in ns
    digipar.setStrobeLength(aopt.strobeLength);   // Strobe length in ns
    // parameters of signal time response: flat-top duration, max rise time and q @ which rise time is 0
    digipar.getSignalShape().setParameters(dopt.strobeFlatTop, dopt.strobeMaxRiseTime, dopt.strobeQRiseTime0);
    digipar.setChargeThreshold(dopt.chargeThreshold); // charge threshold in electrons
    digipar.setNoisePerPixel(dopt.noisePerPixel);     // noise level
    digipar.setTimeOffset(dopt.timeOffset);
    digipar.setNSimSteps(dopt.nSimSteps);
  }
};

constexpr o2::detectors::DetID::ID ITSDPLDigitizerTask::DETID;
constexpr o2::header::DataOrigin ITSDPLDigitizerTask::DETOR;

//_______________________________________________
class MFTDPLDigitizerTask : public ITSMFTDPLDigitizerTask
{
 public:
  // FIXME: origina should be extractable from the DetID, the problem is 3d party header dependencies
  static constexpr o2::detectors::DetID::ID DETID = o2::detectors::DetID::MFT;
  static constexpr o2::header::DataOrigin DETOR = o2::header::gDataOriginMFT;
  MFTDPLDigitizerTask()
  {
    mID = DETID;
    mOrigin = DETOR;
  }

  void setDigitizationOptions() override
  {
    auto& dopt = o2::itsmft::DPLDigitizerParam<DETID>::Instance();
    auto& aopt = o2::itsmft::DPLAlpideParam<DETID>::Instance();
    auto& digipar = mDigitizer.getParams();
    digipar.setContinuous(dopt.continuous);
    digipar.setROFrameLength(aopt.roFrameLength); // RO frame in ns
    digipar.setStrobeDelay(aopt.strobeDelay);     // Strobe delay wrt beginning of the RO frame, in ns
    digipar.setStrobeLength(aopt.strobeLength);   // Strobe length in ns
    // parameters of signal time response: flat-top duration, max rise time and q @ which rise time is 0
    digipar.getSignalShape().setParameters(dopt.strobeFlatTop, dopt.strobeMaxRiseTime, dopt.strobeQRiseTime0);
    digipar.setChargeThreshold(dopt.chargeThreshold); // charge threshold in electrons
    digipar.setNoisePerPixel(dopt.noisePerPixel);     // noise level
    digipar.setTimeOffset(dopt.timeOffset);
    digipar.setNSimSteps(dopt.nSimSteps);
  }
};

constexpr o2::detectors::DetID::ID MFTDPLDigitizerTask::DETID;
constexpr o2::header::DataOrigin MFTDPLDigitizerTask::DETOR;

DataProcessorSpec getITSDigitizerSpec(int channel)
{
  std::string detStr = o2::detectors::DetID::getName(ITSDPLDigitizerTask::DETID);
  auto detOrig = ITSDPLDigitizerTask::DETOR;
  std::stringstream parHelper;
  parHelper << "Params as " << o2::itsmft::DPLDigitizerParam<ITSDPLDigitizerTask::DETID>::getParamName().data() << ".<param>=value;... with"
            << o2::itsmft::DPLDigitizerParam<ITSDPLDigitizerTask::DETID>::Instance()
            << "\n or " << o2::itsmft::DPLAlpideParam<ITSDPLDigitizerTask::DETID>::getParamName().data() << ".<param>=value;... with"
            << o2::itsmft::DPLAlpideParam<ITSDPLDigitizerTask::DETID>::Instance();
  return DataProcessorSpec{(detStr + "Digitizer").c_str(),
                           Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT",
                                            static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
                           Outputs{
                             OutputSpec{detOrig, "DIGITS", 0, Lifetime::Timeframe},
                             OutputSpec{detOrig, "DIGITSROF", 0, Lifetime::Timeframe},
                             OutputSpec{detOrig, "DIGITSMC2ROF", 0, Lifetime::Timeframe},
                             OutputSpec{detOrig, "DIGITSMCTR", 0, Lifetime::Timeframe},
                             OutputSpec{detOrig, "ROMode", 0, Lifetime::Timeframe}},
                           AlgorithmSpec{adaptFromTask<ITSDPLDigitizerTask>()},
                           Options{
                             {"simFile", VariantType::String, "o2sim.root", {"Sim (background) input filename"}},
                             {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}},
                             {"simFileQED", VariantType::String, "", {"Sim (QED) input filename"}},
                             //  { "configKeyValues", VariantType::String, "", { parHelper.str().c_str() } }
                           }};
}

DataProcessorSpec getMFTDigitizerSpec(int channel)
{
  std::string detStr = o2::detectors::DetID::getName(MFTDPLDigitizerTask::DETID);
  auto detOrig = MFTDPLDigitizerTask::DETOR;
  std::stringstream parHelper;

  parHelper << "Params as " << o2::itsmft::DPLDigitizerParam<ITSDPLDigitizerTask::DETID>::getParamName().data() << ".<param>=value;... with"
            << o2::itsmft::DPLDigitizerParam<ITSDPLDigitizerTask::DETID>::Instance()
            << " or " << o2::itsmft::DPLAlpideParam<ITSDPLDigitizerTask::DETID>::getParamName().data() << ".<param>=value;... with"
            << o2::itsmft::DPLAlpideParam<ITSDPLDigitizerTask::DETID>::Instance();
  return DataProcessorSpec{(detStr + "Digitizer").c_str(),
                           Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT",
                                            static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
                           Outputs{
                             OutputSpec{detOrig, "DIGITS", 0, Lifetime::Timeframe},
                             OutputSpec{detOrig, "DIGITSROF", 0, Lifetime::Timeframe},
                             OutputSpec{detOrig, "DIGITSMC2ROF", 0, Lifetime::Timeframe},
                             OutputSpec{detOrig, "DIGITSMCTR", 0, Lifetime::Timeframe},
                             OutputSpec{detOrig, "ROMode", 0, Lifetime::Timeframe}},
                           AlgorithmSpec{adaptFromTask<MFTDPLDigitizerTask>()},
                           Options{
                             {"simFile", VariantType::String, "o2sim.root", {"Sim (background) input filename"}},
                             {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}},
                             {"simFileQED", VariantType::String, "", {"Sim (QED) input filename"}}}};
}

} // end namespace itsmft
} // end namespace o2
