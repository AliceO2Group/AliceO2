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

/// \author R+Preghenella - August 2017

#include "Generators/Generator.h"
#include "Generators/Trigger.h"
#include "Generators/PrimaryGenerator.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/ParticleStatus.h"
#include "SimulationDataFormat/MCGenProperties.h"
#include <SimConfig/SimConfig.h>
#include "FairPrimaryGenerator.h"
#include <fairlogger/Logger.h>
#include <cmath>
#include "TClonesArray.h"
#include "TParticle.h"
#include "TString.h"
#include "TSystem.h"
#include "TGrid.h"
#include "CCDB/BasicCCDBManager.h"
#include <filesystem>
#ifdef GENERATORS_WITH_TPCLOOPERS
#include "Generators/TPCLoopers.h"
#include "Generators/TPCLoopersParam.h"
#endif

namespace o2
{
namespace eventgen
{

std::atomic<int> Generator::InstanceCounter{0};
unsigned int Generator::gTotalNEvents = 0;
/*****************************************************************/
/*****************************************************************/

Generator::Generator() : FairGenerator("ALICEo2", "ALICEo2 Generator"),
                         mBoost(0.)
{
  /** default constructor **/
  mThisInstanceID = Generator::InstanceCounter;
  Generator::InstanceCounter++;
#ifdef GENERATORS_WITH_TPCLOOPERS
  const auto& simConfig = o2::conf::SimConfig::Instance();
  const auto& loopersParam = o2::eventgen::GenTPCLoopersParam::Instance();
  if (!loopersParam.loopersVeto) {
    bool transport = (simConfig.getMCEngine() != "O2TrivialMCEngine");
    if (transport) {
      bool tpcActive = (std::find(simConfig.getReadoutDetectors().begin(), simConfig.getReadoutDetectors().end(), "TPC") != simConfig.getReadoutDetectors().end());
      if (tpcActive) {
        if (initTPCLoopersGen()) {
          mAddTPCLoopers = kTRUE;
        }
      } else {
        LOG(info) << "TPC not active in readout detectors: loopers fast generator disabled.";
      }
    }
  } else {
    LOG(info) << "Loopers fast generator turned OFF with veto flag.";
  }
#endif
}

/*****************************************************************/

Generator::Generator(const Char_t* name, const Char_t* title) : FairGenerator(name, title),
                                                                mBoost(0.)
{
  /** constructor **/
  mThisInstanceID = Generator::InstanceCounter;
  Generator::InstanceCounter++;
#ifdef GENERATORS_WITH_TPCLOOPERS
  const auto& simConfig = o2::conf::SimConfig::Instance();
  const auto& loopersParam = o2::eventgen::GenTPCLoopersParam::Instance();
  if (!loopersParam.loopersVeto) {
    bool transport = (simConfig.getMCEngine() != "O2TrivialMCEngine");
    if (transport) {
      bool tpcActive = (std::find(simConfig.getReadoutDetectors().begin(), simConfig.getReadoutDetectors().end(), "TPC") != simConfig.getReadoutDetectors().end());
      if (tpcActive) {
        if (initTPCLoopersGen()) {
          mAddTPCLoopers = kTRUE;
        }
      } else {
        LOG(info) << "TPC not active in readout detectors: loopers fast generator disabled.";
      }
    }
  } else {
    LOG(info) << "Loopers fast generator turned OFF with veto flag.";
  }
#endif
}

/*****************************************************************/

Generator::~Generator()
{
  /** destructor **/
#ifdef GENERATORS_WITH_TPCLOOPERS
  if (mTPCLoopersGen) {
    delete mTPCLoopersGen;
    mTPCLoopersGen = nullptr;
  }
#endif
}

/*****************************************************************/
#ifdef GENERATORS_WITH_TPCLOOPERS
bool Generator::initTPCLoopersGen()
{
  // Expand all environment paths
  const auto& loopersParam = o2::eventgen::GenTPCLoopersParam::Instance();
  auto expandPathName = [](const std::string& path) {
    TString expandedPath = path;
    gSystem->ExpandPathName(expandedPath);
    return std::string(expandedPath.Data());
  };
  std::string model_pairs = expandPathName(loopersParam.model_pairs);
  std::string model_compton = expandPathName(loopersParam.model_compton);
  std::string nclxrate = expandPathName(loopersParam.nclxrate);
  const std::string scaler_pair = expandPathName(loopersParam.scaler_pair);
  const std::string scaler_compton = expandPathName(loopersParam.scaler_compton);
  const std::string poisson = expandPathName(loopersParam.poisson);
  const std::string gauss = expandPathName(loopersParam.gauss);
  const auto& flat_gas = loopersParam.flat_gas;
  const auto& colsys = loopersParam.colsys;
  if (flat_gas) {
    if (colsys != "PbPb" && colsys != "pp") {
      LOG(warning) << "Automatic background loopers configuration supports only 'pp' and 'PbPb' systems.";
      LOG(warning) << "Fast loopers generator will remain OFF.";
      return kFALSE;
    }
    bool isContext = std::filesystem::exists("collisioncontext.root");
    if (!isContext) {
      LOG(warning) << "Warning: No collisioncontext.root file found!";
      LOG(warning) << "Loopers will be kept OFF.";
      return kFALSE;
    }
  }
  std::array<float, 2> multiplier = {loopersParam.multiplier[0], loopersParam.multiplier[1]};
  unsigned int nLoopersPairs = loopersParam.fixedNLoopers[0];
  unsigned int nLoopersCompton = loopersParam.fixedNLoopers[1];
  const std::array<std::string, 3> models = {model_pairs, model_compton, nclxrate};
  const std::array<std::string, 3> local_names = {"WGANpair.onnx", "WGANcompton.onnx", "nclxrate.root"};
  const std::array<bool, 3> isAlien = {models[0].starts_with("alien://"), models[1].starts_with("alien://"), models[2].starts_with("alien://")};
  const std::array<bool, 3> isCCDB = {models[0].starts_with("ccdb://"), models[1].starts_with("ccdb://"), models[2].starts_with("ccdb://")};
  if (std::any_of(isAlien.begin(), isAlien.end(), [](bool v) { return v; })) {
    if (!gGrid) {
      TGrid::Connect("alien://");
      if (!gGrid) {
        LOG(fatal) << "AliEn connection failed, check token.";
        exit(1);
      }
    }
    for (size_t i = 0; i < models.size(); ++i) {
      if (isAlien[i] && !TFile::Cp(models[i].c_str(), local_names[i].c_str())) {
        LOG(fatal) << "Error: Model file " << models[i] << " does not exist!";
        exit(1);
      }
    }
  }
  if (std::any_of(isCCDB.begin(), isCCDB.end(), [](bool v) { return v; })) {
    auto& ccdb = o2::ccdb::BasicCCDBManager::instance();
    ccdb.setURL("http://alice-ccdb.cern.ch");
    // Get underlying CCDB API from BasicCCDBManager
    auto& ccdb_api = ccdb.getCCDBAccessor();
    for (size_t i = 0; i < models.size(); ++i) {
      if (isCCDB[i]) {
        auto model_path = models[i].substr(7); // Remove "ccdb://"
        // Treat filename if provided in the CCDB path
        auto extension = model_path.find(".onnx");
        if (extension != std::string::npos) {
          auto last_slash = model_path.find_last_of('/');
          model_path = model_path.substr(0, last_slash);
        }
        std::map<std::string, std::string> filter;
        if (!ccdb_api.retrieveBlob(model_path, "./", filter, o2::ccdb::getCurrentTimestamp(), false, local_names[i].c_str())) {
          LOG(fatal) << "Error: issues in retrieving " << model_path << " from CCDB!";
          exit(1);
        }
      }
    }
  }
  model_pairs = isAlien[0] || isCCDB[0] ? local_names[0] : model_pairs;
  model_compton = isAlien[1] || isCCDB[1] ? local_names[1] : model_compton;
  nclxrate = isAlien[2] || isCCDB[2] ? local_names[2] : nclxrate;
  try {
    // Create the TPC loopers generator with the provided parameters
    mTPCLoopersGen = new o2::eventgen::GenTPCLoopers(model_pairs, model_compton, poisson, gauss, scaler_pair, scaler_compton);
    const auto& intrate = loopersParam.intrate;
    // Configure the generator with flat gas loopers defined per orbit with clusters/track info
    // If intrate is negative (default), automatic IR from collisioncontext.root will be used
    if (flat_gas) {
      mTPCLoopersGen->SetRate(nclxrate, (colsys == "PbPb") ? true : false, intrate);
      mTPCLoopersGen->SetAdjust(loopersParam.adjust_flatgas);
    } else {
      // Otherwise, Poisson+Gauss sampling or fixed number of loopers per event will be used
      // Multiplier is applied only with distribution sampling
      // This configuration can be used for testing purposes, in all other cases flat gas is recommended
      mTPCLoopersGen->SetNLoopers(nLoopersPairs, nLoopersCompton);
      mTPCLoopersGen->SetMultiplier(multiplier);
    }
    LOG(info) << "TPC Loopers generator initialized successfully";
  } catch (const std::exception& e) {
    LOG(error) << "Failed to initialize TPC Loopers generator: " << e.what();
    delete mTPCLoopersGen;
    mTPCLoopersGen = nullptr;
    return kFALSE;
  }
  return kTRUE;
}
#endif

/*****************************************************************/

Bool_t
  Generator::Init()
{
  /** init **/

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t
  Generator::finalizeEvent()
{
#ifdef GENERATORS_WITH_TPCLOOPERS
  if (mAddTPCLoopers) {
    if (!mTPCLoopersGen) {
      LOG(error) << "Loopers generator not initialized";
      return kFALSE;
    }

    // Generate loopers using the initialized TPC loopers generator
    if (!mTPCLoopersGen->generateEvent()) {
      LOG(error) << "Failed to generate loopers event";
      return kFALSE;
    }
    if (mTPCLoopersGen->getNLoopers() == 0) {
      LOG(warning) << "No loopers generated for this event";
      return kTRUE;
    }
    const auto& looperParticles = mTPCLoopersGen->importParticles();
    if (looperParticles.empty()) {
      LOG(error) << "Failed to import loopers particles";
      return kFALSE;
    }
    // Append the generated looper particles to the main particle list
    mParticles.insert(mParticles.end(), looperParticles.begin(), looperParticles.end());

    LOG(debug) << "Added " << looperParticles.size() << " looper particles";
  }
#endif
  return kTRUE;
}

/*****************************************************************/

Bool_t
  Generator::ReadEvent(FairPrimaryGenerator* primGen)
{
  /** read event **/

  /** endless generate-and-trigger loop **/
  while (true) {
    mReadEventCounter++;

    /** clear particle vector **/
    mParticles.clear();

    /** reset the sub-generator ID **/
    mSubGeneratorId = -1;

    /** generate event **/
    if (!generateEvent()) {
      LOG(error) << "ReadEvent failed in generateEvent";
      return kFALSE;
    }

    /** import particles **/
    if (!importParticles()) {
      LOG(error) << "ReadEvent failed in importParticles";
      return kFALSE;
    }

    /** Event finalization**/
    if (!finalizeEvent()) {
      LOG(error) << "ReadEvent failed in finalizeEvent";
      return kFALSE;
    }

    if (mSubGeneratorsIdToDesc.empty() && mSubGeneratorId > -1) {
      LOG(fatal) << "ReadEvent failed because no SubGenerator description given";
    }

    if (!mSubGeneratorsIdToDesc.empty() && mSubGeneratorId < 0) {
      LOG(fatal) << "ReadEvent failed because SubGenerator description given but sub-generator not set";
    }

    /** trigger event **/
    if (triggerEvent()) {
      mTriggerOkHook(mParticles, mReadEventCounter);
      break;
    } else {
      mTriggerFalseHook(mParticles, mReadEventCounter);
    }
  }

  /** add tracks **/
  if (!addTracks(primGen)) {
    LOG(error) << "ReadEvent failed in addTracks";
    return kFALSE;
  }

  /** update header **/
  auto header = primGen->GetEvent();
  auto o2header = dynamic_cast<o2::dataformats::MCEventHeader*>(header);
  if (!header) {
    LOG(fatal) << "MC event header is not a 'o2::dataformats::MCEventHeader' object";
    return kFALSE;
  }
  updateHeader(o2header);
  updateSubGeneratorInformation(o2header);

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t
  Generator::addTracks(FairPrimaryGenerator* primGen)
{
  /** add tracks **/

  auto o2primGen = dynamic_cast<PrimaryGenerator*>(primGen);
  if (!o2primGen) {
    LOG(fatal) << "PrimaryGenerator is not a o2::eventgen::PrimaryGenerator";
    return kFALSE;
  }

  /** loop over particles **/
  for (const auto& particle : mParticles) {
    o2primGen->AddTrack(particle.GetPdgCode(),
                        particle.Px() * mMomentumUnit,
                        particle.Py() * mMomentumUnit,
                        particle.Pz() * mMomentumUnit,
                        particle.Vx() * mPositionUnit,
                        particle.Vy() * mPositionUnit,
                        particle.Vz() * mPositionUnit,
                        particle.GetMother(0),
                        particle.GetMother(1),
                        particle.GetDaughter(0),
                        particle.GetDaughter(1),
                        particle.TestBit(ParticleStatus::kToBeDone),
                        particle.Energy() * mEnergyUnit,
                        particle.T() * mTimeUnit,
                        particle.GetWeight(),
                        (TMCProcess)particle.GetUniqueID(),
                        particle.GetStatusCode()); // generator status information passed as status code field
  }

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t
  Generator::boostEvent()
{
  /** boost event **/

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t
  Generator::triggerEvent()
{
  /** trigger event **/

  /** check trigger presence **/
  if (mTriggers.size() == 0 && mDeepTriggers.size() == 0) {
    return kTRUE;
  }

  /** check trigger mode **/
  Bool_t triggered;
  if (mTriggerMode == kTriggerOFF) {
    return kTRUE;
  } else if (mTriggerMode == kTriggerOR) {
    triggered = kFALSE;
  } else if (mTriggerMode == kTriggerAND) {
    triggered = kTRUE;
  } else {
    return kTRUE;
  }

  /** loop over triggers **/
  for (const auto& trigger : mTriggers) {
    auto retval = trigger(mParticles);
    if (mTriggerMode == kTriggerOR) {
      triggered |= retval;
    }
    if (mTriggerMode == kTriggerAND) {
      triggered &= retval;
    }
  }

  /** loop over deep triggers **/
  for (const auto& trigger : mDeepTriggers) {
    auto retval = trigger(mInterface, mInterfaceName);
    if (mTriggerMode == kTriggerOR) {
      triggered |= retval;
    }
    if (mTriggerMode == kTriggerAND) {
      triggered &= retval;
    }
  }

  /** return **/
  return triggered;
}

/*****************************************************************/

void Generator::addSubGenerator(int subGeneratorId, std::string const& subGeneratorDescription)
{
  if (subGeneratorId < 0) {
    LOG(fatal) << "Sub-generator IDs must be >= 0, instead, passed value is " << subGeneratorId;
  }
  mSubGeneratorsIdToDesc.insert({subGeneratorId, subGeneratorDescription});
}

/*****************************************************************/

void Generator::updateSubGeneratorInformation(o2::dataformats::MCEventHeader* header) const
{
  if (mSubGeneratorId < 0) {
    return;
  }
  header->putInfo<int>(o2::mcgenid::GeneratorProperty::SUBGENERATORID, mSubGeneratorId);
  header->putInfo<std::unordered_map<int, std::string>>(o2::mcgenid::GeneratorProperty::SUBGENERATORDESCRIPTIONMAP, mSubGeneratorsIdToDesc);
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

ClassImp(o2::eventgen::Generator);
