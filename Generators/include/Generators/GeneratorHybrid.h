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

/// \author M. Giacalone - October 2024

#ifndef ALICEO2_EVENTGEN_GENERATORHYBRID_H_
#define ALICEO2_EVENTGEN_GENERATORHYBRID_H_

#include "Generators/Generator.h"
#include "Generators/BoxGenerator.h"
#include <Generators/GeneratorPythia8.h>
#include <Generators/GeneratorHepMC.h>
#include <Generators/GeneratorFromFile.h>
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCGenProperties.h"
#include "SimulationDataFormat/ParticleStatus.h"
#include "Generators/GeneratorHybridParam.h"
#include "Generators/GeneratorHepMCParam.h"
#include "Generators/GeneratorPythia8Param.h"
#include "Generators/GeneratorFileOrCmdParam.h"
#include "Generators/GeneratorFromO2KineParam.h"
#include "Generators/GeneratorExternalParam.h"
#include <TRandom3.h>
#include "CommonUtils/ConfigurationMacroHelper.h"
#include "FairGenerator.h"
#include <DetectorsBase/Stack.h>
#include <SimConfig/SimConfig.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/writer.h>
#include "TBufferJSON.h"

#include <tbb/concurrent_queue.h>
#include <tbb/task_arena.h>
#include <iostream>
#include <thread>
#include <atomic>

namespace o2
{
namespace eventgen
{

class GeneratorHybrid : public Generator
{

 public:
  GeneratorHybrid& operator=(const GeneratorHybrid&) = delete;
  GeneratorHybrid(const GeneratorHybrid&) = delete;

  // Singleton access method
  static GeneratorHybrid& Instance(const std::string& inputgens = "");

  Bool_t Init() override;
  Bool_t generateEvent() override;
  Bool_t importParticles() override;
  void updateHeader(o2::dataformats::MCEventHeader* eventHeader) override;

  Bool_t parseJSON(const std::string& path);
  Bool_t confSetter(const auto& gen);
  template <typename T>
  std::string jsonValueToString(const T& value);
  std::vector<std::shared_ptr<o2::eventgen::Generator>> const& getGenerators() { return gens; }

 private:
  GeneratorHybrid(const std::string& inputgens);
  ~GeneratorHybrid();
  o2::eventgen::Generator* currentgen = nullptr;
  std::vector<std::shared_ptr<o2::eventgen::Generator>> gens;
  const std::vector<std::string> generatorNames = {"evtpool", "boxgen", "external", "hepmc", "pythia8", "pythia8pp", "pythia8hi", "pythia8hf", "pythia8powheg"};
  std::vector<std::string> mInputGens;
  std::vector<std::string> mGens;
  std::vector<std::string> mConfigs;
  std::vector<std::string> mConfsPythia8;

  std::vector<bool> mGenIsInitialized;

  // Parameters configurations
  std::vector<std::unique_ptr<o2::eventgen::BoxGenConfig>> mBoxGenConfigs;
  std::vector<std::unique_ptr<o2::eventgen::Pythia8GenConfig>> mPythia8GenConfigs;
  std::vector<std::unique_ptr<o2::eventgen::O2KineGenConfig>> mO2KineGenConfigs;
  std::vector<o2::eventgen::EventPoolGenConfig> mEventPoolConfigs;
  std::vector<std::unique_ptr<o2::eventgen::ExternalGenConfig>> mExternalGenConfigs;
  std::vector<std::unique_ptr<o2::eventgen::FileOrCmdGenConfig>> mFileOrCmdGenConfigs;
  std::vector<std::unique_ptr<o2::eventgen::HepMCGenConfig>> mHepMCGenConfigs;

  bool mRandomize = false;
  std::vector<int> mFractions;
  std::vector<float> mRngFractions;
  int mseqCounter = 0;
  int mCurrentFraction = 0;
  int mIndex = 0;
  int mEventCounter = 0;
  int mTasksStarted = 0;

  // Cocktail mode
  bool mCocktailMode = false;
  std::vector<std::vector<int>> mGroups;

  // Trigger configuration
  std::vector<ETriggerMode_t> mTriggerModes;            // trigger mode for each generator
  std::vector<std::vector<std::string>> mTriggerMacros; // trigger macros for each generator (multiple triggers for each generator possible)
  std::vector<std::vector<std::string>> mTriggerFuncs;  // trigger functions for each generator (multiple triggers for each generator possible)

  // Create a task arena with a specified number of threads
  std::thread mTBBTaskPoolRunner;
  tbb::concurrent_bounded_queue<int> mInputTaskQueue;
  std::vector<tbb::concurrent_bounded_queue<int>> mResultQueue;
  tbb::task_arena mTaskArena;
  std::atomic<bool> mStopFlag;
  bool mIsInitialized = false;

  o2::dataformats::MCEventHeader mMCEventHeader; // to capture event headers
  int mHeaderGeneratorIndex = -1;                // index of the generator that updated the header in current event

  enum class GenMode {
    kSeq,
    kParallel
  };

  // hybrid gen operation mode - should be either 'sequential' or 'parallel'
  // parallel means that we have clones of the same generator collaborating on event generation
  // sequential means that events will be produced in the order given by fractions; async processing is still happening
  GenMode mGenerationMode = GenMode::kSeq; //!
};

} // namespace eventgen
} // namespace o2

#endif
