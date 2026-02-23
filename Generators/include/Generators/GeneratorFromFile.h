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

/// \author S. Wenzel - Mai 2017

#ifndef ALICEO2_GENERATORFROMFILE_H_
#define ALICEO2_GENERATORFROMFILE_H_

#include "FairGenerator.h"
#include "Generators/Generator.h"
#include "Generators/GeneratorFromO2KineParam.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include <TRandom3.h>
#include <random>

class TBranch;
class TFile;
class TParticle;
class TGrid;

namespace o2
{
namespace eventgen
{
/// This class implements a generic FairGenerator which
/// reads the particles from an external file
/// at the moment, this only supports reading from an AliRoot kinematics file
/// TODO: generalize this to be able to read from files of various formats
/// (idea: use Reader policies or classes)
class GeneratorFromFile : public FairGenerator
{
 public:
  GeneratorFromFile() = default;
  GeneratorFromFile(const char* name);

  // the FairGenerator interface methods

  /** Generates (or reads) one event and adds the tracks to the
   ** injected primary generator instance.
   ** @param primGen  pointer to the primary FairPrimaryGenerator
   **/
  bool ReadEvent(FairPrimaryGenerator* primGen) override;

  // Set from which event to start
  void SetStartEvent(int start);

  void SetSkipNonTrackable(bool b) { mSkipNonTrackable = b; }
  void setFixOffShell(bool b) { mFixOffShell = b; }
  bool rejectOrFixKinematics(TParticle& p);

 private:
  TFile* mEventFile = nullptr; //! the file containing the persistent events
  int mEventCounter = 0;
  int mEventsAvailable = 0;
  bool mSkipNonTrackable = true; //! whether to pass non-trackable (decayed particles) to the MC stack
  bool mFixOffShell = true;      // fix particles with M_assigned != M_calculated
  ClassDefOverride(GeneratorFromFile, 1);
};

/// This class implements a generic FairGenerator which
/// reads the particles from an external O2 sim kinematics file.
class GeneratorFromO2Kine : public o2::eventgen::Generator
{
 public:
  GeneratorFromO2Kine() = default;
  GeneratorFromO2Kine(const char* name);
  GeneratorFromO2Kine(O2KineGenConfig const& pars);

  bool Init() override;

  // the o2 Generator interface methods
  bool generateEvent() override
  { /* trivial - actual work in importParticles */
    return true;
  }
  bool importParticles() override;

  // Set from which event to start
  void SetStartEvent(int start);

  void setContinueMode(bool val) { mContinueMode = val; };
  /** methods that can be overridden **/
  void updateHeader(o2::dataformats::MCEventHeader* eventHeader) override;

 private:
  TFile* mEventFile = nullptr;     //! the file containing the persistent events
  TBranch* mEventBranch = nullptr; //! the branch containing the persistent events
  TBranch* mMCHeaderBranch = nullptr; //! branch containing MC event headers
  int mEventCounter = 0;
  int mEventsAvailable = 0;
  bool mSkipNonTrackable = true; //! whether to pass non-trackable (decayed particles) to the MC stack
  bool mContinueMode = false;    //! whether we want to continue simulation of previously inhibited tracks
  bool mRoundRobin = false;      //! whether we want to take events from file in a round robin fashion
  bool mRandomize = false;       //! whether we want to randomize the order of events in the input file
  unsigned int mRngSeed = 0;     //! randomizer seed, 0 for random value
  bool mRandomPhi = false;       //! whether we want to randomize the phi angle of the particles
  TGrid* mAlienInstance = nullptr; // a cached connection to TGrid (needed for Alien locations)
  std::unique_ptr<O2KineGenConfig> mConfig; //! Configuration object

  std::unique_ptr<o2::dataformats::MCEventHeader> mOrigMCEventHeader; //! the MC event header of the original file

  ClassDefOverride(GeneratorFromO2Kine, 2);
};

/// Special generator for event pools.
/// What do we like to have:
/// - ability to give a file which contains the list of files to read
/// - ability to give directly a file to read the event from
/// - ability to give a pool path and to find the top N list of files closest to myself
/// - ability to select itself one file from the pool
class GeneratorFromEventPool : public o2::eventgen::Generator
{
 public:
  constexpr static std::string_view eventpool_filename = "evtpool.root";
  constexpr static std::string_view alien_protocol_prefix = "alien://";

  GeneratorFromEventPool() = default; // mainly for ROOT IO
  GeneratorFromEventPool(EventPoolGenConfig const& pars);

  bool Init() override;

  // the o2 Generator interface methods
  bool generateEvent() override
  { /* trivial - actual work in importParticles */
    return mO2KineGenerator->generateEvent();
  }
  bool importParticles() override
  {
    mO2KineGenerator->clearParticles(); // clear old container before filling with new ones
    auto import_good = mO2KineGenerator->importParticles();
    // transfer the particles (could be avoided)
    mParticles = mO2KineGenerator->getParticles();

    return import_good;
  }

  void updateHeader(o2::dataformats::MCEventHeader* eventHeader) override
  {
    mO2KineGenerator->updateHeader(eventHeader);
  }

  // determine the collection of available files
  std::vector<std::string> setupFileUniverse(std::string const& path) const;

  std::vector<std::string> const& getFileUniverse() const { return mPoolFilesAvailable; }

 private:
  EventPoolGenConfig mConfig;                                                    //! Configuration object
  std::unique_ptr<o2::eventgen::GeneratorFromO2Kine> mO2KineGenerator = nullptr; //! actual generator doing the work
  std::vector<std::string> mPoolFilesAvailable;                                  //! container keeping the collection of files in the event pool
  std::string mFileChosen;                                                       //! the file chosen for the pool
  // random number generator to determine a concrete file name
  std::mt19937 mRandomEngine; //!

  ClassDefOverride(GeneratorFromEventPool, 1);
};

} // end namespace eventgen
} // end namespace o2

#endif
