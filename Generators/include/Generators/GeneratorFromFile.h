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
#include <TGrid.h>
#include <random>
#include <string>
#include <vector>

class TBranch;
class TFile;
class TParticle;

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
/// reads the particles from one or more external O2 sim kinematics files.
class GeneratorFromO2Kine : public o2::eventgen::Generator
{
 public:
  GeneratorFromO2Kine() = default;
  /// name may be a single file or a comma-separated list of files to be read one after the other
  GeneratorFromO2Kine(const char* name);
  GeneratorFromO2Kine(std::vector<std::string> const& filenames);
  GeneratorFromO2Kine(O2KineGenConfig const& pars);
  /// same as above but with an explicit list of files. Used for event pools
  GeneratorFromO2Kine(O2KineGenConfig const& pars, std::vector<std::string> const& filenames);
  ~GeneratorFromO2Kine() override;

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
  const o2::dataformats::MCEventHeader* getOrigMCEventHeader() const { return mOrigMCEventHeader.get(); }

  /// number of events available in the file that is currently open
  int getEventsAvailable() const { return mEventsAvailable; }
  /// number of input files this generator can go through
  int getNumberOfFiles() const { return (int)mFileNames.size(); }
  /// index (within the file list) of the file currently open, -1 if none
  int getCurrentFileIndex() const { return mCurrentFileIndex; }
  /// name of the file currently open, empty if none
  std::string getCurrentFileName() const;
  /// number of opened files so far (including the current one)
  int getNumberOfFilesUsed() const { return mFilesUsed; }
  /// total number of events delivered so far
  int getEventsServed() const { return mEventsServed; }

  /// helper splitting a comma-separated list of file names into its components
  static std::vector<std::string> splitFileNames(std::string const& filenames);

 private:
  /// closes the currently opened file currently
  void closeCurrentFile();
  /// fixes the order in which the events of the current file are served
  void establishEventOrder();
  /// opens the file at the given index of the file list.
  /// Returns false in case the file cannot be used
  bool openFile(int index);
  /// moves on to the next usable file of the list; wraps around in round robin mode.
  /// Returns false when no further file is available
  bool openNextFile(bool wrapAround);

  std::vector<std::string> mFileNames;      //! the list of input files, read one after the other
  int mCurrentFileIndex = -1;               //! index of the file currently open
  int mFilesUsed = 0;                       //! how many files have been opened so far
  TFile* mCurrentFile = nullptr;            //! the file currently open
  TBranch* mEventBranch = nullptr;          //! the branch containing the persistent events
  TBranch* mMCHeaderBranch = nullptr;       //! branch containing MC event headers
  std::vector<int> mEventOrder;             //! order in which the entries of the current file are served
  int mEventCounter = 0;                    //! events already delivered from the current file
  int mEventsServed = 0;                    //! events delivered in total, across all files
  int mEventsAvailable = 0;                 //! events contained in the current file
  int mStartEvent = 0;                      //! event to start from in the very first file
  int mLastEntryRead = -1;                  //! entry of the current event within the current file
  bool mSkipNonTrackable = true;            //! whether to pass non-trackable (decayed particles) to the MC stack
  bool mContinueMode = false;               //! whether we want to continue simulation of previously inhibited tracks
  bool mRoundRobin = false;                 //! whether we want to take events from file in a round robin fashion
  bool mRandomize = false;                  //! whether we want to randomize the order of events in the input file
  unsigned int mRngSeed = 0;                //! randomizer seed, 0 for random value
  bool mRandomPhi = false;                  //! whether we want to randomize the phi angle of the particles
  TGrid* mAlienInstance = nullptr;          // a cached connection to TGrid (needed for Alien locations)
  std::unique_ptr<O2KineGenConfig> mConfig; //! Configuration object

  std::unique_ptr<o2::dataformats::MCEventHeader> mOrigMCEventHeader; //! the MC event header of the original file

  ClassDefOverride(GeneratorFromO2Kine, 3);
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
    auto original_header = mO2KineGenerator->getOrigMCEventHeader();
    // Workaround to fix vertex shifted particles from event pools (valid for builds released before 14 March 2026)
    if (original_header) {
      double vertex[3] = {original_header->GetX(), original_header->GetY(), original_header->GetZ()};
      if (vertex[0] != 0. || vertex[1] != 0. || vertex[2] != 0.) {
        LOG(debug) << "Subtracting shifted vertex from EventPool: (" << vertex[0] << ", " << vertex[1] << ", " << vertex[2] << ")";
        for (auto& p : mParticles) {
          p.SetProductionVertex(p.Vx() - vertex[0], p.Vy() - vertex[1], p.Vz() - vertex[2], p.T());
        }
      }
    }
    return import_good;
  }

  void updateHeader(o2::dataformats::MCEventHeader* eventHeader) override
  {
    // Copy current vertex position from the event header
    const double xyz[3] = {eventHeader->GetX(), eventHeader->GetY(), eventHeader->GetZ()};
    mO2KineGenerator->updateHeader(eventHeader);
    // Event pool uses vertex position from current simulation, only extKinO2 takes the one from the file instead
    eventHeader->SetVertex(xyz[0], xyz[1], xyz[2]);
  }

  // determine the collection of available files
  std::vector<std::string> setupFileUniverse(std::string const& path) const;

  std::vector<std::string> const& getFileUniverse() const { return mPoolFilesAvailable; }

  /// the file universe, in the order this generator instance will go through it
  std::vector<std::string> const& getChosenFiles() const { return mFilesChosen; }

  /// shuffles the given universe of pool files into the order this instance will use
  std::vector<std::string> selectFiles(std::vector<std::string> const& universe);

  /// access to the underlying kinematics generator
  o2::eventgen::GeneratorFromO2Kine const* getO2KineGenerator() const { return mO2KineGenerator.get(); }

 private:
  EventPoolGenConfig mConfig;                                                    //! Configuration object
  std::unique_ptr<o2::eventgen::GeneratorFromO2Kine> mO2KineGenerator = nullptr; //! actual generator doing the work
  std::vector<std::string> mPoolFilesAvailable;                                  //! container keeping the collection of files in the event pool
  std::vector<std::string> mFilesChosen;                                         //! the file(s) chosen from the pool
  // random number generator to determine a concrete file name
  std::mt19937 mRandomEngine; //!

  ClassDefOverride(GeneratorFromEventPool, 2);
};

} // end namespace eventgen
} // end namespace o2

#endif
