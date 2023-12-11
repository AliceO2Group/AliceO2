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
/// reads the particles from an external O2 sim kinematics file.
class GeneratorFromO2Kine : public o2::eventgen::Generator
{
 public:
  GeneratorFromO2Kine() = default;
  GeneratorFromO2Kine(const char* name);

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

 private:
  /** methods that can be overridden **/
  void updateHeader(o2::dataformats::MCEventHeader* eventHeader) override;

  TFile* mEventFile = nullptr;     //! the file containing the persistent events
  TBranch* mEventBranch = nullptr; //! the branch containing the persistent events
  int mEventCounter = 0;
  int mEventsAvailable = 0;
  bool mSkipNonTrackable = true; //! whether to pass non-trackable (decayed particles) to the MC stack
  bool mContinueMode = false;    //! whether we want to continue simulation of previously inhibited tracks
  bool mRoundRobin = false;      //! whether we want to take events from file in a round robin fashion
  ClassDefOverride(GeneratorFromO2Kine, 2);
};

} // end namespace eventgen
} // end namespace o2

#endif
