// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author S. Wenzel - Mai 2017

#ifndef ALICEO2_GENERATORFROMFILE_H_
#define ALICEO2_GENERATORFROMFILE_H_

#include "FairGenerator.h"

class TBranch;
class TFile;

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

 private:
  TFile* mEventFile = nullptr; //! the file containing the persistent events
  int mEventCounter = 0;
  int mEventsAvailable = 0;
  bool mSkipNonTrackable = true; //! whether to pass non-trackable (decayed particles) to the MC stack

  ClassDefOverride(GeneratorFromFile, 1);
};

} // end namespace eventgen
} // end namespace o2

#endif
