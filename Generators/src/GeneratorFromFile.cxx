// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Generators/GeneratorFromFile.h"
#include <FairLogger.h>
#include <FairPrimaryGenerator.h>
#include <TBranch.h>
#include <TClonesArray.h>
#include <TFile.h>
#include <TParticle.h>
#include <TTree.h>
#include <sstream>

namespace o2
{
namespace eventgen
{
GeneratorFromFile::GeneratorFromFile(const char* name)
{
  mEventFile = TFile::Open(name);
  if (mEventFile == nullptr) {
    LOG(FATAL) << "EventFile " << name << " not found \n";
    return;
  }
  // the kinematics will be stored inside a Tree "TreeK" with branch "Particles"
  // different events are stored inside TDirectories

  // we need to probe for the number of events
  TObject* object = nullptr;
  do {
    std::stringstream eventstringstr;
    eventstringstr << "Event" << mEventsAvailable;
    // std::cout << "probing for " << eventstring << "\n";
    object = mEventFile->Get(eventstringstr.str().c_str());
    // std::cout << "got " << object << "\n";
    if (object != nullptr)
      mEventsAvailable++;
  } while (object != nullptr);
  LOG(INFO) << "Found " << mEventsAvailable << " events in this file \n";
}

void GeneratorFromFile::SetStartEvent(int start)
{
  if (start < mEventsAvailable) {
    mEventCounter = start;
  } else {
    LOG(ERROR) << "start event bigger than available events\n";
  }
}

bool isOnMassShell(TParticle const& p)
{
  const auto nominalmass = p.GetMass();
  auto calculatedmass = p.Energy() * p.Energy() - (p.Px() * p.Px() + p.Py() * p.Py() + p.Pz() * p.Pz());
  calculatedmass = (calculatedmass >= 0.) ? std::sqrt(calculatedmass) : -std::sqrt(-calculatedmass);
  const double tol = 1.E-4;
  auto difference = std::abs(nominalmass - calculatedmass);
  LOG(DEBUG) << "ISONMASSSHELL INFO" << difference << " " << nominalmass << " " << calculatedmass;
  return std::abs(nominalmass - calculatedmass) < tol;
}

Bool_t GeneratorFromFile::ReadEvent(FairPrimaryGenerator* primGen)
{
  if (mEventCounter < mEventsAvailable) {
    int particlecounter = 0;

    // get the tree and the branch
    std::stringstream treestringstr;
    treestringstr << "Event" << mEventCounter << "/TreeK";
    TTree* tree = (TTree*)mEventFile->Get(treestringstr.str().c_str());
    if (tree == nullptr) {
      return kFALSE;
    }

    auto branch = tree->GetBranch("Particles");
    TParticle* particle = nullptr;
    branch->SetAddress(&particle);
    LOG(INFO) << "Reading " << branch->GetEntries() << " particles from Kinematics file";

    // read the whole kinematics initially
    std::vector<TParticle> particles;
    for (int i = 0; i < branch->GetEntries(); ++i) {
      branch->GetEntry(i);
      particles.push_back(*particle);
    }

    // filter the particles from Kinematics.root originally put by a generator
    // and which are trackable
    auto isFirstTrackableDescendant = [](TParticle const& p) {
      // according to the current understanding in AliRoot, we
      // have status code:
      // == 0    <--->   particle is put by transportation
      // == 1    <--->   particle is trackable
      // != 1 but different from 0    <--->   particle is not directly trackable
      // Note: This might have to be refined (using other information such as UniqueID)
      if (p.GetStatusCode() == 1) {
        return true;
      }
      return false;
    };

    for (int i = 0; i < branch->GetEntries(); ++i) {
      auto& p = particles[i];

      if (!isFirstTrackableDescendant(p)) {
        continue;
      }

      auto pdgid = p.GetPdgCode();
      auto px = p.Px();
      auto py = p.Py();
      auto pz = p.Pz();
      auto vx = p.Vx();
      auto vy = p.Vy();
      auto vz = p.Vz();

      // a status of 1 means "trackable" in AliRoot kinematics
      auto status = p.GetStatusCode();
      bool wanttracking = status == 1;
      if (wanttracking || !mSkipNonTrackable) {
        auto parent = -1;
        auto e = p.Energy();
        auto tof = p.T();
        auto weight = p.GetWeight();
        if (!isOnMassShell(p)) {
          LOG(WARNING) << "Skipping " << pdgid << " since off-mass shell";
          continue;
        }
        LOG(DEBUG) << "Putting primary " << pdgid << " " << p.GetStatusCode() << " " << p.GetUniqueID();
        primGen->AddTrack(pdgid, px, py, pz, vx, vy, vz, parent, wanttracking, e, tof, weight);
        particlecounter++;
      }
    }
    mEventCounter++;

    LOG(INFO) << "Event generator put " << particlecounter << " on stack";
    return kTRUE;
  } else {
    LOG(ERROR) << "GeneratorFromFile: Ran out of events\n";
  }
  return kFALSE;
}

} // namespace eventgen
} // end namespace o2

ClassImp(o2::eventgen::GeneratorFromFile);
