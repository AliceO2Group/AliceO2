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

#include "Generators/GeneratorFromFile.h"
#include "Generators/GeneratorFromO2KineParam.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include <fairlogger/Logger.h>
#include <FairPrimaryGenerator.h>
#include <TBranch.h>
#include <TClonesArray.h>
#include <TFile.h>
#include <TMCProcess.h>
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
    LOG(fatal) << "EventFile " << name << " not found \n";
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
    if (object != nullptr) {
      mEventsAvailable++;
    }
  } while (object != nullptr);
  LOG(info) << "Found " << mEventsAvailable << " events in this file \n";
}

void GeneratorFromFile::SetStartEvent(int start)
{
  if (start < mEventsAvailable) {
    mEventCounter = start;
  } else {
    LOG(error) << "start event bigger than available events\n";
  }
}

bool GeneratorFromFile::rejectOrFixKinematics(TParticle& p)
{
  const auto nominalmass = p.GetMass();
  auto mom2 = p.Px() * p.Px() + p.Py() * p.Py() + p.Pz() * p.Pz();
  auto calculatedmass = p.Energy() * p.Energy() - mom2;
  calculatedmass = (calculatedmass >= 0.) ? std::sqrt(calculatedmass) : -std::sqrt(-calculatedmass);
  const double tol = 1.E-4;
  auto difference = std::abs(nominalmass - calculatedmass);
  if (std::abs(nominalmass - calculatedmass) > tol) {
    const auto asgmass = p.GetCalcMass();
    bool fix = mFixOffShell && std::abs(nominalmass - asgmass) < tol;
    LOG(warn) << "Particle " << p.GetPdgCode() << " has off-shell mass: M_PDG= " << nominalmass << " (assigned= " << asgmass
              << ") calculated= " << calculatedmass << " -> diff= " << difference << " | " << (fix ? "fixing" : "skipping");
    if (fix) {
      double e = std::sqrt(nominalmass * nominalmass + mom2);
      p.SetMomentum(p.Px(), p.Py(), p.Pz(), e);
      p.SetCalcMass(nominalmass);
    } else {
      return false;
    }
  }
  return true;
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
    LOG(info) << "Reading " << branch->GetEntries() << " particles from Kinematics file";

    // read the whole kinematics initially
    std::vector<TParticle> particles;
    for (int i = 0; i < branch->GetEntries(); ++i) {
      branch->GetEntry(i);
      particles.push_back(*particle);
    }

    // filter the particles from Kinematics.root originally put by a generator
    // and which are trackable
    auto isFirstTrackableDescendant = [](TParticle const& p) {
      const int kTransportBit = BIT(14);
      // The particle should have not set kDone bit and its status should not exceed 1
      if ((p.GetUniqueID() > 0 && p.GetUniqueID() != kPNoProcess) || !p.TestBit(kTransportBit)) {
        return false;
      }
      return true;
    };

    for (int i = 0; i < branch->GetEntries(); ++i) {
      auto& p = particles[i];
      if (!isFirstTrackableDescendant(p)) {
        continue;
      }

      bool wanttracking = true; // RS as far as I understand, if it reached this point, it is trackable
      if (wanttracking || !mSkipNonTrackable) {
        if (!rejectOrFixKinematics(p)) {
          continue;
        }
        auto pdgid = p.GetPdgCode();
        auto px = p.Px();
        auto py = p.Py();
        auto pz = p.Pz();
        auto vx = p.Vx();
        auto vy = p.Vy();
        auto vz = p.Vz();
        auto parent = -1;
        auto e = p.Energy();
        auto tof = p.T();
        auto weight = p.GetWeight();
        LOG(debug) << "Putting primary " << pdgid << " " << p.GetStatusCode() << " " << p.GetUniqueID();
        primGen->AddTrack(pdgid, px, py, pz, vx, vy, vz, parent, wanttracking, e, tof, weight);
        particlecounter++;
      }
    }
    mEventCounter++;

    LOG(info) << "Event generator put " << particlecounter << " on stack";
    return kTRUE;
  } else {
    LOG(error) << "GeneratorFromFile: Ran out of events\n";
  }
  return kFALSE;
}

// based on O2 kinematics

GeneratorFromO2Kine::GeneratorFromO2Kine(const char* name)
{
  mEventFile = TFile::Open(name);
  if (mEventFile == nullptr) {
    LOG(fatal) << "EventFile " << name << " not found";
    return;
  }
  // the kinematics will be stored inside a branch MCTrack
  // different events are stored inside different entries
  auto tree = (TTree*)mEventFile->Get("o2sim");
  if (tree) {
    mEventBranch = tree->GetBranch("MCTrack");
    if (mEventBranch) {
      mEventsAvailable = mEventBranch->GetEntries();
      LOG(info) << "Found " << mEventsAvailable << " events in this file";
      return;
    }
  }
  LOG(error) << "Problem reading events from file " << name;
}

bool GeneratorFromO2Kine::Init()
{

  // read and set params
  auto& param = GeneratorFromO2KineParam::Instance();
  LOG(info) << "Init \'FromO2Kine\' generator with following parameters";
  LOG(info) << param;
  mSkipNonTrackable = param.skipNonTrackable;
  mContinueMode = param.continueMode;

  return true;
}

void GeneratorFromO2Kine::SetStartEvent(int start)
{
  if (start < mEventsAvailable) {
    mEventCounter = start;
  } else {
    LOG(error) << "start event bigger than available events\n";
  }
}

bool GeneratorFromO2Kine::importParticles()
{
  // NOTE: This should be usable with kinematics files without secondaries
  // It might need some adjustment to make it work with secondaries or to continue
  // from a kinematics snapshot

  if (mEventCounter < mEventsAvailable) {
    int particlecounter = 0;

    std::vector<o2::MCTrack>* tracks = nullptr;
    mEventBranch->SetAddress(&tracks);
    mEventBranch->GetEntry(mEventCounter);

    for (auto& t : *tracks) {

      // in case we do not want to continue, take only primaries
      if (!mContinueMode && !t.isPrimary()) {
        continue;
      }

      auto pdg = t.GetPdgCode();
      auto px = t.Px();
      auto py = t.Py();
      auto pz = t.Pz();
      auto vx = t.Vx();
      auto vy = t.Vy();
      auto vz = t.Vz();
      auto m1 = t.getMotherTrackId();
      auto m2 = t.getSecondMotherTrackId();
      auto d1 = t.getFirstDaughterTrackId();
      auto d2 = t.getLastDaughterTrackId();
      auto e = t.GetEnergy();
      auto vt = t.T();
      auto weight = t.getWeight();
      auto wanttracking = t.getToBeDone();

      if (mContinueMode) { // in case we want to continue, do only inhibited tracks
        wanttracking &= t.getInhibited();
      }

      LOG(debug) << "Putting primary " << pdg;

      mParticles.push_back(TParticle(pdg, t.getStatusCode().fullEncoding, m1, m2, d1, d2, px, py, pz, e, vx, vy, vz, vt));
      mParticles.back().SetUniqueID((unsigned int)t.getProcess()); // we should propagate the process ID
      mParticles.back().SetBit(ParticleStatus::kToBeDone, wanttracking);
      mParticles.back().SetWeight(weight);

      particlecounter++;
    }
    mEventCounter++;

    if (tracks) {
      delete tracks;
    }

    LOG(info) << "Event generator put " << particlecounter << " on stack";
    return true;
  } else {
    LOG(error) << "GeneratorFromO2Kine: Ran out of events\n";
  }
  return false;
}

void GeneratorFromO2Kine::updateHeader(o2::dataformats::MCEventHeader* eventHeader)
{
  /** update header **/

  // put information about input file and event number of the current event

  eventHeader->putInfo<std::string>("generator", "generatorFromO2Kine");
  eventHeader->putInfo<std::string>("inputFile", mEventFile->GetName());
  eventHeader->putInfo<int>("inputEventNumber", mEventCounter - 1);
}

} // namespace eventgen
} // end namespace o2

ClassImp(o2::eventgen::GeneratorFromFile);
ClassImp(o2::eventgen::GeneratorFromO2Kine);
