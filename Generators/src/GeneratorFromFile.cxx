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
    LOG(ERROR) << "EventFile " << name << " not found \n";
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
  std::cout << "Found " << mEventsAvailable << " events in this file \n";
}

void GeneratorFromFile::SetStartEvent(int start)
{
  if (start < mEventsAvailable) {
    mEventCounter = start;
  } else {
    std::cerr << "start event bigger than available events\n";
  }
}

Bool_t GeneratorFromFile::ReadEvent(FairPrimaryGenerator* primGen)
{
  if (mEventCounter < mEventsAvailable) {
    // get the tree and the branch
    std::stringstream treestringstr;
    treestringstr << "Event" << mEventCounter << "/TreeK";
    TTree* tree = (TTree*)mEventFile->Get(treestringstr.str().c_str());
    if (tree == nullptr)
      return kFALSE;

    auto branch = tree->GetBranch("Particles");
    TParticle* primary = new TParticle();
    branch->SetAddress(&primary);
    for (int i = 0; i < branch->GetEntries(); ++i) {
      branch->GetEntry(i); // fill primary
      auto pdgid = primary->GetPdgCode();
      auto px = primary->Px();
      auto py = primary->Py();
      auto pz = primary->Pz();
      auto vx = primary->Vx();
      auto vy = primary->Vy();
      auto vz = primary->Vz();

      auto parent = -1;
      bool wanttracking = true;
      auto e = primary->Energy();
      auto tof = primary->T();
      auto weight = primary->GetWeight();
      primGen->AddTrack(pdgid, px, py, pz, vx, vy, vz, parent, wanttracking, e, tof, weight);
    }
    mEventCounter++;
    return kTRUE;
  } else {
    LOG(ERROR) << "GeneratorFromFile: Ran out of events\n";
  }
  return kFALSE;
}

} // end namespace
} // end namespace o2

ClassImp(o2::eventgen::GeneratorFromFile)
