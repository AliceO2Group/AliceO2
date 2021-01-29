// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Stack.cxx
/// \brief Implementation of the Stack class
/// \author M. Al-Turany, S. Wenzel - June 2014

#include "SimulationDataFormat/Stack.h"
#include "DetectorsBase/Detector.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimConfig/SimCutParams.h"

#include "FairDetector.h" // for FairDetector
#include "FairLogger.h"   // for FairLogger
#include "FairRootManager.h"
#include "SimulationDataFormat/BaseHits.h"
#include "SimulationDataFormat/StackParam.h"
#include "CommonUtils/ConfigurationMacroHelper.h"

#include "TLorentzVector.h" // for TLorentzVector
#include "TParticle.h"      // for TParticle
#include "TRefArray.h"      // for TRefArray
#include "TVirtualMC.h"     // for VMC
#include "TMCProcess.h"     // for VMC Particle Production Process

#include <algorithm>
#include <cassert>
#include <cstddef> // for NULL
#include <cmath>

using std::cout;
using std::endl;
using std::pair;
using namespace o2::data;

// small helper function to append to vector at arbitrary position
template <typename T, typename I>
void insertInVector(std::vector<T>& v, I index, T e)
{
  auto currentsize = v.size();
  if (index >= currentsize) {
    const auto newsize = std::max(index + 1, (I)(1 + currentsize * 1.2));
    v.resize(newsize, T(-1));
  }
  // new size must at least be as large as index
  // assert(index < v.size());
  v[index] = e;
}

Stack::Stack(Int_t size)
  : FairGenericStack(),
    mStack(),
    mParticles(),
    mTracks(new std::vector<o2::MCTrack>),
    mTrackIDtoParticlesEntry(1000000, -1),
    mIndexMap(),
    mIndexOfCurrentTrack(-1),
    mIndexOfCurrentPrimary(-1),
    mNumberOfPrimaryParticles(0),
    mNumberOfEntriesInParticles(0),
    mNumberOfEntriesInTracks(0),
    mNumberOfPrimariesforTracking(0),
    mNumberOfPrimariesPopped(0),
    mStoreMothers(kTRUE),
    mStoreSecondaries(kTRUE),
    mMinHits(1),
    mEnergyCut(0.),
    mTrackRefs(new std::vector<o2::TrackReference>),
    mIsG4Like(false)
{
  auto vmc = TVirtualMC::GetMC();
  if (vmc) {
    mIsG4Like = !(vmc->SecondariesAreOrdered());
  }

  auto& param = o2::sim::StackParam::Instance();
  LOG(INFO) << param;
  TransportFcn transportPrimary;
  if (param.transportPrimary.compare("none") == 0) {
    transportPrimary = [](const TParticle& p, const std::vector<TParticle>& particles) {
      return false;
    };
  } else if (param.transportPrimary.compare("all") == 0) {
    transportPrimary = [](const TParticle& p, const std::vector<TParticle>& particles) {
      return true;
    };
  } else if (param.transportPrimary.compare("barrel") == 0) {
    transportPrimary = [](const TParticle& p, const std::vector<TParticle>& particles) {
      return (std::fabs(p.Eta()) < 2.0);
    };
  } else if (param.transportPrimary.compare("external") == 0) {
    transportPrimary = o2::conf::GetFromMacro<o2::data::Stack::TransportFcn>(param.transportPrimaryFileName,
                                                                             param.transportPrimaryFuncName,
                                                                             "o2::data::Stack::TransportFcn", "stack_transport_primary");
    if (!mTransportPrimary) {
      LOG(FATAL) << "Failed to retrieve external \'transportPrimary\' function: problem with configuration ";
    }
  } else {
    LOG(FATAL) << "unsupported \'trasportPrimary\' mode: " << param.transportPrimary;
  }

  if (param.transportPrimaryInvert) {
    mTransportPrimary = [transportPrimary](const TParticle& p, const std::vector<TParticle>& particles) { return !transportPrimary; };
  } else {
    mTransportPrimary = transportPrimary;
  }
}

Stack::Stack(const Stack& rhs)
  : FairGenericStack(rhs),
    mStack(),
    mParticles(),
    mTracks(nullptr),
    mIndexMap(),
    mIndexOfCurrentTrack(-1),
    mIndexOfCurrentPrimary(-1),
    mNumberOfPrimaryParticles(0),
    mNumberOfEntriesInParticles(0),
    mNumberOfEntriesInTracks(0),
    mNumberOfPrimariesforTracking(0),
    mNumberOfPrimariesPopped(0),
    mStoreMothers(rhs.mStoreMothers),
    mStoreSecondaries(rhs.mStoreSecondaries),
    mMinHits(rhs.mMinHits),
    mEnergyCut(rhs.mEnergyCut),
    mTrackRefs(new std::vector<o2::TrackReference>),
    mIsG4Like(rhs.mIsG4Like)
{
  LOG(DEBUG) << "copy constructor called";
  mTracks = new std::vector<MCTrack>();
}

Stack::~Stack()
{
  if (mTracks) {
    delete mTracks;
  }
}

Stack& Stack::operator=(const Stack& rhs)
{
  LOG(FATAL) << "operator= called";
  // check assignment to self
  if (this == &rhs) {
    return *this;
  }

  // base class assignment
  FairGenericStack::operator=(rhs);
  mTracks = new std::vector<MCTrack>(rhs.mTracks->size());
  mIndexOfCurrentTrack = -1;
  mIndexOfCurrentPrimary = -1;
  mNumberOfPrimaryParticles = 0;
  mNumberOfEntriesInParticles = 0;
  mNumberOfEntriesInTracks = 0;
  mNumberOfPrimariesforTracking = 0;
  mNumberOfPrimariesPopped = 0;
  mStoreMothers = rhs.mStoreMothers;
  mStoreSecondaries = rhs.mStoreSecondaries;
  mMinHits = rhs.mMinHits;
  mEnergyCut = rhs.mEnergyCut;
  mIsG4Like = rhs.mIsG4Like;

  return *this;
}

void Stack::PushTrack(Int_t toBeDone, Int_t parentId, Int_t pdgCode, Double_t px, Double_t py, Double_t pz, Double_t e,
                      Double_t vx, Double_t vy, Double_t vz, Double_t time, Double_t polx, Double_t poly, Double_t polz,
                      TMCProcess proc, Int_t& ntr, Double_t weight, Int_t is)
{
  PushTrack(toBeDone, parentId, pdgCode, px, py, pz, e, vx, vy, vz, time, polx, poly, polz, proc, ntr, weight, is, -1);
}

void Stack::PushTrack(Int_t toBeDone, Int_t parentId, Int_t pdgCode, Double_t px, Double_t py, Double_t pz, Double_t e,
                      Double_t vx, Double_t vy, Double_t vz, Double_t time, Double_t polx, Double_t poly, Double_t polz,
                      TMCProcess proc, Int_t& ntr, Double_t weight, Int_t is, Int_t secondparentId)
{
  PushTrack(toBeDone, parentId, pdgCode, px, py, pz, e, vx, vy, vz, time, polx, poly, polz, proc, ntr, weight, is, secondparentId, -1, -1);
}

void Stack::PushTrack(Int_t toBeDone, Int_t parentId, Int_t pdgCode, Double_t px, Double_t py, Double_t pz, Double_t e,
                      Double_t vx, Double_t vy, Double_t vz, Double_t time, Double_t polx, Double_t poly, Double_t polz,
                      TMCProcess proc, Int_t& ntr, Double_t weight, Int_t is, Int_t secondparentId, Int_t daughter1Id, Int_t daughter2Id)
{
  //  printf("Pushing %s toBeDone %5d parentId %5d pdgCode %5d is %5d entries %5d \n",
  //	 proc == kPPrimary ? "Primary:   " : "Secondary: ",
  //	 toBeDone, parentId, pdgCode, is, mNumberOfEntriesInParticles);

  //
  // This method is called
  //
  // - during serial   simulation directly from the event generator
  // - during parallel simulation to fill the stack of the primary generator device
  // - in all cases to push a secondary particle
  //
  //
  // Create new TParticle and add it to the TParticle array

  Int_t trackId = mNumberOfEntriesInParticles;
  // Set track variable
  ntr = trackId;
  //  Int_t daughter1Id = -1;
  //  Int_t daughter2Id = -1;
  Int_t iStatus = (proc == kPPrimary) ? is : trackId;
  TParticle p(pdgCode, iStatus, parentId, secondparentId, daughter1Id, daughter2Id, px, py, pz, e, vx, vy, vz, time);
  p.SetPolarisation(polx, poly, polz);
  p.SetWeight(weight);
  p.SetUniqueID(proc); // using the unique ID to transfer process ID
  mNumberOfEntriesInParticles++;

  insertInVector(mTrackIDtoParticlesEntry, trackId, (int)(mParticles.size()));

  // Push particle on the stack if toBeDone is set
  if (proc == kPPrimary) {
    // This is a particle from the primary particle generator
    //
    // SetBit is used to pass information about the primary particle to the stack during transport.
    // Sime particles have already decayed or are partons from a shower. They are needed for the
    // event history in the stack, but not for transport.
    //
    mIndexMap[trackId] = trackId;
    p.SetBit(ParticleStatus::kKeep);
    p.SetBit(ParticleStatus::kPrimary);
    if (toBeDone == 1) {
      handleTransportPrimary(p);
    } else {
      p.SetBit(ParticleStatus::kToBeDone, 0);
    }
    mNumberOfPrimaryParticles++;
    mPrimaryParticles.push_back(p);
    mTracks->emplace_back(p);
  } else {
    p.SetBit(ParticleStatus::kPrimary, 0);
    if (toBeDone == 1) {
      p.SetBit(ParticleStatus::kToBeDone, 1);
    } else {
      p.SetBit(ParticleStatus::kToBeDone, 0);
    }
    mParticles.emplace_back(p);
    mCurrentParticle0 = p;
  }
  mStack.push(p);
}

void Stack::handleTransportPrimary(TParticle& p)
{
  // this function tests whether we really want to transport
  // this particle and sets the relevant bits accordingly

  if (mTransportPrimary(p, mPrimaryParticles)) {
    p.SetBit(ParticleStatus::kToBeDone, 1);
    mNumberOfPrimariesforTracking++;
  } else {
    p.SetBit(ParticleStatus::kToBeDone, 0);
    p.SetBit(ParticleStatus::kInhibited, 1);
  }
}

void Stack::PushTrack(int toBeDone, TParticle& p)
{
  //  printf("stack -> Pushing Primary toBeDone %5d %5d parentId %5d pdgCode %5d is %5d entries %5d \n", toBeDone, p.TestBit(ParticleStatus::kToBeDone), p.GetFirstMother(), p.GetPdgCode(), p.GetStatusCode(),  mNumberOfEntriesInParticles);

  // This method is called
  //
  // - during parallel simulation to push primary particles (called by the stack itself)
  if (p.GetUniqueID() == 0) {
    // one to one mapping for primaries
    mIndexMap[mNumberOfPrimaryParticles] = mNumberOfPrimaryParticles;
    // Push particle on the stack
    if (p.TestBit(ParticleStatus::kPrimary) && p.TestBit(ParticleStatus::kToBeDone)) {
      handleTransportPrimary(p);
    }
    mNumberOfPrimaryParticles++;
    mPrimaryParticles.push_back(p);
    mStack.push(p);
    mTracks->emplace_back(p);
  }
}

/// Set the current track number
/// Declared in TVirtualMCStack
/// \param iTrack track number
void Stack::SetCurrentTrack(Int_t iTrack)
{
  mIndexOfCurrentTrack = iTrack;
  if (iTrack < mPrimaryParticles.size()) {
    auto& p = mPrimaryParticles[iTrack];
    mCurrentParticle = p;
    mIndexOfCurrentPrimary = iTrack;
  } else {
    mCurrentParticle = mCurrentParticle0;
  }
}

// calculates a hash based on particle properties
// hash may serve as seed for this track
ULong_t getHash(TParticle const& p)
{
  auto asLong = [](double x) {
    return (ULong_t) * (reinterpret_cast<ULong_t*>(&x));
  };

  ULong_t hash;
  o2::MCTrackT<double> track(p);

  hash = asLong(track.GetStartVertexCoordinatesX());
  hash ^= asLong(track.GetStartVertexCoordinatesY());
  hash ^= asLong(track.GetStartVertexCoordinatesZ());
  hash ^= asLong(track.GetStartVertexCoordinatesT());
  hash ^= asLong(track.GetStartVertexMomentumX());
  hash ^= asLong(track.GetStartVertexMomentumY());
  hash ^= asLong(track.GetStartVertexMomentumZ());
  hash += (ULong_t)track.GetPdgCode();
  return hash;
}

TParticle* Stack::PopNextTrack(Int_t& iTrack)
{

  // This functions is mainly used by Geant3?
  Int_t nprod = (int)(mParticles.size());

  // If end of stack: Return empty pointer
  if (mStack.empty()) {
    iTrack = -1;
    return nullptr;
  }
  Bool_t found = kFALSE;

  TParticle* nextParticle = nullptr;
  while (!found && !mStack.empty()) {
    // get next particle from stack
    mCurrentParticle = mStack.top();
    // remove particle from the top
    mStack.pop();
    // test if primary to be transported
    if (mCurrentParticle.TestBit(ParticleStatus::kToBeDone)) {
      if (mCurrentParticle.TestBit(ParticleStatus::kPrimary)) {
        // particle is primary and needs to be tracked -> indicates that previous particle finished
        mNumberOfPrimariesPopped++;
        mIndexOfCurrentPrimary = mStack.size();
        mIndexOfCurrentTrack = mIndexOfCurrentPrimary;
      } else {
        mIndexOfCurrentTrack = mCurrentParticle.GetStatusCode();
      }
      iTrack = mIndexOfCurrentTrack;
      if (o2::conf::SimCutParams::Instance().trackSeed) {
        auto hash = getHash(mCurrentParticle);
        // LOG(INFO) << "SEEDING NEW TRACK USING HASH" << hash;
        // init seed per track
        gRandom->SetSeed(hash);
        // NOTE: THE BETTER PLACE WOULD BE IN PRETRACK HOOK BUT THIS DOES NOT SEEM TO WORK
        // WORKS ONLY WITH G3 SINCE G4 DOES NOT CALL THIS FUNCTION
      } // .trackSeed ?
      nextParticle = &mCurrentParticle;
      found = kTRUE;
    } else {
      iTrack = -1;
      nextParticle = nullptr;
    }
  } // while
  return nextParticle;
}

TParticle* Stack::PopPrimaryForTracking(Int_t iPrim)
{
  // This function is used by Geant4 to setup their own internal stack
  //  printf("PopPrimary for tracking %5d %5d \n", iPrim, mNumberOfPrimaryParticles);
  // Remark: Contrary to what the interface name is suggesting
  // this is not a pop operation (but rather a get)

  // Test for index
  if (iPrim < 0 || iPrim >= mNumberOfPrimaryParticles) {
    LOG(FATAL) << "Stack::PopPrimaryForTracking: Stack: Primary index out of range! " << iPrim << " ";
    return nullptr;
  }
  // Return the iPrim-th TParticle from the fParticle array. This should be
  // a primary.
  auto particle = &mPrimaryParticles[iPrim];
  if (particle->TestBit(ParticleStatus::kToBeDone)) {
    return particle;
  } else {
    return nullptr;
  }
}

void Stack::updateEventStats()
{
  if (mMCEventStats) {
    mMCEventStats->setNHits(mHitCounter);
    mMCEventStats->setNTransportedTracks(mNumberOfEntriesInParticles);
    mMCEventStats->setNKeptTracks(mTracks->size());
  }
}

void Stack::FillTrackArray()
{
  /// This interface is not implemented since we are filtering/filling the output array
  /// after each primary ... just give a summary message
  LOG(INFO) << "Stack: " << mTracks->size() << " out of " << mNumberOfEntriesInParticles << " stored \n";
}

void Stack::FinishPrimary()
{
  // Here transport of a primary and all its secondaries is finished
  // we can do some cleanup of the memory structures
  mPrimariesDone++;
  LOG(DEBUG) << "Finish primary hook " << mPrimariesDone;
  // preserve particles and theire ancestors that produced hits

  auto selected = selectTracks();

  // loop over current particle buffer
  // - build index map indicesKept
  // - update mother index information
  int indexOld = 0;
  int indexNew = 0;
  int indexoffset = mTracks->size();
  int neglected = 0;
  std::vector<int> indicesKept((int)(mParticles.size()));
  std::vector<MCTrack> tmpTracks;
  Int_t ic = 0;

  // mTrackIDtoParticlesEntry
  // trackID to mTrack -> index in mParticles
  //
  // indicesKept
  // index in mParticles -> index in mTrack - highWaterMark
  //
  // mReorderIndices
  // old (mTrack-highWaterMark) -> new (mTrack-highWaterMark)
  //

  for (auto& particle : mParticles) {
    if (particle.getStore() || !mPruneKinematics) {
      // map the global track index to the new persistent index
      auto imother = particle.getMotherTrackId();
      // here mother is relative to the mParticles array
      if (imother >= 0) {
        // daughter of a secondary: obtain index from lookup table
        imother = indicesKept[imother];
        particle.SetMotherTrackId(imother);
      }
      // at this point we have the correct mother index in mParticles or
      // a negative one which is a pointer to a primary
      tmpTracks.emplace_back(particle);
      indicesKept[indexOld] = indexNew;
      indexNew++;
    } else {
      indicesKept[indexOld] = -1;
      neglected++;
    }
    indexOld++;
    mTracksDone++;
  }
  Int_t ntr = (int)(tmpTracks.size());
  std::vector<int> reOrderedIndices(ntr);
  std::vector<int> invreOrderedIndices(ntr);
  for (Int_t i = 0; i < ntr; i++) {
    invreOrderedIndices[i] = i;
    reOrderedIndices[i] = i;
  }

  if (mIsG4Like) {
    ReorderKine(tmpTracks, reOrderedIndices);
    for (Int_t i = 0; i < ntr; i++) {
      Int_t index = reOrderedIndices[i];
      invreOrderedIndices[index] = i;
    }
  }
  for (Int_t i = 0; i < ntr; i++) {
    Int_t index = reOrderedIndices[i];
    auto& particle = tmpTracks[index];
    Int_t imo = particle.getMotherTrackId();
    Int_t imo0 = imo;
    if (imo >= 0) {
      imo = invreOrderedIndices[imo];
    }
    imo += indexoffset;
    particle.SetMotherTrackId(imo);
    mTracks->emplace_back(particle);
    auto& mother = mTracks->at(imo);
    if (mother.getFirstDaughterTrackId() == -1) {
      mother.SetFirstDaughterTrackId((int)(mTracks->size()) - 1);
    }
    mother.SetLastDaughterTrackId((int)(mTracks->size()) - 1);
  }

  //
  // Update index map
  //
  Int_t imax = mNumberOfEntriesInParticles;
  Int_t imin = imax - mParticles.size();
  for (Int_t idTrack = imin; idTrack < imax; idTrack++) {
    Int_t index1 = mTrackIDtoParticlesEntry[idTrack];
    Int_t index2 = indicesKept[index1];
    if (index2 == -1) {
      continue;
    }
    Int_t index3 = (mIsG4Like) ? invreOrderedIndices[index2] : index2;
    mIndexMap[idTrack] = index3 + indexoffset;
  }

  // we can now clear the particles buffer!
  reOrderedIndices.clear();
  invreOrderedIndices.clear();
  mParticles.clear();
  tmpTracks.clear();
  mTransportedIDs.clear();
  mTrackIDtoParticlesEntry.clear();
  mIndexOfPrimaries.clear();
}

void Stack::UpdateTrackIndex(TRefArray* detList)
{
  // we can avoid any updating in case no tracks have been filtered out
  // check this like this
  if (mIndexMap.size() == 0) {
    LOG(INFO) << "No TrackIndex update necessary\n";
    return;
  }

  // we are getting the detectorlist from FairRoot as TRefArray
  // at each call, but this list never changes so we cache it here
  // as the right type to avoid repeated dynamic casts
  if (mActiveDetectors.size() == 0) {
    if (detList == nullptr) {
      LOG(FATAL) << "No detList passed to Stack";
    }
    auto iter = detList->MakeIterator();
    while (auto det = iter->Next()) {
      auto o2det = dynamic_cast<o2::base::Detector*>(det);
      if (o2det) {
        mActiveDetectors.emplace_back(o2det);
      } else {
        LOG(INFO) << "Found nonconforming detector " << det->GetName();
      }
    }
  }

  LOG(DEBUG) << "Stack::UpdateTrackIndex: Stack: Updating track indices...";
  Int_t nColl = 0;

  // update track references
  // use some caching since repeated trackIDs
  for (auto& ref : *mTrackRefs) {
    const auto id = ref.getTrackID();
    auto iter = mIndexMap.find(id);
    if (iter == mIndexMap.end()) {
      LOG(INFO) << "Invalid trackref ... needs to be rmoved \n";
      ref.setTrackID(-1);
    } else {
      ref.setTrackID(iter->second);
    }
  }

  // sort trackrefs according to new track index
  // then according to track length
  std::sort(mTrackRefs->begin(), mTrackRefs->end(), [](const o2::TrackReference& a, const o2::TrackReference& b) {
    if (a.getTrackID() == b.getTrackID()) {
      return a.getLength() < b.getLength();
    }
    return a.getTrackID() < b.getTrackID();
  });

  for (auto det : mActiveDetectors) {
    // update the track indices by delegating to specialized detector functions
    det->updateHitTrackIndices(mIndexMap);
  } // List of active detectors

  LOG(DEBUG) << "Stack::UpdateTrackIndex: ...stack and " << nColl << " collections updated.";
}

void Stack::Reset()
{

  mIndexOfCurrentTrack = -1;
  mNumberOfPrimaryParticles = mNumberOfEntriesInParticles = mNumberOfEntriesInTracks = 0;
  while (!mStack.empty()) {
    mStack.pop();
  }
  mParticles.clear();
  mTracks->clear();
  if (!mIsExternalMode && (mPrimariesDone != mNumberOfPrimariesforTracking)) {
    LOG(FATAL) << "Inconsistency in primary particles treated " << mPrimariesDone << " vs expected "
               << mNumberOfPrimariesforTracking << "\n(This points to a flaw in the stack logic)";
  }
  mPrimariesDone = 0;
  mNumberOfPrimariesforTracking = 0;
  mNumberOfPrimariesPopped = 0;
  mPrimaryParticles.clear();
  mTrackRefs->clear();
  mTrackIDtoParticlesEntry.clear();
  mHitCounter = 0;
}

void Stack::Register()
{
  FairRootManager::Instance()->RegisterAny("MCTrack", mTracks, kTRUE);
  FairRootManager::Instance()->RegisterAny("TrackRefs", mTrackRefs, kTRUE);
}

void Stack::Print(Int_t iVerbose) const
{
  cout << "-I- Stack: Number of primaries        = " << mNumberOfPrimaryParticles << endl;
  cout << "              Total number of particles  = " << mNumberOfEntriesInParticles << endl;
  cout << "              Number of tracks in output = " << mNumberOfEntriesInTracks << endl;
  if (iVerbose) {
    for (auto& track : *mTracks) {
      track.Print();
    }
  }
}

void Stack::Print(Option_t* option) const
{
  Int_t verbose = 0;
  if (option) {
    verbose = 1;
  }
  Print(verbose);
}

void Stack::addHit(int iDet)
{
  if (mIndexOfCurrentTrack < mNumberOfPrimaryParticles) {
    auto& part = mTracks->at(mIndexOfCurrentTrack);
    part.setHit(iDet);

  } else {
    Int_t iTrack = mTrackIDtoParticlesEntry[mIndexOfCurrentTrack];
    auto& part = mParticles[iTrack];
    part.setHit(iDet);
  }
  mHitCounter++;
}
void Stack::addHit(int iDet, Int_t iTrack)
{
  mHitCounter++;
  auto& part = mParticles[iTrack];
  part.setHit(iDet);
}

Int_t Stack::GetCurrentParentTrackNumber() const
{
  TParticle* currentPart = GetCurrentTrack();
  if (currentPart) {
    return currentPart->GetFirstMother();
  } else {
    return -1;
  }
}

bool Stack::selectTracks()
{
  bool tracksdiscarded = false;
  // Check particles in the fParticle array
  int prim = -1; // counter how many primaries seen (mainly to constrain search in motherindex remapping)
  LOG(DEBUG) << "Stack: Entering track selection on " << mParticles.size() << " tracks";
  for (auto& thisPart : mParticles) {
    Bool_t store = kTRUE;
    // Get track parameters
    Bool_t isPrimary = (thisPart.getProcess() == 0);
    Int_t iMother = thisPart.getMotherTrackId();
    auto& mother = mParticles[iMother];
    Bool_t motherIsPrimary = (iMother < mNumberOfPrimaryParticles);

    if (isPrimary) {
      prim++;
      // for primaries we are done quickly
      // in fact the primary has already been stored
      // so this should not happen
      store = kTRUE;
    } else {
      // for other particles we potentially need to correct the mother indices
      if (!motherIsPrimary) {
        // the mapping is from the index in the stack to the index in mParticles
        thisPart.SetMotherTrackId(mTrackIDtoParticlesEntry[iMother]);
      } else {
        // for a secondary which is a direct decendant of a primary use a negative index which will be restored later
        thisPart.SetMotherTrackId(iMother - (int)(mTracks->size()));
      }
      // no secondaries; also done
      if (!mStoreSecondaries) {
        store = kFALSE;
        tracksdiscarded = true;
      } else {
        // Calculate number of hits created by this track
        // Note: we only distinguish between no hit and more than 0 hits
        int nHits = thisPart.hasHits();
        // Check for cuts (store primaries in any case)
        if (nHits < mMinHits) {
          store = kFALSE;
          tracksdiscarded = true;
        }
        // only if we have non-trival energy cut
        if (mEnergyCut > 0.) {
          Double_t energy = thisPart.GetEnergy();
          Double_t mass = thisPart.GetMass();
          Double_t eKin = energy - mass;
          if (eKin < mEnergyCut) {
            store = kFALSE;
            tracksdiscarded = true;
          }
        }
        if (keepPhysics(thisPart)) {
          store = kTRUE;
          tracksdiscarded = false;
        }
      }
    }

    store = store || thisPart.getStore();
    thisPart.setStore(store);
  }

  // If flag is set, flag recursively mothers of selected tracks
  //
  if (mStoreMothers) {
    for (auto& particle : mParticles) {
      if (particle.getStore()) {
        Int_t iMother = particle.getMotherTrackId();
        while (iMother >= 0) {
          auto& mother = mParticles[iMother];
          mother.setStore(true);
          iMother = mother.getMotherTrackId();
        } // while mother
      }   // store ?
    }     // particle loop
  }

  return !tracksdiscarded;
}

bool Stack::isPrimary(const MCTrack& part)
{
  /** check if primary **/
  if (part.getProcess() == kPPrimary || part.getMotherTrackId() < 0) {
    return true;
  }
  /** not primary **/
  return false;
}

bool Stack::isFromPrimaryDecayChain(const MCTrack& part)
{
  /** check if the particle is from the 
      decay chain of a primary particle **/

  /** check if from decay **/
  if (part.getProcess() != kPDecay) {
    return false;
  }
  /** check if mother is primary **/
  auto imother = part.getMotherTrackId();
  auto mother = mParticles[imother];
  if (isPrimary(mother)) {
    return true;
  }
  /** else check if mother is from primary decay **/
  return isFromPrimaryDecayChain(mother);
}

bool Stack::isFromPrimaryPairProduction(const MCTrack& part)
{
  /** check if the particle is from 
      pair production from a particle
      belonging to the primary decay chain **/

  /** check if from pair production **/
  if (part.getProcess() != kPPair) {
    return false;
  }
  /** check if mother is primary **/
  auto imother = part.getMotherTrackId();
  auto mother = mParticles[imother];
  if (isPrimary(mother)) {
    return true;
  }
  /** else check if mother is from primary decay **/
  return isFromPrimaryDecayChain(mother);
}

bool Stack::keepPhysics(const MCTrack& part)
{
  //
  // Some particles have to kept on the stack for reasons motivated
  // by physics analysis. Decision is put here.
  //

  if (isPrimary(part)) {
    return true;
  }
  if (isFromPrimaryDecayChain(part)) {
    return true;
  }
  if (isFromPrimaryPairProduction(part)) {
    return true;
  }

  return false;
}

TClonesArray* Stack::GetListOfParticles()
{
  LOG(FATAL) << "Stack::GetListOfParticles interface not implemented\n";
  return nullptr;
}

bool Stack::isTrackDaughterOf(int trackid, int parentid) const
{
  // a daughter trackid should be larger than parentid
  if (trackid < parentid) {
    return false;
  }

  if (trackid == parentid) {
    return true;
  }

  auto mother = getMotherTrackId(trackid);
  while (mother != -1) {
    if (mother == parentid) {
      return true;
    }
    mother = getMotherTrackId(mother);
  }
  return false;
}

void Stack::fillParentIDs(std::vector<int>& parentids) const
{
  parentids.clear();
  int mother = mIndexOfCurrentTrack;
  do {
    if (mother != -1) {
      parentids.push_back(mother);
    }
    mother = getMotherTrackId(mother);
  } while (mother != -1);
}

void Stack::ReorderKine(std::vector<MCTrack>& particles, std::vector<int>& reOrderedIndices)
{
  //
  // Particles are ordered in a way that descendants of a particle appear next to each other.
  // This has the advantage that their position in the stack can be identified by two number,
  // for example the index of the first and last descentant.
  // The result of the ordering is returned via the look-up table reOrderedIndices
  //

  Int_t ntr = (int)(particles.size());
  std::vector<bool> done(ntr, false);

  int indexoffset = mTracks->size();
  Int_t index = 0;
  Int_t imoOld = 0;
  for (Int_t i = 0; i < ntr; i++) {
    reOrderedIndices[i] = i;
  }

  for (Int_t i = -1; i < ntr; i++) {
    if (i != -1) {
      // secondaries
      if (!done[i]) {
        reOrderedIndices[index] = i;
        index++;
        done[i] = true;
      }
      imoOld = i;
    } else {
      // current primary
      imoOld = mIndexOfCurrentPrimary - indexoffset;
    }
    for (Int_t j = i + 1; j < ntr; j++) {
      if (!done[j]) {
        if ((particles[j]).getMotherTrackId() == imoOld) {
          reOrderedIndices[index] = j;
          index++;
          done[j] = true;
        } // child found
      }   // done
    }     // j
  }       // i
}

FairGenericStack* Stack::CloneStack() const { return new o2::data::Stack(*this); }

ClassImp(o2::data::Stack);
