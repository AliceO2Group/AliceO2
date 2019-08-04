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

#include "TLorentzVector.h" // for TLorentzVector
#include "TParticle.h"      // for TParticle
#include "TRefArray.h"      // for TRefArray
#include "TVirtualMC.h"     // for VMC

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
    // mParticles(new TClonesArray("TParticle", size)),
    mParticles(),
    mTracks(new std::vector<o2::MCTrack>),
    mTrackIDtoParticlesEntry(1000000, -1),
    mIndexMap(),
    mIndexOfCurrentTrack(-1),
    mNumberOfPrimaryParticles(0),
    mNumberOfEntriesInParticles(0),
    mNumberOfEntriesInTracks(0),
    mStoreMothers(kTRUE),
    mStoreSecondaries(kTRUE),
    mMinHits(1),
    mEnergyCut(0.),
    mTrackRefs(new std::vector<o2::TrackReference>),
    mIndexedTrackRefs(new typename std::remove_pointer<decltype(mIndexedTrackRefs)>::type),
    mIsG4Like(false)
{
  auto vmc = TVirtualMC::GetMC();
  if (vmc && strcmp(vmc->GetName(), "TGeant4") == 0) {
    mIsG4Like = true;
  }
}

Stack::Stack(const Stack& rhs)
  : FairGenericStack(rhs),
    mStack(),
    mParticles(),
    mTracks(nullptr),
    mIndexMap(),
    mIndexOfCurrentTrack(-1),
    mNumberOfPrimaryParticles(0),
    mNumberOfEntriesInParticles(0),
    mNumberOfEntriesInTracks(0),
    mStoreMothers(rhs.mStoreMothers),
    mStoreSecondaries(rhs.mStoreSecondaries),
    mMinHits(rhs.mMinHits),
    mEnergyCut(rhs.mEnergyCut),
    mTrackRefs(new std::vector<o2::TrackReference>),
    mIsG4Like(rhs.mIsG4Like)
{
  LOG(DEBUG) << "copy constructor called" << FairLogger::endl;
  mTracks = new std::vector<MCTrack>();
  // LOG(INFO) << "Stack::Stack(rhs) " << this << " mTracks " << mTracks << std::endl;
}

Stack::~Stack()
{
  if (mTracks) {
    delete mTracks;
  }
}

Stack& Stack::operator=(const Stack& rhs)
{
  LOG(FATAL) << "operator= called" << FairLogger::endl;
  // check assignment to self
  if (this == &rhs) {
    return *this;
  }

  // base class assignment
  FairGenericStack::operator=(rhs);

  // assignment operator
  // mParticles = new std::vector<TParticle*>;//new TClonesArray("TParticle", rhs.mParticles->GetSize());
  mTracks = new std::vector<MCTrack>(rhs.mTracks->size());
  mIndexOfCurrentTrack = -1;
  mNumberOfPrimaryParticles = 0;
  mNumberOfEntriesInParticles = 0;
  mNumberOfEntriesInTracks = 0;
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
                      TMCProcess proc, Int_t& ntr, Double_t weight, Int_t is, Int_t secondparentID)
{
  // Create new TParticle and add it to the TParticle array
  Int_t trackId = mNumberOfEntriesInParticles;
  // Set track variable
  ntr = trackId;

  Int_t nPoints = 0;
  Int_t daughter1Id = -1;
  Int_t daughter2Id = -1;

  // LOG(INFO) << "Pushing " << trackId << " with parent " << parentId << FairLogger::endl;

  TParticle p(pdgCode, trackId, parentId, nPoints, daughter1Id, daughter2Id, px, py, pz, e, vx, vy, vz, time);
  p.SetPolarisation(polx, poly, polz);
  p.SetWeight(weight);
  p.SetUniqueID(proc); // using the unique ID to transfer process ID
  mNumberOfEntriesInParticles++;

  // currently I only know of G4 who pushes particles like this (but never pops)
  // so we have to register the particles here
  if (mIsG4Like && parentId >= 0) {
    // p.SetStatusCode(mParticles.size());
    mParticles.emplace_back(p);
    mTransportedIDs.emplace_back(p.GetStatusCode());
    const auto trackID = p.GetStatusCode();
    insertInVector(mTrackIDtoParticlesEntry, trackID, (int)(mParticles.size() - 1));

    mCurrentParticle = p;
  }

  // Increment counter
  if (parentId < 0) {
    mNumberOfPrimaryParticles++;
    mPrimaryParticles.push_back(p);
  }

  // Push particle on the stack if toBeDone is set
  if (toBeDone == 1) {
    mStack.push(p);
  }
}

void Stack::PushTrack(int toBeDone, TParticle const& p)
{
  auto parentId = p.GetMother(0);
  // currently I only know of G4 who pushes particles like this (but never pops)
  // so we have to register the particles here
  if (mIsG4Like && parentId >= 0) {
    //p.SetStatusCode(mParticles.size());
    mParticles.emplace_back(p);
    mTransportedIDs.emplace_back(p.GetStatusCode());
    mCurrentParticle = p;
  }

  // Increment counter
  if (parentId < 0) {
    mNumberOfPrimaryParticles++;
    // fix trackID
    mPrimaryParticles.push_back(p);
  }

  // Push particle on the stack if toBeDone is set
  if (toBeDone == 1) {
    mStack.push(p);
  }
}

/// Set the current track number
/// Declared in TVirtualMCStack
/// \param iTrack track number
void Stack::SetCurrentTrack(Int_t iTrack)
{
  mIndexOfCurrentTrack = iTrack;

  if (mIsG4Like) {
    if (iTrack < mPrimaryParticles.size()) {
      // This interface is called by Geant4 when activating a certain primary
      auto& p = mPrimaryParticles[iTrack];
      mIndexOfPrimaries.emplace_back(mParticles.size());
      mParticles.emplace_back(p);
      mTransportedIDs.emplace_back(p.GetStatusCode());
      insertInVector(mTrackIDtoParticlesEntry, p.GetStatusCode(), (int)(mParticles.size() - 1));
      mCurrentParticle = p;
    }
  }
}

void Stack::notifyFinishPrimary()
{
  // someone notifies us that a primary is finished
  // this means we can do some filtering and cleanup

  mPrimariesDone++;
  LOG(DEBUG) << "Finish primary hook " << mPrimariesDone << FairLogger::endl;
  mCleanupCounter++;
  if (mCleanupCounter == mCleanupThreshold || mCleanupCounter == mPrimaryParticles.size() ||
      mPrimariesDone == mPrimaryParticles.size()) {
    finishCurrentPrimary();
    mCleanupCounter = 0;
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

  // If end of stack: Return empty pointer
  if (mStack.empty()) {
    if (mParticles.size() > 0) { // make sure something was tracked at all
      notifyFinishPrimary();
    }
    iTrack = -1;
    return nullptr;
  }

  // If not, get next particle from stack
  mCurrentParticle = mStack.top();
  mStack.pop();

  if (mCurrentParticle.GetMother(0) < 0) {
    // particle is primary -> indicates that previous particle finished
    if (mParticles.size() > 0) {
      notifyFinishPrimary();
    }
    mIndexOfPrimaries.emplace_back(mParticles.size());
  }
  mParticles.emplace_back(mCurrentParticle);
  mTransportedIDs.emplace_back(mCurrentParticle.GetStatusCode());
  insertInVector(mTrackIDtoParticlesEntry, mCurrentParticle.GetStatusCode(), (int)(mParticles.size() - 1));

  mIndexOfCurrentTrack = mCurrentParticle.GetStatusCode();
  iTrack = mIndexOfCurrentTrack;

  if (o2::conf::SimCutParams::Instance().trackSeed) {
    auto hash = getHash(mCurrentParticle);
    // LOG(INFO) << "SEEDING NEW TRACK USING HASH" << hash;
    // init seed per track
    gRandom->SetSeed(hash);

    // NOTE: THE BETTER PLACE WOULD BE IN PRETRACK HOOK BUT THIS DOES NOT SEEM TO WORK
    // WORKS ONLY WITH G3 SINCE G4 DOES NOT CALL THIS FUNCTION
  }

  // LOG(INFO) << "transporting ID " << mIndexOfCurrentTrack << "\n";
  return &mCurrentParticle;
}

TParticle* Stack::PopPrimaryForTracking(Int_t iPrim)
{
  // This function is used by Geant4 to setup their own internal stack

  // Remark: Contrary to what the interface name is suggesting
  // this is not a pop operation (but rather a get)

  // Test for index
  if (iPrim < 0 || iPrim >= mNumberOfPrimaryParticles) {
    LOG(FATAL) << "Stack::PopPrimaryForTracking: Stack: Primary index out of range! " << iPrim << " ";
    return nullptr;
  }
  // Return the iPrim-th TParticle from the fParticle array. This should be
  // a primary.
  return &mPrimaryParticles[iPrim];
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

void Stack::finishCurrentPrimary()
{
  // Here transport of a primary and all its secondaries is finished
  // we can do some cleanup of the memory structures
  LOG(DEBUG) << "STACK: Cleaning up" << FairLogger::endl;
  auto selected = selectTracks();
  // loop over current particle buffer
  int index = 0;
  int indexoffset = mTracks->size();
  int neglected = 0;
  std::vector<int> indicesKept;
  for (const auto& particle : mParticles) {
    if (particle.getStore() || !mPruneKinematics) {
      // map the global track index to the new persistent index
      // FIXME: only fill the map for non-trivial mappings in which mTransportedIDs[index]!=mTracks->size();
      mIndexMap[mTransportedIDs[index]] = mTracks->size();
      auto mother = particle.getMotherTrackId();
      assert(mother < index);
      mTracks->emplace_back(particle);
      indicesKept.emplace_back(index);
      if (mother != -1) {
        auto iter = std::find_if(indicesKept.begin(), indicesKept.end(), [mother](int x) { return x == mother; });
        if (iter != indicesKept.end()) {
          // complexity should be constant
          auto newmother = std::distance(indicesKept.begin(), iter);
          mTracks->back().SetMotherTrackId(newmother + indexoffset);
        }
      }
      // LOG(INFO) << "Adding to map " << mTransportedIDs[index] << " to " << mIndexMap[mTransportedIDs[index]];
    } else {
      neglected++;
    }
    index++;
    mTracksDone++;
  }
  // we can now clear the particles buffer!
  mParticles.clear();
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
      LOG(FATAL) << "No detList passed to Stack" << FairLogger::endl;
    }
    auto iter = detList->MakeIterator();
    while (auto det = iter->Next()) {
      auto o2det = dynamic_cast<o2::base::Detector*>(det);
      if (o2det) {
        mActiveDetectors.emplace_back(o2det);
      } else {
        LOG(INFO) << "Found nonconforming detector " << det->GetName() << FairLogger::endl;
      }
    }
  }

  LOG(DEBUG) << "Stack::UpdateTrackIndex: Stack: Updating track indices...";
  Int_t nColl = 0;

  //  // First update mother ID in MCTracks
  //  //for (Int_t i = 0; i < mNumberOfEntriesInTracks; i++) {
  //  for (Int_t i = 0; i < mTracks->size(); i++) {
  //    auto& track = (*mTracks)[i];
  //    Int_t iMotherOld = track.getMotherTrackId();
  //    if (iMotherOld == -1) {
  //      // no need to lookup this case
  //      continue;
  //    }
  //    auto iter = mIndexMap.find(iMotherOld);
  //    //if (iter == mIndexMap.end()) {
  //    //  LOG(FATAL) << "Stack::UpdateTrackIndex: Stack: Track index "
  //    //             << iMotherOld << " not found index map! ";
  //    //}
  //    //track.SetMotherTrackId(iter->second);
  //    //LOG(INFO) << "Mapping mother " << iMotherOld << " to " << iter->second;
  //  }

  // update track references
  // use some caching since repeated trackIDs
  for (auto& ref : *mTrackRefs) {
    const auto id = ref.getTrackID();
    auto iter = mIndexMap.find(id);
    if (iter == mIndexMap.end()) {
      LOG(INFO) << "Invalid trackref ... needs to be removed\n";
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

  // make final indexed container for track references
  // fill empty
  for (auto& ref : *mTrackRefs) {
    if (ref.getTrackID() >= 0) {
      mIndexedTrackRefs->addElement(ref.getTrackID(), ref);
    }
  }

  for (auto det : mActiveDetectors) {
    // update the track indices by delegating to specialized detector functions
    det->updateHitTrackIndices(mIndexMap);
  } // List of active detectors

  LOG(DEBUG) << "Stack::UpdateTrackIndex: ...stack and " << nColl << " collections updated.";
}

void Stack::FinishPrimary()
{
  if (mIsG4Like) {
    notifyFinishPrimary();
  }
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
  if (!mIsExternalMode && (mPrimariesDone != mPrimaryParticles.size())) {
    LOG(FATAL) << "Inconsistency in primary particles treated " << mPrimariesDone << " vs expected "
               << mPrimaryParticles.size() << "\n(This points to a flaw in the stack logic)" << FairLogger::endl;
  }
  mPrimariesDone = 0;
  mPrimaryParticles.clear();
  mTrackRefs->clear();
  mIndexedTrackRefs->clear();
  mTrackIDtoParticlesEntry.clear();
  mHitCounter = 0;
}

void Stack::Register()
{
  FairRootManager::Instance()->RegisterAny("MCTrack", mTracks, kTRUE);
  FairRootManager::Instance()->RegisterAny("TrackRefs", mTrackRefs, kTRUE);
  FairRootManager::Instance()->RegisterAny("IndexedTrackRefs", mIndexedTrackRefs, kTRUE);
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
  if (option)
    verbose = 1;
  Print(verbose);
}

void Stack::addHit(int iDet) { addHit(iDet, mParticles.size() - 1); }
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
    Int_t iMother = thisPart.getMotherTrackId();
    if (iMother < 0) {
      prim++;
      // for primaries we are done quickly
      store = kTRUE;
    } else {
      // for other particles we potentially need to correct the mother indices
      thisPart.SetMotherTrackId(mTrackIDtoParticlesEntry[iMother]);

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
      }
    }
    // LOG(INFO) << "storing " << store << FairLogger::endl;
    thisPart.setStore(store);
  }

  // If flag is set, flag recursively mothers of selected tracks
  if (mStoreMothers) {
    for (auto& particle : mParticles) {
      if (particle.getStore()) {
        Int_t iMother = particle.getMotherTrackId();
        while (iMother >= 0) {
          auto& mother = mParticles[iMother];
          mother.setStore(true);
          iMother = mother.getMotherTrackId();
        }
      }
    }
  }

  return !tracksdiscarded;
}

TClonesArray* Stack::GetListOfParticles()
{
  LOG(FATAL) << "Stack::GetListOfParticles interface not implemented\n"
             << FairLogger::endl;
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

FairGenericStack* Stack::CloneStack() const { return new o2::data::Stack(*this); }
ClassImp(o2::data::Stack);
