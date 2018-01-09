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

#include "DetectorsBase/DetID.h"
#include "DetectorsBase/Detector.h"
#include "SimulationDataFormat/Stack.h"
#include "SimulationDataFormat/MCTrack.h"

#include "FairDetector.h"     // for FairDetector
#include "FairLogger.h"       // for MESSAGE_ORIGIN, FairLogger
#include "SimulationDataFormat/BaseHits.h"
#include "FairRootManager.h"

#include "TLorentzVector.h"   // for TLorentzVector
#include "TParticle.h"        // for TParticle
#include "TRefArray.h"        // for TRefArray
#include "TVirtualMC.h"       // for VMC

#include <cstddef>           // for NULL

using std::cout;
using std::endl;
using std::pair;
using namespace o2::Data;

Stack::Stack(Int_t size)
  : FairGenericStack(),
    mStack(),
    //mParticles(new TClonesArray("TParticle", size)),
    mParticles(),
	mTracks(new std::vector<o2::MCTrack>),
    mIndexMap(),
    mIndexOfCurrentTrack(-1),
    mNumberOfPrimaryParticles(0),
    mNumberOfEntriesInParticles(0),
    mNumberOfEntriesInTracks(0),
    mIndex(0),
    mStoreMothers(kTRUE),
    mStoreSecondaries(kTRUE),
    mMinHits(1),
    mEnergyCut(0.),
    mLogger(FairLogger::GetLogger())
{
  auto vmc = TVirtualMC::GetMC();
  if (!vmc) {
    LOG(FATAL) << "Must have VMC initialized before Stack construction" << FairLogger::endl;
  }
  if (strcmp(vmc->GetName(), "TGeant4") == 0 ) {
    mIsG4Like = true;
  }
}

Stack::Stack(const Stack &rhs)
  : FairGenericStack(rhs),
    mStack(),
    mParticles(),
    mTracks(nullptr),
    mIndexMap(),
    mIndexOfCurrentTrack(-1),
    mNumberOfPrimaryParticles(0),
    mNumberOfEntriesInParticles(0),
    mNumberOfEntriesInTracks(0),
    mIndex(0),
    mStoreMothers(rhs.mStoreMothers),
    mStoreSecondaries(rhs.mStoreSecondaries),
    mMinHits(rhs.mMinHits),
    mEnergyCut(rhs.mEnergyCut),
    mLogger(FairLogger::GetLogger()),
    mIsG4Like(rhs.mIsG4Like)
{
  LOG(FATAL) << "copy constructor called" << FairLogger::endl;
  mTracks = new std::vector<MCTrack>(rhs.mTracks->size());
}

Stack::~Stack()
{
  if (mTracks) {
    delete mTracks;
  }
}

Stack &Stack::operator=(const Stack &rhs)
{
  LOG(FATAL) << "operator= called" << FairLogger::endl;
  // check assignment to self
  if (this == &rhs) { return *this; }

  // base class assignment
  FairGenericStack::operator=(rhs);

  // assignment operator
 // mParticles = new std::vector<TParticle*>;//new TClonesArray("TParticle", rhs.mParticles->GetSize());
  mTracks = new std::vector<MCTrack>(rhs.mTracks->size());
  mIndexOfCurrentTrack = -1;
  mNumberOfPrimaryParticles = 0;
  mNumberOfEntriesInParticles = 0;
  mNumberOfEntriesInTracks = 0;
  mIndex = 0;
  mStoreMothers = rhs.mStoreMothers;
  mStoreSecondaries = rhs.mStoreSecondaries;
  mMinHits = rhs.mMinHits;
  mEnergyCut = rhs.mEnergyCut;
  mLogger = nullptr;
  mIsG4Like = rhs.mIsG4Like;

  return *this;
}

void Stack::PushTrack(Int_t toBeDone, Int_t parentId, Int_t pdgCode, Double_t px, Double_t py, Double_t pz, Double_t e,
                      Double_t vx, Double_t vy, Double_t vz, Double_t time, Double_t polx, Double_t poly, Double_t polz,
                      TMCProcess proc, Int_t &ntr, Double_t weight, Int_t is)
{

  PushTrack(toBeDone, parentId, pdgCode, px, py, pz, e, vx, vy, vz, time, polx, poly, polz, proc, ntr, weight, is, -1);
}

void Stack::PushTrack(Int_t toBeDone, Int_t parentId, Int_t pdgCode, Double_t px, Double_t py, Double_t pz, Double_t e,
                      Double_t vx, Double_t vy, Double_t vz, Double_t time, Double_t polx, Double_t poly, Double_t polz,
                      TMCProcess proc, Int_t &ntr, Double_t weight, Int_t is, Int_t secondparentID)
{
  // Create new TParticle and add it to the TParticle array
  Int_t trackId = mNumberOfEntriesInParticles;
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
    //p.SetStatusCode(mParticles.size());
    mParticles.emplace_back(p);
    mTransportedIDs.emplace_back(p.GetStatusCode());
    mCurrentParticle = p;
  }
  
  // Increment counter
  if (parentId < 0) {
    mNumberOfPrimaryParticles++;
    mPrimaryParticles.push_back(p);
  }

  // Set argument variable
  ntr = trackId;

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
  
   if(mIsG4Like) {
     if(iTrack < mPrimaryParticles.size()) {   
       // This interface is called by Geant4 when activating a certain primary
       auto& p = mPrimaryParticles[iTrack];
       mIndexOfPrimaries.emplace_back(mParticles.size());
       mParticles.emplace_back(p);
       mTransportedIDs.emplace_back(p.GetStatusCode());
       mCurrentParticle=p;
     }
   }
}


void Stack::notifyFinishPrimary() {
  // someone notifies us that a primary is finished
  // this means we can do some filtering and cleanup
  mPrimariesDone++;
  LOG(DEBUG) << "Finish primary hook " << mPrimariesDone << FairLogger::endl;
  mCleanupCounter++;
  if (mCleanupCounter == mCleanupThreshold 
      || mCleanupCounter == mPrimaryParticles.size()
      || mPrimariesDone == mPrimaryParticles.size() ) {
    finishPrimary();
    mCleanupCounter = 0;
  }
}

TParticle* Stack::PopNextTrack(Int_t& iTrack)
{
  // This functions is mainly used by Geant3?
  
  // If end of stack: Return empty pointer
  if (mStack.empty()) {
    notifyFinishPrimary();
    iTrack = -1;
    return nullptr;
  }

  // If not, get next particle from stack
  mCurrentParticle = mStack.top();
  mStack.pop();
  
  if (mCurrentParticle.GetMother(0) < 0) {
    // particle is primary -> indicates that previous particle finished
    if(mParticles.size() > 0){
      notifyFinishPrimary();
    }
    mIndexOfPrimaries.emplace_back(mParticles.size());
  }
  mParticles.emplace_back(mCurrentParticle);
  mTransportedIDs.emplace_back(mCurrentParticle.GetStatusCode());

  mIndexOfCurrentTrack = mCurrentParticle.GetStatusCode();
  iTrack = mIndexOfCurrentTrack;
  
  // LOG(INFO) << "transporting ID " << mIndexOfCurrentTrack << "\n"; 
  return &mCurrentParticle;
}

TParticle* Stack::PopPrimaryForTracking(Int_t iPrim)
{
  // This function is used by Geant4 to setup their own onternal stack

  // Remark: Contrary to what the interface name is suggesting
  // this is not a pop operation (but rather a get)

  // Test for index
  if (iPrim < 0 || iPrim >= mNumberOfPrimaryParticles) {
    if (mLogger) {
      mLogger->Fatal(MESSAGE_ORIGIN, "Stack: Primary index out of range! %i ", iPrim);
    }
    Fatal("Stack::PopPrimaryForTracking", "Index out of range");
    return nullptr;
  }
  // Return the iPrim-th TParticle from the fParticle array. This should be
  // a primary.
  return &mPrimaryParticles[iPrim];
}

void Stack::FillTrackArray()
{
  /// This interface is not implemented since we are filtering/filling the output array
  /// after each primary ... just give a summary message
  LOG(INFO) << "Stack: " << mTracks->size() << " out of " << mNumberOfEntriesInParticles << " stored \n";
}

void Stack::finishPrimary() {
  // Here transport of a primary and all its secondaries is finished
  // we can do some cleanup of the memory structures
  LOG(DEBUG) << "STACK: Cleaning up" << FairLogger::endl;
  auto selected = selectTracks();
  // loop over current particle buffer
  int index=0;
  for(const auto& particle : mParticles) {
    if (particle.getStore()) {
      // map the global track index to the new persistent index
      // FIXME: only fill the map for non-trivial mappings in which mTransportedIDs[index]!=mTracks->size();
      mIndexMap[mTransportedIDs[index]] = mTracks->size();
      mTracks->emplace_back(particle);
    }
    index++;
    mTracksDone++;
  }
  // we can now clear the particles buffer!
  mParticles.clear();
  mTransportedIDs.clear();
  mIndexOfPrimaries.clear();
}

void Stack::UpdateTrackIndex(TRefArray *detList)
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
      auto o2det = dynamic_cast<o2::Base::Detector*>(det);
      if (o2det) {
        mActiveDetectors.emplace_back(o2det);
      } else {
        LOG(INFO) << "Found nonconforming detector" << FairLogger::endl;
      }
    }
  }

  if (mLogger) {
    mLogger->Debug(MESSAGE_ORIGIN, "Stack: Updating track indices...");
  } else {
    cout << "Stack: Updating track indices..." << endl;
  }
  Int_t nColl = 0;

  // First update mother ID in MCTracks
  for (Int_t i = 0; i < mNumberOfEntriesInTracks; i++) {
    auto& track = (*mTracks)[i];
    Int_t iMotherOld = track.getMotherTrackId();
    if (iMotherOld == -1) {
      // no need to lookup this case
      continue;
    }
    auto iter = mIndexMap.find(iMotherOld);
    if (iter == mIndexMap.end()) {
      if (mLogger) {
        mLogger->Fatal(MESSAGE_ORIGIN, "Stack: Track index %i not found index map! ", iMotherOld);
      }
      Fatal("Stack::UpdateTrackIndex", "Track index not found in map");
    }
    track.SetMotherTrackId(iter->second);
  }

  for(auto det : mActiveDetectors) {
    // update the track indices by delegating to specialized detector functions
    det->updateHitTrackIndices(mIndexMap);
  } // List of active detectors

  if (mLogger) {
    mLogger->Debug(MESSAGE_ORIGIN, "...stack and  %i collections updated.", nColl);
  } else {
    cout << "...stack and  " << nColl << " collections updated." << endl;
  }
}

void Stack::Reset()
{
  mIndex = 0;
  mIndexOfCurrentTrack = -1;
  mNumberOfPrimaryParticles = mNumberOfEntriesInParticles = mNumberOfEntriesInTracks = 0;
  while (!mStack.empty()) {
    mStack.pop();
  }
  mParticles.clear();
  mTracks->clear();
  if (mPrimariesDone != mPrimaryParticles.size()) {
    LOG(FATAL) << "Inconsistency in primary particles treated vs expected (This points " 
               << "to a flaw in the stack logic)" << FairLogger::endl; 
  }
  mPrimariesDone = 0;
  mPrimaryParticles.clear();
}

void Stack::Register()
{
  FairRootManager::Instance()->RegisterAny("MCTrack", mTracks, kTRUE);
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
  if ( option ) verbose = 1;
  Print(verbose);
}

void Stack::addHit(int iDet)
{
  addHit(iDet, mParticles.size()-1);
}

void Stack::addHit(int iDet, Int_t iTrack)
{
  auto& part=mParticles[iTrack];
  part.setHit(iDet);
}

Int_t Stack::GetCurrentParentTrackNumber() const
{
  TParticle *currentPart = GetCurrentTrack();
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
  for(auto& thisPart : mParticles) {
    Bool_t store = kTRUE;

    // Get track parameters
    Int_t iMother = thisPart.getMotherTrackId();
    if (iMother < 0) {
      prim++;
      // for primaries we are done quickly
      store = kTRUE;
    }
    else {
      // for other particles we (potentially need to correct the mother indices
      auto rangestart = mTransportedIDs.begin() + mIndexOfPrimaries[prim];
      auto rangeend = (prim < (mIndexOfPrimaries.size()-1)) ? mTransportedIDs.begin() + mIndexOfPrimaries[prim+1] : mTransportedIDs.end();
      // auto rangeend = mTransportedIDs.end();

      auto iter = std::find_if(rangestart, rangeend, [iMother](int x){return x == iMother;});
      if (iter!=rangeend) {
        // complexity should be constant
        auto newmother = std::distance(mTransportedIDs.begin(), iter);
        // LOG(INFO) << "Fixing mother from " << iMother << " to " << newmother << FairLogger::endl;
        thisPart.SetMotherTrackId(newmother);
      }

      // no secondaries; also done
      if (!mStoreSecondaries) {
        store = kFALSE;
        tracksdiscarded = true;
      }
      else {
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
  LOG(FATAL) << "Stack::GetListOfParticles interface not implemented\n" << FairLogger::endl;
  return nullptr;
}

FairGenericStack *Stack::CloneStack() const
{
  return new o2::Data::Stack(*this);
}

ClassImp(o2::Data::Stack)
