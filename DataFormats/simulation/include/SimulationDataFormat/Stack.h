// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Stack.h
/// \brief Definition of the Stack class
/// \author M. Al-Turany - June 2014

#ifndef ALICEO2_DATA_STACK_H_
#define ALICEO2_DATA_STACK_H_

#include "DetectorsCommonDataFormats/DetID.h"
#include "FairGenericStack.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/MCEventStats.h"

#include "Rtypes.h"
#include "TParticle.h"

#include <map>
#include <memory>
#include <stack>
#include <utility>

class TClonesArray;
class TRefArray;

namespace o2
{
namespace base
{
class Detector;
}

namespace data
{
/// This class handles the particle stack for the transport simulation.
/// For the stack FILO functunality, it uses the STL stack. To store
/// the tracks during transport, a TParticle array is used.
/// At the end of the event, tracks satisfying the filter criteria
/// are copied to a MCTrack array, which is stored in the output.
///
/// The filtering criteria for the output tracks are:
/// - primary tracks are stored in any case.
/// - secondary tracks are stored if they have a minimal number of
///   hits (sum of all detectors) and a minimal energy, or are the
///
/// The storage of secondaries can be switched off.
/// The storage of all mothers can be switched off.
/// By default, the minimal number of hits is 1 and the energy cut is 0.
class Stack : public FairGenericStack
{
 public:
  /// Default constructor
  /// \param size Estimated track number
  Stack(Int_t size = 100);

  /// Default destructor
  ~Stack() override;

  /// Add a TParticle to the stack.
  /// Declared in TVirtualMCStack
  /// \param toBeDone Flag for tracking
  /// \param parentID Index of mother particle
  /// \param pdgCode Particle type (PDG encoding)
  /// \param px,py,pz Momentum components at start vertex [GeV]
  /// \param e Total energy at start vertex [GeV]
  /// \param vx,vy,vz Coordinates of start vertex [cm]
  /// \param time Start time of track [s]
  /// \param polx,poly,polz Polarisation vector
  /// \param proc Production mechanism (VMC encoding)
  /// \param ntr Track number (filled by the stack)
  /// \param weight Particle weight
  /// \param is Generation status code (whatever that means)
  void PushTrack(Int_t toBeDone, Int_t parentID, Int_t pdgCode, Double_t px, Double_t py, Double_t pz, Double_t e,
                 Double_t vx, Double_t vy, Double_t vz, Double_t time, Double_t polx, Double_t poly, Double_t polz,
                 TMCProcess proc, Int_t& ntr, Double_t weight, Int_t is) override;

  void PushTrack(Int_t toBeDone, Int_t parentID, Int_t pdgCode, Double_t px, Double_t py, Double_t pz, Double_t e,
                 Double_t vx, Double_t vy, Double_t vz, Double_t time, Double_t polx, Double_t poly, Double_t polz,
                 TMCProcess proc, Int_t& ntr, Double_t weight, Int_t is, Int_t secondParentId) override;

  // similar function taking a particle
  void PushTrack(Int_t toBeDone, TParticle const&);

  /// Get next particle for tracking from the stack.
  /// Declared in TVirtualMCStack
  /// Returns a pointer to the TParticle of the track
  /// \param iTrack index of popped track (return)
  TParticle* PopNextTrack(Int_t& iTrack) override;

  /// Get primary particle by index for tracking from stack
  /// Declared in TVirtualMCStack
  /// Returns a pointer to the TParticle of the track
  /// \param iPrim index of primary particle
  TParticle* PopPrimaryForTracking(Int_t iPrim) override;

  /// Set the current track number
  /// Declared in TVirtualMCStack
  /// \param iTrack track number
  void SetCurrentTrack(Int_t iTrack) override;

  /// Get total number of tracks
  /// Declared in TVirtualMCStack
  Int_t GetNtrack() const override { return mNumberOfEntriesInParticles; }
  /// Get number of primary tracks
  /// Declared in TVirtualMCStack
  Int_t GetNprimary() const override { return mNumberOfPrimaryParticles; }
  /// Get the current track's particle
  /// Declared in TVirtualMCStack
  TParticle* GetCurrentTrack() const override
  {
    // the const cast is necessary ... the interface should have been `const TParticle* GetCurrentParticle() const`
    return const_cast<TParticle*>(&mCurrentParticle);
  }

  /// Get the number of the current track
  /// Declared in TVirtualMCStack
  Int_t GetCurrentTrackNumber() const override { return mIndexOfCurrentTrack; }
  /// Get the track number of the parent of the current track
  /// Declared in TVirtualMCStack
  Int_t GetCurrentParentTrackNumber() const override;

  /// Returns the production process of the current track
  TMCProcess GetProdProcessOfCurrentTrack() const;

  /// Fill the MCTrack output array, applying filter criteria
  void FillTrackArray() override;

  /// Update the track index in the MCTracks and data produced by detectors
  void UpdateTrackIndex(TRefArray* detArray = nullptr) override;

  /// Finish primary
  void FinishPrimary() override;

  /// Resets arrays and stack and deletes particles and tracks
  void Reset() override;

  /// Register the MCTrack array to the Root Manager
  void Register() override;

  /// Output to screen
  /// \param iVerbose: 0=events summary, 1=track info

  virtual void Print(Int_t iVerbose = 0) const;

  /// Output to screen (derived from base class)
  /// \param option: 0=events summary, non0=track info
  void Print(Option_t* option = nullptr) const override;

  /// Modifiers
  void StoreSecondaries(Bool_t choice = kTRUE) { mStoreSecondaries = choice; }
  void pruneKinematics(bool choice = true) { mPruneKinematics = choice; }
  void setMinHits(Int_t min) { mMinHits = min; }
  void SetEnergyCut(Double_t eMin) { mEnergyCut = eMin; }
  void StoreMothers(Bool_t choice = kTRUE) { mStoreMothers = choice; }
  /// Increment number of hits for the current track in a given detector
  /// \param iDet  Detector unique identifier
  void addHit(int iDet);

  TClonesArray* GetListOfParticles() override;

  std::vector<MCTrack> const* const getMCTracks() const { return mTracks; }

  /// Clone for worker (used in MT mode only)
  FairGenericStack* CloneStack() const override;

  // receive notification that primary is finished
  void notifyFinishPrimary();

  // methods concerning track references
  void addTrackReference(const o2::TrackReference& p);

  // get primaries
  const std::vector<TParticle>& getPrimaries() const { return mPrimaryParticles; }

  // initialize Stack from external vector containing primaries
  void initFromPrimaries(std::vector<TParticle> const& primaries)
  {
    Reset();
    for (auto p : primaries) {
      PushTrack(1, p);
    }
    mNumberOfPrimaryParticles = primaries.size();
    mNumberOfEntriesInParticles = mNumberOfPrimaryParticles;
  }

  void setExternalMode(bool m) { mIsExternalMode = m; }

  /// Allow to query the **direct** mother track ID of an arbitrary trackID managed by stack
  int getMotherTrackId(int /*trackid*/) const;

  /// query if a track is a direct **or** indirect daughter of a parentID
  /// if trackid is same as parentid it returns true
  bool isTrackDaughterOf(int /*trackid*/, int /*parentid*/) const;

  bool isCurrentTrackDaughterOf(int parentid) const;

  // returns the index of the currently transported primary
  int getCurrentPrimaryIndex() const;

  // Fill container with all parent ids for current track
  // The resulting ids will be in strictly monotonously decreasing order
  void fillParentIDs(std::vector<int>& ids) const;

  /// set MCEventStats (for current event)
  /// used by MCApplication to inject here so that
  /// stack can set some information
  void setMCEventStats(o2::dataformats::MCEventStats* header);

  /// update values in the current event header
  void updateEventStats();

 private:
  /// STL stack (FILO) used to handle the TParticles for tracking
  /// stack entries refer to
  std::stack<TParticle> mStack; //!

  /// Array of TParticles (contains all TParticles put into or created
  /// by the transport)
  std::vector<o2::MCTrack> mParticles; //!
  std::vector<int> mTransportedIDs;    //! prim + sec trackIDs transported for "current" primary
  std::vector<int> mIndexOfPrimaries;  //! index of primaries in mParticles

  std::vector<int> mTrackIDtoParticlesEntry; //! an O(1) mapping of trackID to the entry of mParticles
                                             //! where this track is stored

  /// the current TParticle object
  TParticle mCurrentParticle;

  // keep primary particles in its original form
  // (mainly for the PopPrimaryParticleInterface
  std::vector<TParticle> mPrimaryParticles;

  /// vector of reducded tracks written to the output
  std::vector<o2::MCTrack>* mTracks;

  /// STL map from particle index to persistent track index
  std::map<Int_t, Int_t> mIndexMap; //!

  /// cache active O2 detectors
  std::vector<o2::base::Detector*> mActiveDetectors; //!

  /// Some indices and counters
  Int_t mIndexOfCurrentTrack;        //! Global index of current track
  Int_t mNumberOfPrimaryParticles;   //! Number of primary particles
  Int_t mNumberOfEntriesInParticles; //! Number of entries in mParticles
  Int_t mNumberOfEntriesInTracks;    //! Number of entries in mTracks
  Int_t mIndex;                      //! Used for merging

  /// Variables defining the criteria for output selection
  Bool_t mStoreMothers;
  Bool_t mStoreSecondaries;
  bool mPruneKinematics = false; // whether or not we filter the output kinematics
  Int_t mMinHits;
  Int_t mHitCounter = 0; //! counts hits communicated via addHit
  Double32_t mEnergyCut;

  // variables for the cleanup / filtering procedure
  Int_t mCleanupCounter = 0;   //!
  Int_t mCleanupThreshold = 1; //! a cleanup is initiated every mCleanupThreshold primaries
  Int_t mPrimariesDone = 0;    //!
  Int_t mTracksDone = 0;       //! number of tracks already done

  bool mIsG4Like = false; //! flag indicating if the stack is used in a manner done by Geant4

  bool mIsExternalMode = false; // is stack an external factory or directly used inside simulation?

  // storage for track references
  std::vector<o2::TrackReference>* mTrackRefs = nullptr; //!

  o2::dataformats::MCTruthContainer<o2::TrackReference>* mIndexedTrackRefs = nullptr; //!

  /// a pointer to the current MCEventStats object
  o2::dataformats::MCEventStats* mMCEventStats = nullptr; //!

  /// Mark tracks for output using selection criteria
  /// returns true if all available tracks are selected
  /// returns false if some tracks are discarded
  bool selectTracks();

  Stack(const Stack&);

  Stack& operator=(const Stack&);

  /// function called after each primary
  /// and all its secondaries where transported
  /// this allows applying selection criteria at a much finer granularity
  /// than done with FillTrackArray which is only called once per event
  void finishCurrentPrimary();

  /// Increment number of hits for an arbitrary track in a given detector
  /// \param iDet    Detector unique identifier
  /// \param iTrack  Track number
  void addHit(int iDet, Int_t iTrack);

  ClassDefOverride(Stack, 1);
};

inline void Stack::addTrackReference(const o2::TrackReference& ref) { mTrackRefs->push_back(ref); }

inline int Stack::getCurrentPrimaryIndex() const { return mPrimaryParticles.size() - 1 - mPrimariesDone; }

inline int Stack::getMotherTrackId(int trackid) const
{
  const auto entryinParticles = mTrackIDtoParticlesEntry[trackid];
  return mParticles[entryinParticles].getMotherTrackId();
}

inline bool Stack::isCurrentTrackDaughterOf(int parentid) const
{
  // if parentid is current primary the answer is certainly yes
  if (parentid == getCurrentPrimaryIndex()) {
    return true;
  }

  // otherwise ...
  return isTrackDaughterOf(mIndexOfCurrentTrack, parentid);
}

inline void Stack::setMCEventStats(o2::dataformats::MCEventStats* header)
{
  mMCEventStats = header;
}

inline TMCProcess Stack::GetProdProcessOfCurrentTrack() const
{
  return (TMCProcess)o2::data::Stack::GetCurrentTrack()->GetUniqueID();
}

} // namespace data
} // namespace o2

#endif
