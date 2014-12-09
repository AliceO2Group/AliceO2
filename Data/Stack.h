/// \file Stack.h
/// \brief Definition of the Stack class
/// \author M. Al-Turany - June 2014

#ifndef ALICEO2_DATA_STACK_H_
#define ALICEO2_DATA_STACK_H_

#include "FairGenericStack.h"

#include "DetectorList.h"

#include "Rtypes.h"
#include "TMCProcess.h"

#include <map>
#include <stack>
#include <utility>

class TClonesArray;
class TParticle;
class TRefArray;
class FairLogger;

namespace AliceO2 {
namespace Data {

/// This class handles the particle stack for the transport simulation.
/// For the stack FILO functunality, it uses the STL stack. To store
/// the tracks during transport, a TParticle array is used.
/// At the end of the event, tracks satisfying the filter criteria
/// are copied to a MCTrack array, which is stored in the output.
///
/// The filtering criteria for the output tracks are:
/// - primary tracks are stored in any case.
/// - secondary tracks are stored if they have a minimal number of
///   points (sum of all detectors) and a minimal energy, or are the
///
/// The storage of secondaries can be switched off.
/// The storage of all mothers can be switched off.
/// By default, the minimal number of points is 1 and the energy cut is 0.
class Stack : public FairGenericStack {

public:
  /// Default constructor
  /// \param size Estimated track number
  Stack(Int_t size = 100);

  /// Default destructor
  virtual ~Stack();

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
  virtual void PushTrack(Int_t toBeDone, Int_t parentID, Int_t pdgCode, Double_t px, Double_t py, Double_t pz,
                         Double_t e, Double_t vx, Double_t vy, Double_t vz, Double_t time, Double_t polx, Double_t poly,
                         Double_t polz, TMCProcess proc, Int_t& ntr, Double_t weight, Int_t is);

  virtual void PushTrack(Int_t toBeDone, Int_t parentID, Int_t pdgCode, Double_t px, Double_t py, Double_t pz,
                         Double_t e, Double_t vx, Double_t vy, Double_t vz, Double_t time, Double_t polx, Double_t poly,
                         Double_t polz, TMCProcess proc, Int_t& ntr, Double_t weight, Int_t is, Int_t secondParentId);

  /// Get next particle for tracking from the stack.
  /// Declared in TVirtualMCStack
  /// Returns a pointer to the TParticle of the track
  /// \param iTrack index of popped track (return)
  virtual TParticle* PopNextTrack(Int_t& iTrack);

  /// Get primary particle by index for tracking from stack
  /// Declared in TVirtualMCStack
  /// Returns a pointer to the TParticle of the track
  /// \param iPrim index of primary particle
  virtual TParticle* PopPrimaryForTracking(Int_t iPrim);

  /// Set the current track number
  /// Declared in TVirtualMCStack
  /// \param iTrack track number
  virtual void SetCurrentTrack(Int_t iTrack)
  {
    mIndexOfCurrentTrack = iTrack;
  }

  /// Get total number of tracks
  /// Declared in TVirtualMCStack
  virtual Int_t GetNtrack() const
  {
    return mNumberOfEntriesInParticles;
  }

  /// Get number of primary tracks
  /// Declared in TVirtualMCStack
  virtual Int_t GetNprimary() const
  {
    return mNumberOfPrimaryParticles;
  }

  /// Get the current track's particle
  /// Declared in TVirtualMCStack
  virtual TParticle* GetCurrentTrack() const;

  /// Get the number of the current track
  /// Declared in TVirtualMCStack
  virtual Int_t GetCurrentTrackNumber() const
  {
    return mIndexOfCurrentTrack;
  }

  /// Get the track number of the parent of the current track
  /// Declared in TVirtualMCStack
  virtual Int_t GetCurrentParentTrackNumber() const;

  /// Add a TParticle to the mParticles array
  virtual void AddParticle(TParticle* part);

  /// Fill the MCTrack output array, applying filter criteria
  virtual void FillTrackArray();

  /// Update the track index in the MCTracks and MCPoints
  virtual void UpdateTrackIndex(TRefArray* detArray = 0);

  /// Resets arrays and stack and deletes particles and tracks
  virtual void Reset();

  /// Register the MCTrack array to the Root Manager
  virtual void Register();

  /// Output to screen
  /// \param iVerbose: 0=events summary, 1=track info

  virtual void Print(Int_t iVerbose = 0) const;

  /// Modifiers
  void StoreSecondaries(Bool_t choice = kTRUE)
  {
    mStoreSecondaries = choice;
  }
  void SetMinPoints(Int_t min)
  {
    mMinPoints = min;
  }
  void SetEnergyCut(Double_t eMin)
  {
    mEnergyCut = eMin;
  }
  void StoreMothers(Bool_t choice = kTRUE)
  {
    mStoreMothers = choice;
  }

  /// Increment number of points for the current track in a given detector
  /// \param iDet  Detector unique identifier

  void AddPoint(DetectorId iDet);

  /// Increment number of points for an arbitrary track in a given detector
  /// \param iDet    Detector unique identifier
  /// \param iTrack  Track number
  void AddPoint(DetectorId iDet, Int_t iTrack);

  /// Accessors
  TParticle* GetParticle(Int_t trackId) const;
  TClonesArray* GetListOmParticles()
  {
    return mParticles;
  }

  /// Clone for worker (used in MT mode only)
  virtual FairGenericStack* CloneStack() const;

private:
  FairLogger* mLogger;

  /// STL stack (FILO) used to handle the TParticles for tracking
  std::stack<TParticle*> mStack; //!

  /// Array of TParticles (contains all TParticles put into or created
  /// by the transport
  TClonesArray* mParticles; //!

  /// Array of FairMCTracks containg the tracks written to the output
  TClonesArray* mTracks;

  /// STL map from particle index to storage flag
  std::map<Int_t, Bool_t> mStoreMap;                //!
  std::map<Int_t, Bool_t>::iterator mStoreIterator; //!

  /// STL map from particle index to track index
  std::map<Int_t, Int_t> mIndexMap;                //!
  std::map<Int_t, Int_t>::iterator mIndexIterator; //!

  /// STL map from track index and detector ID to number of MCPoints
  std::map<std::pair<Int_t, Int_t>, Int_t> mPointsMap; //!

  /// Some indices and counters
  Int_t mIndexOfCurrentTrack;        //! Index of current track
  Int_t mNumberOfPrimaryParticles;   //! Number of primary particles
  Int_t mNumberOfEntriesInParticles; //! Number of entries in mParticles
  Int_t mNumberOfEntriesInTracks;    //! Number of entries in mTracks
  Int_t mIndex;                      //! Used for merging

  /// Variables defining the criteria for output selection
  Bool_t mStoreMothers;
  Bool_t mStoreSecondaries;
  Int_t mMinPoints;
  Double32_t mEnergyCut;

  /// Mark tracks for output using selection criteria
  void SelectTracks();

  Stack(const Stack&);
  Stack& operator=(const Stack&);

  ClassDef(Stack, 1)
};
}
}

#endif
