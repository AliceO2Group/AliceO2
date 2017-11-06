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

#include "FairGenericStack.h"
#include "SimulationDataFormat/MCTrack.h"
#include "DetectorsBase/DetID.h"

#include "Rtypes.h"
#include "TParticle.h"

#include <map>
#include <stack>
#include <utility>

class TClonesArray;

class TRefArray;

class FairLogger;

namespace o2 {

namespace Base {
class Detector;
}

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
    void PushTrack(Int_t toBeDone, Int_t parentID, Int_t pdgCode, Double_t px, Double_t py, Double_t pz,
                           Double_t e, Double_t vx, Double_t vy, Double_t vz, Double_t time, Double_t polx,
                           Double_t poly,
                           Double_t polz, TMCProcess proc, Int_t &ntr, Double_t weight, Int_t is) override;

    void PushTrack(Int_t toBeDone, Int_t parentID, Int_t pdgCode, Double_t px, Double_t py, Double_t pz,
                           Double_t e, Double_t vx, Double_t vy, Double_t vz, Double_t time, Double_t polx,
                           Double_t poly,
                           Double_t polz, TMCProcess proc, Int_t &ntr, Double_t weight, Int_t is, Int_t secondParentId) override;

    /// Get next particle for tracking from the stack.
    /// Declared in TVirtualMCStack
    /// Returns a pointer to the TParticle of the track
    /// \param iTrack index of popped track (return)
    TParticle *PopNextTrack(Int_t &iTrack) override;

    /// Get primary particle by index for tracking from stack
    /// Declared in TVirtualMCStack
    /// Returns a pointer to the TParticle of the track
    /// \param iPrim index of primary particle
    TParticle *PopPrimaryForTracking(Int_t iPrim) override;

    /// Set the current track number
    /// Declared in TVirtualMCStack
    /// \param iTrack track number
    void SetCurrentTrack(Int_t iTrack) override
    {
      mIndexOfCurrentTrack = iTrack;
    }

    /// Get total number of tracks
    /// Declared in TVirtualMCStack
    Int_t GetNtrack() const override
    {
      return mNumberOfEntriesInParticles;
    }

    /// Get number of primary tracks
    /// Declared in TVirtualMCStack
    Int_t GetNprimary() const override
    {
      return mNumberOfPrimaryParticles;
    }

    /// Get the current track's particle
    /// Declared in TVirtualMCStack
    TParticle *GetCurrentTrack() const override;

    /// Get the number of the current track
    /// Declared in TVirtualMCStack
    Int_t GetCurrentTrackNumber() const override
    {
      return mIndexOfCurrentTrack;
    }

    /// Get the track number of the parent of the current track
    /// Declared in TVirtualMCStack
    Int_t GetCurrentParentTrackNumber() const override;

    /// Fill the MCTrack output array, applying filter criteria
    void FillTrackArray() override;

    /// Update the track index in the MCTracks and data produced by detectors
    void UpdateTrackIndex(TRefArray *detArray = nullptr) override;

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
    void StoreSecondaries(Bool_t choice = kTRUE)
    {
      mStoreSecondaries = choice;
    }

    void setMinHits(Int_t min)
    {
      mMinHits = min;
    }

    void SetEnergyCut(Double_t eMin)
    {
      mEnergyCut = eMin;
    }

    void StoreMothers(Bool_t choice = kTRUE)
    {
      mStoreMothers = choice;
    }

    /// Increment number of hits for the current track in a given detector
    /// \param iDet  Detector unique identifier
    void addHit(int iDet);

    /// Increment number of hits for an arbitrary track in a given detector
    /// \param iDet    Detector unique identifier
    /// \param iTrack  Track number
    void addHit(int iDet, Int_t iTrack);

    /// Accessors
    TParticle *GetParticle(Int_t trackId) const;
    
    TClonesArray *GetListOfParticles() override
    {
      return mParticles;
    }

    /// Clone for worker (used in MT mode only)
    FairGenericStack *CloneStack() const override;

    // encode whether to store the particle or not as part of the TParticles (user) bits
    static void setStore(TParticle &p, bool store) {
      // using the first available user-bit (according to TObject docu)
      p.SetBit(BIT(14), store);
    }
    static bool isStore(const TParticle& p) {
      return p.TestBit(BIT(14));
    }
    // store integer encoding hits/detector information inside the TParticle
    static void setHitEncoding(TParticle &p, Int_t i) {
      p.SetUniqueID(i);
    }
    static Int_t getHitEncoding(const TParticle &p) {
      return p.GetUniqueID();
    }
    
  private:
    FairLogger *mLogger;

    /// STL stack (FILO) used to handle the TParticles for tracking
    std::stack<TParticle *> mStack; //!

    /// Array of TParticles (contains all TParticles put into or created
    /// by the transport
    TClonesArray* mParticles; //!
    
    /// vector of reducded tracks written to the output
    std::vector<o2::MCTrack>* mTracks;
    
    /// STL map from particle index to track index
    std::map<Int_t, Int_t> mIndexMap;                //!

    /// cache active O2 detectors
    std::vector<o2::Base::Detector *> mActiveDetectors; //!
    
    /// Some indices and counters
    Int_t mIndexOfCurrentTrack;        //! Index of current track
    Int_t mNumberOfPrimaryParticles;   //! Number of primary particles
    Int_t mNumberOfEntriesInParticles; //! Number of entries in mParticles
    Int_t mNumberOfEntriesInTracks;    //! Number of entries in mTracks
    Int_t mIndex;                      //! Used for merging

    /// Variables defining the criteria for output selection
    Bool_t mStoreMothers;
    Bool_t mStoreSecondaries;
    Int_t mMinHits;
    Double32_t mEnergyCut;

    /// Mark tracks for output using selection criteria
    /// returns true if all available tracks are selected
    /// returns false if some tracks are discarded
    bool selectTracks();
    
    Stack(const Stack &);

    Stack &operator=(const Stack &);

  ClassDefOverride(Stack, 1)
};


inline
TParticle *Stack::GetParticle(Int_t trackID) const
{
  return (TParticle *) mParticles->At(trackID);
}

}
}

#endif
