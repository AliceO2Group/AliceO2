// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_DETECTOR_H_
#define ALICEO2_EMCAL_DETECTOR_H_

#include "DetectorsBase/Detector.h"
#include "MathUtils/Cartesian3D.h"
#include "RStringView.h"
#include "Rtypes.h"

#include <set>
#include <vector>

class FairVolume;
class TClonesArray;

namespace o2
{
namespace EMCAL
{
class Hit;
class Geometry;

/// \struct Track hit
/// \brief Helper structure storing hits assigned to the same track within the event
///
/// Hits are handled additive in the EMCAL simulation, meaning all hits belonging to
/// the same track are added. This struct is used internally in the EMCAL detector class
/// setting up a search structure for hits belonging to a certain track
struct TrackHit {
  Int_t mTrackID; ///< track ID
  Hit* mHit;      ///< Associated EMCAL hit

  bool operator==(const TrackHit& other) const { return mTrackID == other.mTrackID; }
  bool operator<(const TrackHit& other) const { return mTrackID < other.mTrackID; }
};

///
/// \class Detector
/// \bief Detector class for the EMCAL detector
///
/// The detector class handles the implementation of the EMCAL detector
/// within the virtual Monte-Carlo framework and the simulation of the
/// EMCAL detector up to hit generation
class Detector : public o2::Base::Detector
{
 public:
  enum { ID_AIR = 0, ID_PB = 1, ID_SC = 2, ID_AL = 3, ID_STEEL = 4, ID_PAPER = 5 };

  ///
  /// Default constructor
  ///
  Detector() = default;

  ///
  /// Main constructor
  ///
  /// \param[in] name Name of the detector (EMCAL)
  /// \param[in] isActive Switch whether detector is active in simulation
  Detector(const char* name, Bool_t isActive);

  ///
  /// Destructor
  ///
  ~Detector() override = default;

  ///
  /// Initializing detector
  ///
  void Initialize() final;

  ///
  /// Processing hit creation in the EMCAL scintillator volume
  ///
  /// \param[in] v Current sensitive volume
  Bool_t ProcessHits(FairVolume* v = nullptr) final;

  ///
  /// Add EMCAL hit
  /// Internally adding hits coming from the same track
  ///
  /// \param[in] trackID Index of the track in the MC stack
  /// \param[in] parentID Index of the parent particle (entering the EMCAL) in the MC stack
  /// \param[in] primary Index of the primary particle in the MC stack
  /// \param[in] initialEnergy Energy of the particle entering the EMCAL
  /// \param[in] detID Index of the detector (cell) for which the hit is created
  /// \param[in] pos Position vector of the particle at the hit
  /// \param[in] mom Momentum vector of the particle at the hit
  /// \param[in] time Time of the hit
  /// \param[in] energyloss Energy deposit in EMCAL
  ///
  Hit* AddHit(Int_t trackID, Int_t parentID, Int_t primary, Double_t initialEnergy, Int_t detID,
              const Point3D<float>& pos, const Vector3D<float>& mom, Double_t time, Double_t energyloss);

  ///
  /// Register TClonesArray with hits
  ///
  void Register() override;

  ///
  /// Get access to the point collection
  /// \return TClonesArray with points
  ///
  TClonesArray* GetCollection(Int_t iColl) const final;

  ///
  /// Reset
  /// Clean point collection
  ///
  void Reset() final;

  ///
  /// Steps to be carried out at the end of the event
  /// For EMCAL cleaning the hit collection and the lookup table
  ///
  void EndOfEvent() final;

  ///
  /// Get the EMCAL geometry desciption
  /// Will be created the first time the function is called
  /// \return Access to the EMCAL Geometry description
  ///
  Geometry* GetGeometry();

 protected:
  ///
  /// Creating detector materials for the EMCAL detector and space frame
  ///
  void CreateMaterials();

  void ConstructGeometry() override;

  ///
  /// Generate EMCAL envelop (mother volume of all supermodules)
  ///
  void CreateEmcalEnvelope();

  ///
  /// Generate tower geometry
  ///
  void CreateShiskebabGeometry();

  ///
  /// Generate super module geometry
  ///
  void CreateSupermoduleGeometry(const std::string_view mother = "XEN1");

  ///
  /// Generate module geometry (2x2 towers)
  ///
  void CreateEmcalModuleGeometry(const std::string_view mother = "SMOD", const std::string_view child = "EMOD");

  ///
  /// Generate aluminium plates geometry
  ///
  void CreateAlFrontPlate(const std::string_view mother = "EMOD", const std::string_view child = "ALFP");

  ///
  /// Calculate the amount of light seen by the APD for a given track segment (charged particles only)
  /// Calculation done according to Bricks law
  ///
  /// \param[in] energydeposit Energy deposited by a charged particle in the track segment
  /// \param[in] tracklength Length of the track segment
  /// \param[in] charge Track charge (in units of elementary charge)
  ///
  Double_t CalculateLightYield(Double_t energydeposit, Double_t tracklength, Int_t charge) const;

 private:
  Int_t mBirkC0;
  Double_t mBirkC1;
  Double_t mBirkC2;

  std::set<TrackHit>
    mEventHits; ///< Set of hits within the event, used for fast lookup of hits connected to a primary particle
  TClonesArray* mPointCollection; ///< Collection of EMCAL points
  Geometry* mGeometry;            ///< Geometry pointer

  Double_t mSampleWidth; //!<! sample width = double(g->GetECPbRadThick()+g->GetECScintThick());
  Double_t mSmodPar0;    //!<! x size of super module
  Double_t mSmodPar1;    //!<! y size of super module
  Double_t mSmodPar2;    //!<! z size of super module
  Double_t mInnerEdge;   //!<! Inner edge of DCAL super module
  Double_t mParEMOD[5];  //!<! parameters of EMCAL module (TRD1,2)

  ClassDefOverride(Detector, 1)
};
}
}
#endif
