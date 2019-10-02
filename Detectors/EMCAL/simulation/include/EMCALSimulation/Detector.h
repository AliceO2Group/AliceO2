// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_DETECTOR_H_
#define ALICEO2_EMCAL_DETECTOR_H_

#include "DetectorsBase/Detector.h"
#include "EMCALBase/Hit.h"
#include "MathUtils/Cartesian3D.h"
#include "RStringView.h"
#include "Rtypes.h"
#include <vector>

class FairVolume;
class TClonesArray;

namespace o2
{
namespace emcal
{
class Hit;
class Geometry;

///
/// \class Detector
/// \bief Detector class for the EMCAL detector
///
/// The detector class handles the implementation of the EMCAL detector
/// within the virtual Monte-Carlo framework and the simulation of the
/// EMCAL detector up to hit generation
class Detector : public o2::base::DetImpl<Detector>
{
 public:
  enum { ID_AIR = 0,
         ID_PB = 1,
         ID_SC = 2,
         ID_AL = 3,
         ID_STEEL = 4,
         ID_PAPER = 5 };

  ///
  /// Default constructor
  ///
  Detector() = default;

  /// \brief Main constructor
  ///
  /// \param[in] name Name of the detector (EMC)
  /// \param[in] isActive Switch whether detector is active in simulation
  Detector(Bool_t isActive);

  /// \brief Destructor
  ~Detector() override;

  /// \brief Initializing detector
  void InitializeO2Detector() override;

  ///
  /// Processing hit creation in the EMCAL scintillator volume
  ///
  /// \param[in] v Current sensitive volume
  Bool_t ProcessHits(FairVolume* v = nullptr) final;

  /// \brief Add EMCAL hit
  /// \param[in] trackID Index of the track in the MC stack
  /// \param[in] primary Index of the primary particle in the MC stack
  /// \param[in] initialEnergy Energy of the particle entering the EMCAL
  /// \param[in] detID Index of the detector (cell) for which the hit is created
  /// \param[in] pos Position vector of the particle at the hit
  /// \param[in] mom Momentum vector of the particle at the hit
  /// \param[in] time Time of the hit
  /// \param[in] energyloss Energy deposit in EMCAL
  /// \return Pointer to the current hit
  ///
  /// Internally adding hits coming from the same track
  Hit* AddHit(Int_t trackID, Int_t primary, Double_t initialEnergy, Int_t detID,
              const Point3D<float>& pos, const Vector3D<float>& mom, Double_t time, Double_t energyloss);

  /// \brief register container with hits
  void Register() override;

  /// \brief Get access to the hits
  /// \return Hit collection
  std::vector<Hit>* getHits(Int_t iColl) const
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

  /// \brief Clean point collection
  void Reset() final;

  /// \brief Steps to be carried out at the end of the event
  ///
  /// For EMCAL cleaning the hit collection and the lookup table
  void EndOfEvent() final;

  /// \brief Get the EMCAL geometry desciption
  /// \return Access to the EMCAL Geometry description
  ///
  /// Will be created the first time the function is called
  Geometry* GetGeometry();

  /// \brief Begin primaray
  ///
  /// Caching current primary ID and set current parent ID to the
  /// current primary ID
  void BeginPrimary() override;

  /// \brief Start new track
  ///
  /// Check whether track is produced outside EMCAL, in this case it
  /// serves as new parent track. In addition cache current track ID
  void PreTrack() override;

  /// \brief Finish current primary
  ///
  /// Reset caches for current primary, current parent and current cell
  void FinishPrimary() override;

 protected:
  /// \brief Creating detector materials for the EMCAL detector and space frame
  void CreateMaterials();

  void ConstructGeometry() override;

  /// \brief Generate EMCAL envelop (mother volume of all supermodules)
  void CreateEmcalEnvelope();

  /// \brief Generate tower geometry
  void CreateShiskebabGeometry();

  /// \brief Generate super module geometry
  void CreateSupermoduleGeometry(const std::string_view mother = "XEN1");

  /// \brief Generate module geometry (2x2 towers)
  void CreateEmcalModuleGeometry(const std::string_view mother = "SMOD", const std::string_view child = "EMOD");

  /// \brief Generate aluminium plates geometry
  void CreateAlFrontPlate(const std::string_view mother = "EMOD", const std::string_view child = "ALFP");

  /// Calculate the amount of light seen by the APD for a given track segment (charged particles only) according to Bricks law
  /// \param[in] energydeposit Energy deposited by a charged particle in the track segment
  /// \param[in] tracklength Length of the track segment
  /// \param[in] charge Track charge (in units of elementary charge)
  /// \return Light yield
  Double_t CalculateLightYield(Double_t energydeposit, Double_t tracklength, Int_t charge) const;

  /// \brief Try to find hit with same cell and parent track ID
  /// \param cellID ID of the tower
  /// \param parentID ID of the parent track
  Hit* FindHit(Int_t cellID, Int_t parentID);

 private:
  /// \brief Copy constructor (used in MT)
  Detector(const Detector& rhs);

  Int_t mBirkC0;    ///< Birk parameter C0
  Double_t mBirkC1; ///< Birk parameter C1
  Double_t mBirkC2; ///< Birk parameter C2

  std::vector<Hit>* mHits; //!<! Collection of EMCAL hits
  Geometry* mGeometry;     //!<! Geometry pointer

  // Worker variables during hit creation
  Int_t mCurrentPrimaryID;   //!<! ID of the current primary
  Int_t mCurrentParentID;    //!<! ID of the current parent track (must be created outside EMCAL)
  Float_t mParentEnergy;     //!<! Initial energy of the parent track
  Bool_t mParentHasTrackRef; //!<! Flag whether parent track has track reference

  Double_t mSampleWidth; //!<! sample width = double(g->GetECPbRadThick()+g->GetECScintThick());
  Double_t mSmodPar0;    //!<! x size of super module
  Double_t mSmodPar1;    //!<! y size of super module
  Double_t mSmodPar2;    //!<! z size of super module
  Double_t mInnerEdge;   //!<! Inner edge of DCAL super module
  Double_t mParEMOD[5];  //!<! parameters of EMCAL module (TRD1,2)

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};
} // namespace emcal
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::emcal::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif
