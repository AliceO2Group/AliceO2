// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PHOS_DETECTOR_H_
#define ALICEO2_PHOS_DETECTOR_H_

#include "DetectorsBase/Detector.h"
#include "MathUtils/Cartesian3D.h"
#include "PHOSBase/Hit.h"
#include "RStringView.h"
#include "Rtypes.h"

#include <map>
#include <vector>

class FairVolume;

namespace o2
{
namespace phos
{
class Hit;
class Geometry;
class GeometryParams;

///
/// \class Detector
/// \brief Detector class for the PHOS detector
///
/// The detector class handles the implementation of the PHOS detector
/// within the virtual Monte-Carlo framework and the simulation of the
/// PHOS detector up to hit generation
class Detector : public o2::base::DetImpl<Detector>
{
 public:
  // PHOS materials/media
  enum {
    ID_PWO = 1,
    ID_AL = 2,
    ID_TYVEK = 3,
    ID_POLYFOAM = 4,
    ID_APD = 5,
    ID_THERMOINS = 6,
    ID_TEXTOLIT = 7,
    ID_PRINTCIRC = 8,
    ID_FE = 9,
    ID_FIBERGLASS = 10,
    ID_CABLES = 11,
    ID_AIR = 12
  };

  ///
  /// Default constructor
  ///
  Detector() = default;

  ///
  /// Main constructor
  ///
  /// \param[in] isActive Switch whether detector is active in simulation
  Detector(Bool_t isActive);

  ///
  /// Destructor
  ///
  ~Detector() override;

  ///
  /// Initializing detector
  ///
  void InitializeO2Detector() final;

  ///
  /// Processing hit creation in the PHOS crystalls
  ///
  /// \param[in] v Current sensitive volume
  Bool_t ProcessHits(FairVolume* v = nullptr) final;

  ///
  /// Add PHOS hit
  /// Internally adding hits coming from the same track
  ///
  /// \param[in] trackID Index of the track in the MC stack first entered PHOS
  /// \param[in] detID Index of the detector (cell) for which the hit is created
  /// \param[in] pos Position vector of the particle first entered PHOS
  /// \param[in] mom Momentum vector of the particle first entered PHOS
  /// \param[in] totE Total energy of the particle entered PHOS
  /// \param[in] time Time of the hit
  /// \param[in] energyloss Energy deposited in this step
  ///
  Hit* AddHit(Int_t trackID, Int_t detID, const Point3D<float>& pos, const Vector3D<float>& mom, Double_t totE,
              Double_t time, Double_t eLoss);

  ///
  /// Register vector with hits
  ///
  void Register() override;

  ///
  /// Get access to the hits
  ///
  std::vector<Hit>* getHits(Int_t iColl) const
  {
    if (iColl == 0)
      return mHits;
    else
      return nullptr;
  }

  ///
  /// Reset
  /// Clean Hits collection
  ///
  void Reset() final;

  /// Sort final hist
  void FinishEvent() final;

  ///
  /// Steps to be carried out at the end of the event
  /// For PHOS cleaning the hit collection and the lookup table
  ///
  void EndOfEvent() final;

  ///
  /// Get the PHOS geometry desciption
  /// Will be created the first time the function is called
  /// \return Access to the PHOS Geometry description
  ///
  Geometry* GetGeometry();

 protected:
  ///
  /// Creating detector materials for the PHOS detector and space frame
  ///
  void CreateMaterials();

  ///
  /// Creating PHOS description for Geant
  ///
  void ConstructGeometry() override;

  /// Creating PHOS/support description for Geant
  void ConstructSupportGeometry();

  /// Creating PHOS/calorimeter part description for Geant
  void ConstructEMCGeometry();

  /// Creating PHOS/CPV description for Geant
  void ConstructCPVGeometry();

  /*

    ///
    /// TODO: Calculate the amount of light seen by the APD for a given track segment (charged particles only)
    /// Calculation done according to Bricks law
    ///
    /// \param[in] energydeposit Energy deposited by a charged particle in the track segment
    /// \param[in] tracklength Length of the track segment
    /// \param[in] charge Track charge (in units of elementary charge)
    ///
    Double_t CalculateLightYield(Double_t energydeposit, Double_t tracklength, Int_t charge) const;
  */
 private:
  /// copy constructor (used in MT)
  Detector(const Detector& rhs);
  Detector& operator=(const Detector&);

  /// Define the sensitive volumes of the geometry
  void defineSensitiveVolumes();

  // Geometry parameters
  Bool_t mCreateHalfMod;   // Should we create  1/2 filled module
  Bool_t mActiveModule[6]; // list of modules to create

  // Simulation
  Geometry* mGeom;                  //!
  std::map<int, int> mSuperParents; //! map of current tracks to SuperParents: entered PHOS active volumes particles
  std::vector<Hit>* mHits;          //! Collection of PHOS hits
  Int_t mCurrentTrackID;            //! current track Id
  Int_t mCurrentCellID;             //! current cell Id
  Int_t mCurentSuperParent;         //! current SuperParent ID: particle entered PHOS
  Hit* mCurrentHit;                 //! current Hit

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};
} // namespace phos
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::phos::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif // Detector.h
