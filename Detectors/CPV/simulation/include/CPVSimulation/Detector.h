// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CPV_DETECTOR_H_
#define ALICEO2_CPV_DETECTOR_H_

#include "DetectorsBase/Detector.h"
#include "MathUtils/Cartesian3D.h"
#include "CPVBase/Hit.h"
#include "RStringView.h"
#include "Rtypes.h"

#include <map>
#include <vector>

class FairVolume;

namespace o2
{
namespace cpv
{
class Hit;
class Geometry;
class GeometryParams;

///
/// \class Detector
/// \brief Detector class for the CPV detector
///
/// The detector class handles the implementation of the CPV detector
/// within the virtual Monte-Carlo framework and the simulation of the
/// CPV detector up to hit generation
class Detector : public o2::base::DetImpl<Detector>
{
 public:
  // CPV materials/media
  enum {
    ID_AIR = 1,
    ID_TEXTOLIT = 2,
    ID_CU = 3,
    ID_AR = 4,
    ID_AL = 5,
    ID_FE = 6
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
  /// Processing hit creation in the CPV crystalls
  ///
  /// \param[in] v Current sensitive volume
  Bool_t ProcessHits(FairVolume* v = nullptr) final;

  ///
  /// Add CPV hit
  /// Internally adding hits coming from the same track
  ///
  /// \param[in] trackID Index of the track in the MC stack
  /// \param[in] detID Index of the detector (pad) for which the hit is created
  /// \param[in] pos Position vector of the particle entered CPV active layer
  /// \param[in] time Time of the hit
  /// \param[in] qdep charge deposited in this pad
  ///
  void AddHit(int trackID, int detID, const Point3D<float>& pos, Double_t time, Double_t qdep);

  ///
  /// Register vector with hits
  ///
  void Register() override;

  ///
  /// Get access to the hits
  ///
  std::vector<Hit>* getHits(Int_t iColl) const
  {
    if (iColl == 0) {
      return mHits;
    } else {
      return nullptr;
    }
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
  /// For CPV cleaning the hit collection and the lookup table
  ///
  void EndOfEvent() final;

  ///
  /// Specifies CPV modules as alignable volumes
  ///
  void addAlignableVolumes() const override;

  ///
  /// Get the CPV geometry desciption
  /// Will be created the first time the function is called
  /// \return Access to the CPV Geometry description
  ///
  Geometry* GetGeometry();

 protected:
  ///
  /// Creating detector materials for the CPV detector and space frame
  ///
  void CreateMaterials();

  ///
  /// Creating CPV description for Geant
  ///
  void ConstructGeometry() override;

  //
  // Calculate the amplitude in one CPV pad using the
  //
  double PadResponseFunction(float qhit, float zhit, float xhit);

  double CPVCumulPadResponse(double x, double y);

 private:
  /// copy constructor (used in MT)
  Detector(const Detector& rhs);
  Detector& operator=(const Detector&);

  /// Define the sensitive volumes of the geometry
  void defineSensitiveVolumes();

  // Geometry parameters
  Bool_t mActiveModule[6]; // list of modules to create

  // Simulation
  Geometry* mGeom;         //!
  std::vector<Hit>* mHits; //! Collection of CPV hits

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};
} // namespace cpv
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::cpv::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif // Detector.h
