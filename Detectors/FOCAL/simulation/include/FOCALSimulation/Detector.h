// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FOCAL_DETECTOR_H_
#define ALICEO2_FOCAL_DETECTOR_H_

#include <vector>

#include "DetectorsBase/Detector.h"
#include "FOCALBase/Hit.h"

class FairVolume;

namespace o2::focal
{

/// \struct Parent
/// \brief Information about superparent (particle entering any FOCAL volume)
/// \ingroup FOCALsimulation
struct Parent {
  int mPDG;                ///< PDG code
  double mEnergy;          ///< Total energy
  bool mHasTrackReference; ///< Flag indicating whether parent has a track reference
};

/// \class Detector
/// \brief FOCAL detector simulation
/// \ingroup FOCALsimulation
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since June 6, 2024
class Detector : public o2::base::DetImpl<Detector>
{
 public:
  enum MediumType_t { ID_TUNGSTEN = 0,
                      ID_SILICON = 1,
                      ID_G10 = 2,
                      ID_COPPER = 3,
                      ID_STEEL = 4,
                      ID_ALLOY = 5,
                      ID_CERAMIC = 6,
                      ID_PB = 7,
                      ID_SC = 8,
                      ID_SIINSENS = 9,
                      ID_VAC = 10,
                      ID_AIR = 11 };
  /// \brief Dummy constructor
  Detector() = default;

  /// \brief Destructor
  ~Detector() override;

  /// \brief Initializing detector
  void InitializeO2Detector() override;

  /// \brief Processing hit creation in the FOCAL sensitive volume
  /// \param v Current sensitive volume
  Bool_t ProcessHits(FairVolume* v = nullptr) final;

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
  /// For FOCAL cleaning the hit collection and the lookup table
  void EndOfEvent() final;

  /// \brief Begin primaray
  ///
  /// Caching current primary ID and set current parent ID to the
  /// current primary ID
  void BeginPrimary() override;

  /// \brief Finish current primary
  ///
  /// Reset caches for current primary, current parent and current cell
  void FinishPrimary() override;

 protected:
  /// \brief Creating detector materials for the FOCAL detector
  void CreateMaterials();

  void ConstructGeometry() override;

  /// \brief Add new superparent to the container
  /// \param trackID Track ID of the superparent
  /// \param pdg PDG code of the superparent
  /// \param energy Energy of the superparent
  Parent* AddSuperparent(int trackID, int pdg, double energy);

 private:
  /// \brief Copy constructor (used in MT)
  Detector(const Detector& rhs);

  std::vector<o2::focal::Hit>* mHits; ///< Container with hits

  std::unordered_map<int, int> mSuperParentsIndices; //!<! Super parent indices (track index - superparent index)
  std::unordered_map<int, Parent> mSuperParents;     //!<! Super parent kine info (superparent index - superparent object)
  Parent* mCurrentSuperparent;                       //!<! Pointer to the current superparent

  // Worker variables during hit creation
  Int_t mCurrentTrack;     //!<! Current track
  Int_t mCurrentPrimaryID; //!<! ID of the current primary
  Int_t mCurrentParentID;  //!<! ID of the current parent

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};
} // namespace o2::focal

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::focal::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif // ALICEO2_FOCAL_DETECTOR_H_