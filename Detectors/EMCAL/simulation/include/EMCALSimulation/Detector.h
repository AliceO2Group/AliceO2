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

#ifndef ALICEO2_EMCAL_DETECTOR_H_
#define ALICEO2_EMCAL_DETECTOR_H_

#include "DetectorsBase/Detector.h"
#include "EMCALBase/Hit.h"
#include "EMCALBase/GeometryBase.h"
#include "MathUtils/Cartesian.h"
#include "RStringView.h"
#include "Rtypes.h"
#include <vector>
#include <unordered_map>

class FairVolume;
class TClonesArray;
class TH2;

namespace o2
{
namespace emcal
{
class Hit;
class Geometry;

/// \struct Parent
/// \brief Information about superparent (particle entering EMCAL)
/// \ingroup EMCALsimulation
struct Parent {
  int mPDG;                ///< PDG code
  double mEnergy;          ///< Total energy
  bool mHasTrackReference; ///< Flag indicating whether parent has a track reference
};

/// \class Detector
/// \brief Detector simulation class for the EMCAL detector
/// \ingroup EMCALsimulation
///
/// The detector class handles the implementation of the EMCAL detector
/// within the virtual Monte-Carlo framework and the simulation of the
/// EMCAL detector up to hit generation
class Detector : public o2::base::DetImpl<Detector>
{
 public:
  enum MediumType_t { ID_AIR = 0,
                      ID_PB = 1,
                      ID_SC = 2,
                      ID_AL = 3,
                      ID_STEEL = 4,
                      ID_PAPER = 5 };

  /// \brief Default constructor
  Detector() = default;

  /// \brief Main constructor
  ///
  /// \param isActive Switch whether detector is active in simulation
  Detector(Bool_t isActive);

  /// \brief Destructor
  ~Detector() override;

  /// \brief Initializing detector
  void InitializeO2Detector() override;

  /// \brief Processing hit creation in the EMCAL scintillator volume
  /// \param v Current sensitive volume
  Bool_t ProcessHits(FairVolume* v = nullptr) final;

  /// \brief Add EMCAL hit
  /// \param trackID Index of the track in the MC stack
  /// \param primary Index of the primary particle in the MC stack
  /// \param initialEnergy Energy of the particle entering the EMCAL
  /// \param detID Index of the detector (cell) for which the hit is created
  /// \param pos Position vector of the particle at the hit
  /// \param mom Momentum vector of the particle at the hit
  /// \param time Time of the hit
  /// \param energyloss Energy deposit in EMCAL
  /// \return Pointer to the current hit
  ///
  /// Internally adding hits coming from the same track
  Hit* AddHit(Int_t trackID, Int_t primary, Double_t initialEnergy, Int_t detID,
              const math_utils::Point3D<float>& pos, const math_utils::Vector3D<float>& mom, Double_t time, Double_t energyloss);

  Parent* AddSuperparent(Int_t trackID, Int_t pdg, Double_t energy);

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

  /// \brief Create new EMCAL volume and add it to the list of sensitive volumes
  /// \param name Name of the volume
  /// \param shape Volume shape type
  /// \param mediumID ID of the medium
  /// \param shapeParams Shape parameters
  /// \param nparams Number of shape parameters
  /// \return ID of the volume
  ///
  /// Should be called for all EMCAL volumes. Internally calls TVirtualMC::Gsvolu(...). Gsvolu
  /// should not be used directly as this function adds the volume to the list of sensitive
  /// volumes. Making all volumes sensitive is essential in order to not have broken decay
  /// chains in the cache due to missing stepping in volumes. This is relevant for the detector
  /// itself, not the space frame.
  int CreateEMCALVolume(const std::string_view name, const std::string_view shape, MediumType_t mediumID, Double_t* shapeParams, Int_t nparams);

  /// \brief Calculate the amount of light seen by the APD for a given track segment (charged particles only) according to Bricks law
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

  std::vector<std::string> mSensitive;                      //!<! List of sensitive volumes
  std::unordered_map<int, EMCALSMType> mSMVolumeID;         //!<! map of EMCAL supermodule volume IDs
  std::unordered_map<std::string, EMCALSMType> mSMVolNames; //!<! map of EMCAL supermodule names
  Int_t mVolumeIDScintillator;                              //!<! Volume ID of the scintillator volume
  std::vector<Hit>* mHits;                                  //!<! Collection of EMCAL hits
  Geometry* mGeometry;                                      //!<! Geometry pointer
  std::unordered_map<int, int> mSuperParentsIndices;        //!<! Super parent indices (track index - superparent index)
  std::unordered_map<int, Parent> mSuperParents;            //!<! Super parent kine info (superparent index - superparent object)
  Parent* mCurrentSuperparent;                              //!<! Pointer to the current superparent

  // Worker variables during hit creation
  Int_t mCurrentTrack;     //!<! Current track
  Int_t mCurrentPrimaryID; //!<! ID of the current primary
  Int_t mCurrentParentID;  //!<! ID of the current parent

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
