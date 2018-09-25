// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ZDC_DETECTOR_H_
#define ALICEO2_ZDC_DETECTOR_H_

#include <vector>                             // for vector
#include "Rtypes.h"                           // for Int_t, Double_t, Float_t, Bool_t, etc
#include "TGeoManager.h"                      // for gGeoManager, TGeoManager (ptr only)
#include "DetectorsBase/GeometryManager.h"    // for getSensID
#include "DetectorsBase/Detector.h"           // for Detector
#include "DetectorsCommonDataFormats/DetID.h" // for Detector
#include "ZDCBase/Geometry.h"

#include "SimulationDataFormat/BaseHits.h"

class FairVolume;
class FairModule;

class TParticle;

namespace o2
{
namespace zdc
{
using HitType = o2::BasicXYZEHit<float>;

class Detector : public o2::Base::DetImpl<Detector>
{
 public:
  enum ZDCMaterial {
    kWalloy = 1,
    kCuZn = 2,
    kSiO2pmc = 3,
    kSiO2pmq = 4,
    kPb = 5,
    kCu = 6,
    kFe = 7,
    kAl = 8,
    kGraphite = 9,
    kVoidNoField = 10,
    kVoidwField = 11,
    kAir = 12
  };

  Detector() = default;

  Detector(Bool_t active);

  ~Detector() override = default;

  void InitializeO2Detector() final;

  Bool_t ProcessHits(FairVolume* v = nullptr) final;

  void Register() override;

  std::vector<HitType>* getHits(int iColl) const
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

  void Reset() final;
  void EndOfEvent() final;


  void ConstructGeometry() final;
  void CreateMaterials();
  void addAlignableVolumes() const override {}

  HitType* AddHit(Int_t trackID, Int_t parentID, Int_t sFlag, Double_t primaryEnergy, Int_t& detID,
              Double_t& pos, Double_t& mom, Double_t tof, Double_t& xImpact, Double_t energyloss, Int_t nphe);

   private:
    /// copy constructor
    Detector(const Detector& rhs);
    void CreateAsideBeamLine();
    void CreateCsideBeamLine();
    void CreateSupports();
    void CreateMagnets();
    void CreateDetectors();
    /// Define sensitive volumes
    void defineSensitiveVolumes();

    void CalculateTableIndexes(int& ibeta, int& iangle, int& iradius);

    Int_t mZDCdetID[2]; //detector|tower in ZDC
    Int_t mPcMother; // track mother 0
    Int_t mCurrentTrackID;
    HitType* mCurrentHit;
    Float_t mTrackEta;
    Int_t   mSecondaryFlag;
    Float_t mPrimaryEnergy;
    Float_t mXImpact[3];
    Float_t mTrackTOF;
    Float_t mTotDepEnergy;
    Float_t mTotLight[2]; //[0]PMC [1]sumPMQi
    //
    Float_t mLumiLength = 0; //TODO: make part of configurable params
    Float_t mTCLIAAPERTURE = 3.5; //TODO: make part of configurable params
    Float_t mTCLIAAPERTURENEG = 3.5; //TODO: make part of configurable params
	  Float_t mVCollSideCCentreY = 0.; //TODO: make part of configurable params

    /// container for data points
    std::vector<HitType>* mHits; //!

  template <typename Det>
  friend class o2::Base::DetImpl;
  ClassDefOverride(Detector, 1);
};
}
}
#endif
