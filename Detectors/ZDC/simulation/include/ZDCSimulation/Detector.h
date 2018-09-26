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
#include "ZDCSimulation/Hit.h"

class FairVolume;
class FairModule;

class TParticle;

namespace o2
{
namespace zdc
{
class Hit;


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

  /// Gets the produced collections
  std::vector<o2::zdc::Hit>* getHits(Int_t iColl) const
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

  o2::zdc::Hit* addHit(Int_t trackID, Int_t parentID, Int_t sFlag, Float_t primaryEnergy, Int_t detID, Int_t secID,
    Vector3D<float> pos, Vector3D<float>  mom, Float_t tof, Float_t *xImpact, Double_t energyloss, Int_t nphePMC, Int_t nphePMQ);

   private:
    /// copy constructor
    Detector(const Detector& rhs);

    void CreateAsideBeamLine();
    void CreateCsideBeamLine();
    void CreateMagnets();
    void CreateDetectors();

    // Define sensitive volumes
    void defineSensitiveVolumes();

    // Methods to calculate the light outpu
    void CalculateTableIndexes(int& ibeta, int& iangle, int& iradius);
    void ReadLightTable();


    Int_t mZDCdetectorID; //detector in ZDC
    Int_t mZDCsectorID; //tower in ZDC
    Int_t mPcMother; // track mother 0
    Int_t mCurrentTrackID;
    Float_t mTrackEta;
    Bool_t  mSecondaryFlag;
    Float_t mPrimaryEnergy;
    Float_t mXImpact[3];
    Float_t mTrackTOF;
    Float_t mTotDepEnergy;
    Float_t mTotLightPMC;
    Float_t mTotLightPMQ;
    Int_t   mMediumPMCid;
    Int_t   mMediumPMQid;
    o2::zdc::Hit* mCurrentHit;
    //
    /// Container for hit data
    std::vector<o2::zdc::Hit>* mHits;

    Float_t mLumiLength = 0; //TODO: make part of configurable params
    Float_t mTCLIAAPERTURE = 3.5; //TODO: make part of configurable params
    Float_t mTCLIAAPERTURENEG = 3.5; //TODO: make part of configurable params
	  Float_t mVCollSideCCentreY = 0.; //TODO: make part of configurable params

    static constexpr int ZNRADIUSBINS = 18;
    static constexpr int ZPRADIUSBINS = 28;
    static constexpr int ANGLEBINS = 90;

    float mLightTableZN[4][ZNRADIUSBINS][ANGLEBINS]={1.}; //!
    float mLightTableZP[4][ZPRADIUSBINS][ANGLEBINS]={1.}; //!

  template <typename Det>
  friend class o2::Base::DetImpl;
  ClassDefOverride(Detector, 1);
};

}
}

#endif
