// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TOF_DETECTOR_H_
#define ALICEO2_TOF_DETECTOR_H_

#include "DetectorsBase/Detector.h"
#include "TOFBase/Geo.h"

#include "SimulationDataFormat/BaseHits.h"

class FairVolume;
class TClonesArray;

namespace o2
{
namespace tof
{
using HitType = o2::BasicXYZEHit<float>;

class Detector : public o2::Base::DetImpl<Detector>
{
 public:
  enum TOFMaterial {
    kAir = 1,
    kNomex = 2,
    kG10 = 3,
    kFiberGlass = 4,
    kAlFrame = 5,
    kHoneycomb = 6,
    kFre = 7,
    kCuS = 8,
    kGlass = 9,
    kWater = 10,
    kCable = 11,
    kCableTubes = 12,
    kCopper = 13,
    kPlastic = 14,
    kCrates = 15,
    kHoneyHoles = 16
  };

  Detector() = default;

  Detector(Bool_t active);

  ~Detector() override = default;

  void Initialize() final;

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

  void CreateMaterials();
  void ConstructGeometry() final;

  void setTOFholes(Bool_t flag = kTRUE) { mTOFHoles = flag; }
 protected:
  HitType* addHit(Float_t x, Float_t y, Float_t z, Float_t time, Float_t energy, Int_t trackId, Int_t detId);
  virtual void DefineGeometry(Float_t xtof, Float_t ytof, Float_t zlenA) final;
  virtual void MaterialMixer(Float_t* p, const Float_t* const a, const Float_t* const m, Int_t n) const final;

 private:
  void createModules(Float_t xtof, Float_t ytof, Float_t zlenA, Float_t xFLT, Float_t yFLT, Float_t zFLTA) const;
  void makeStripsInModules(Float_t ytof, Float_t zlenA) const;
  void createModuleCovers(Float_t xtof, Float_t zlenA) const;
  void createBackZone(Float_t xtof, Float_t ytof, Float_t zlenA) const;
  void makeFrontEndElectronics(Float_t xtof) const;
  void makeFEACooling(Float_t xtof) const;
  void makeNinoMask(Float_t xtof) const;
  void makeSuperModuleCooling(Float_t xtof, Float_t ytof, Float_t zlenA) const;
  void makeSuperModuleServices(Float_t xtof, Float_t ytof, Float_t zlenA) const;
  void makeReadoutCrates(Float_t ytof) const;

  void makeModulesInBTOFvolumes(Float_t ytof, Float_t zlenA) const;
  void makeCoversInBTOFvolumes() const;
  void makeBackInBTOFvolumes(Float_t ytof) const;

  void addAlignableVolumes() const override;

  Int_t mEventNr; // event count
  Int_t mTOFSectors[o2::tof::Geo::NSECTORS];
  Bool_t mTOFHoles; // flag to allow for holes in front of the PHOS

  /// container for data points
  std::vector<HitType>* mHits; //!

  ClassDefOverride(Detector, 1);
};
}
}
#endif
