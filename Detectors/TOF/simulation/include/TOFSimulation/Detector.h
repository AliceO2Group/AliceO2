// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TOF_DETECTOR_H_
#define ALICEO2_TOF_DETECTOR_H_

#include "DetectorsBase/Detector.h"
#include "TOFBase/Geo.h"

#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/PositionVector3D.h"

template <typename T>
using Point3D = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Vector3D = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;

class FairVolume;
class TClonesArray;
namespace o2 { namespace TOF { class Hit; } }

namespace o2 {
  namespace TOF {
    
    class Detector : public o2::Base::Detector {
    public:
      
      enum TOFMaterial{
	kAir=1,
	kNomex=2,
	kG10=3,
	kFiberGlass=4,
	kAlFrame=5,
	kHoneycomb=6,
	kFre=7,
	kCuS=8,
	kGlass=9,
	kWater=10,
	kCable=11,
	kCableTubes=12,
	kCopper=13,
	kPlastic=14,
	kCrates=15,
	kHoneyHoles=16
      };

      Detector() = default;
      
      Detector(const char* Name, Bool_t Active);
      
      ~Detector() override = default;
      
      void   Initialize() final;
      
      Bool_t ProcessHits( FairVolume* v=nullptr) final;
      
      Hit *AddHit(Int_t shunt, Int_t trackID, Int_t parentID, Int_t primary, Double_t initialEnergy,
                    Int_t detID, const Point3D<float> &pos, const Vector3D<float> &mom, Double_t time, Double_t length);
      
      void   Register() override;
      
      TClonesArray* GetCollection(Int_t iColl) const final;
      
      void   Reset() final;
      
      virtual void CreateMaterials() final;
      virtual void ConstructGeometry() final;

      void SetTOFholes(Bool_t flag=kTRUE) {mTOFHoles = flag;}

    protected:
      
      virtual void ConstructSuperModule(Int_t imodule) final;
      virtual void DefineGeometry(Float_t xtof, Float_t ytof, Float_t zlenA) final;
      virtual void MaterialMixer(Float_t * p, const Float_t * const a,
					   const Float_t * const m, Int_t n) const final;
    private:

      void CreateModules(Float_t xtof,  Float_t ytof, Float_t zlenA,
				   Float_t xFLT,  Float_t yFLT, Float_t zFLTA) const;  
      void MakeStripsInModules(Float_t ytof, Float_t zlenA) const;
      void CreateModuleCovers(Float_t xtof, Float_t zlenA) const;
      void CreateBackZone(Float_t xtof, Float_t ytof, Float_t zlenA) const;

      void MakeModulesInBTOFvolumes(Float_t ytof, Float_t zlenA) const;

      void AddAlignableVolumes() const;

      Int_t mTOFSectors[o2::TOF::Geo::NSECTORS];
      Bool_t mTOFHoles; // flag to allow for holes in front of the PHOS

      ClassDefOverride(Detector, 1)
    };
  }
}
#endif
