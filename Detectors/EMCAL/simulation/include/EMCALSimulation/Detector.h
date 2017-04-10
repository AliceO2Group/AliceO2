#ifndef ALICEO2_EMCAL_DETECTOR_H_
#define ALICEO2_EMCAL_DETECTOR_H_

#include "DetectorsBase/Detector.h"
#include "Rtypes.h"

#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/PositionVector3D.h"

template <typename T>
using Point3D = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Vector3D = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;

class FairVolume;
class TClonesArray;
namespace o2 { namespace EMCAL { class Hit; } }

namespace o2 {
  namespace EMCAL {
    
    class Detector : public o2::Base::Detector {
    public:
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
      
    protected:
      
      virtual void CreateMaterials() final;
      
    private:
    
      TClonesArray        *mPointCollection;            ///< Collection of EMCAL points
      
      ClassDefOverride(Detector, 1)
    };
  }
}
#endif
