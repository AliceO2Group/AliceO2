#ifndef ALICEO2_EMCAL_DETECTOR_H_
#define ALICEO2_EMCAL_DETECTOR_H_

#include "DetectorsBase/Detector.h"
#include "Rtypes.h"
#include "TVector3.h"

class FairVolume;
class TClonesArray;
namespace AliceO2 { namespace EMCAL { class Point; } }

namespace AliceO2 {
  namespace EMCAL {
    
    class Detector : public AliceO2::Base::Detector {
    public:
      Detector() = default;
      
      Detector(const char* Name, Bool_t Active);
      
      virtual ~Detector() = default;
      
      virtual void   Initialize() final;
      
      virtual Bool_t ProcessHits( FairVolume* v=0) final;
      
      Point *AddHit(Int_t shunt, Int_t trackID, Int_t parentID, Int_t primary, Double_t initialEnergy,
                    Int_t detID, Double_t *pos, Double_t *mom, Double_t time, Double_t length);
      
      virtual void   Register();
      
      virtual TClonesArray* GetCollection(Int_t iColl) const final;
      
      virtual void   Reset() final;
      
    protected:
      
      virtual void CreateMaterials() final;
      
    private:
    
      TClonesArray        *mPointCollection;            ///< Collection of EMCAL points
      
      ClassDef(Detector, 1)
    };
  }
}
#endif
