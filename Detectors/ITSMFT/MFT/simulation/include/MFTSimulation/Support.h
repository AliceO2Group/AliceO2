/// \file Support.h
/// \brief Class describing geometry of one MFT half-disk support
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#ifndef ALICEO2_MFT_SUPPORT_H_
#define ALICEO2_MFT_SUPPORT_H_

#include "TNamed.h"
#include "TGeoVolume.h"

#include "FairLogger.h"

namespace AliceO2 {
namespace MFT {

class Support : public TNamed {
  
public:
  
  Support();
  
  virtual ~Support();
  
  TGeoVolumeAssembly* CreateVolume(Int_t half, Int_t disk);
  TGeoVolumeAssembly* CreatePCBs(Int_t half, Int_t disk);
  TGeoVolume* CreateSupport(Int_t half, Int_t disk);
  
protected:
  
  TGeoVolumeAssembly * fSupportVolume;
  Double_t fSupportThickness;
  Double_t fPCBThickness;

private:
  
  ClassDef(Support,1)
  
};

}
}

#endif

