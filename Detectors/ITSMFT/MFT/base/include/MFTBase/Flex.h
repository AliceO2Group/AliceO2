/// \file Flex.h
/// \brief Flex (Printed Cabled Board) class for ALICE MFT upgrade
/// \author Franck Manso <franck.manso@cern.ch>

#ifndef ALICEO2_MFT_FLEX_H_
#define ALICEO2_MFT_FLEX_H_

#include "TNamed.h"

class TGeoVolume;
class TGeoVolumeAssembly;

namespace o2 {
namespace MFT {

class Flex : public TNamed {

public:

  Flex();
  Flex(LadderSegmentation *ladder);
  virtual ~Flex();
  TGeoVolumeAssembly*  MakeFlex(Int_t nbsensors, Double_t length);
  void Make_ElectricComponents(TGeoVolumeAssembly*  flex, Int_t nbsensors, Double_t length, Double_t zvarnish);

private:

  TGeoVolume*  Make_Lines(Int_t nbsensors, Double_t length, Double_t width, Double_t thickness);
  TGeoVolume*  Make_AGND_DGND(Double_t length, Double_t width, Double_t thickness);
  TGeoVolume*  Make_Kapton(Double_t length, Double_t width, Double_t thickness);
  TGeoVolume*  Make_Varnish(Double_t length, Double_t width, Double_t thickness, Int_t iflag);
  TGeoVolumeAssembly*  Make_ElectricComponent(Double_t dx, Double_t dy, Double_t dz, Int_t iflag);

  Double_t *mFlexOrigin;
  LadderSegmentation * mLadderSeg;

  ClassDef(Flex,1)

};

}
}

#endif
