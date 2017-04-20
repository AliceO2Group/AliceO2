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
  ~Flex() override;
  TGeoVolumeAssembly*  makeFlex(Int_t nbsensors, Double_t length);
  void makeElectricComponents(TGeoVolumeAssembly*  flex, Int_t nbsensors, Double_t length, Double_t zvarnish);

private:

  TGeoVolume*  makeLines(Int_t nbsensors, Double_t length, Double_t width, Double_t thickness);
  TGeoVolume*  makeAGNDandDGND(Double_t length, Double_t width, Double_t thickness);
  TGeoVolume*  makeKapton(Double_t length, Double_t width, Double_t thickness);
  TGeoVolume*  makeVarnish(Double_t length, Double_t width, Double_t thickness, Int_t iflag);
  TGeoVolumeAssembly*  makeElectricComponent(Double_t dx, Double_t dy, Double_t dz, Int_t iflag);

  Double_t *mFlexOrigin;
  LadderSegmentation * mLadderSeg;

  ClassDefOverride(Flex,1)

};

}
}

#endif
