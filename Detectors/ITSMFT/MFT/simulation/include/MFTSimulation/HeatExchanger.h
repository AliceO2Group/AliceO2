/// \file HeatExchanger.h
/// \brief MFT heat exchanger builder
/// \author P. Demongandin, Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_HEATEXCHANGER_H_
#define ALICEO2_MFT_HEATEXCHANGER_H_

#include "TNamed.h"
#include "TGeoVolume.h"
#include "TGeoMatrix.h"

namespace AliceO2 { namespace MFT { class Constants; } }

namespace AliceO2 {
namespace MFT {

class HeatExchanger : public TNamed {

 public:

  HeatExchanger();
  HeatExchanger(Double_t Rwater, Double_t DRPipe, Double_t HeatExchangerThickness, Double_t CarbonThickness);
  
  virtual ~HeatExchanger() {};
  
  TGeoVolumeAssembly* Create(Int_t kHalf, Int_t disk);

  void CreateHalfDisk0(Int_t half);
  void CreateHalfDisk1(Int_t half);
  void CreateHalfDisk2(Int_t half);
  void CreateHalfDisk3(Int_t half);
  void CreateHalfDisk4(Int_t half);

  Double_t GetWaterRadius() { return fRWater; }
  void SetWaterRadius(Double_t &Rwater) { fRWater = Rwater; }

  Double_t GetPipeThickness() { return fDRPipe; }
  void SetPipeThickness(Double_t &DRPipe) { fDRPipe = DRPipe; }

  Double_t GetExchangerWidth() { return fHeatExchangerThickness; }
  void SetExchangerWidth(Double_t &HeatExchangerThickness) { fHeatExchangerThickness = HeatExchangerThickness; }

  Double_t GetCarbonThickness() { return fCarbonThickness; }
  void SetCarbonThickness(Double_t &CarbonThickness) { fCarbonThickness = CarbonThickness; }

  TGeoMaterial *material;
  TGeoMedium *water;
  TGeoMedium *pipe;
  TGeoMedium *carbon;
  TGeoMedium *rohacell;

 private:

  enum EMedia{kAir, kSi, kReadout, kSupport, kCarbon, kAlu, kWater, kSiO2, kInox, kPEEK, kFR4, kCu, kRohacell};  // media IDs used in CreateMaterials

  enum {kBottom, kTop, kNHalves};

  Int_t fNMaxDisks;

  TGeoVolumeAssembly *fHalfDisk;
  
  TGeoRotation ***fHalfDiskRotation;
  TGeoCombiTrans ***fHalfDiskTransformation;

  void InitParameters();

  Double_t fRWater;  // Radius of the water 
  Double_t fDRPipe;  // Thickness of the pipe
  Double_t fHeatExchangerThickness; //width of the heat exchanger
  Double_t fCarbonThickness; //thickness of carbon plate over 2
  Double_t fHalfDiskGap; //gap between half disks

  Double_t fRohacellThickness;

  //Parameters for carbon and Rohacell
  Int_t fnPart[5]; //number of part of each half-disk
  Double_t fRMin[5]; // radius of the central hole of each disk
  Double_t fZPlan[5]; //position on z axis of each plane

  //Dimensions of carbon and Rohacell planes
  Double_t **fSupportXDimensions;
  Double_t **fSupportYDimensions;

  //Parameters for disk0, disk1 and disk2
  Double_t fLWater; // Length of tube part
  Double_t fXPosition0[3]; //position on x axis of each tube for disk 0, 1 and 2 
  Double_t fangle0; //angle of the sides torus part of each pipe for disk 0, 1 and 2
  Double_t fradius0; // radius of the sides torus part for disk 0, 1 and 2
  Double_t fLpartial0; // length of partial tube part

   //Parameters for disk3 
  Double_t fLWater3[3]; // length of tube part for third plan 
  Double_t fXPosition3[4]; // tube position on x axe of each tube for disk 3
  Double_t fangle3[3]; // angle of sides torus of each pipe for disk 3
  Double_t fradius3[3]; // radius of the sides torus for disk 3
  Double_t fangleThirdPipe3; // angle with x axe of tube part of third pipe
  Double_t fLpartial3[2]; // length of partial tube

  Double_t fradius3fourth[4]; // radius of fourth pipe torus of fourth pipe
  Double_t fangle3fourth[4]; // angle of fourth pipe torus of fourth pipe
  Double_t fbeta3fourth[3]; // shift angle of different torus part of fourth pipe of disk 3

  //Parameters for disk4
  Double_t fLwater4[3]; // length of tube part for fourth plan
  Double_t fXposition4[5]; // tube position on x axe of each tube for disk 4
  Double_t fangle4[6]; // angle of sides torus of each pipe for disk 4
  Double_t fradius4[5]; // radius of the sides torus for disk 4
  Double_t fLpartial4[2]; // length of partial tube for first and second pipe of disk 4
  Double_t fangle4fifth[4]; // angle of torus for fifth pipe of disk 4
  Double_t fradius4fifth[4]; // radius of torus for fifth pipe of disk 4

  /// \cond CLASSIMP
  ClassDef(HeatExchanger, 1);
  /// \endcond

};

}
}

#endif
