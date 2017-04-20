/// \file HeatExchanger.h
/// \brief MFT heat exchanger builder
/// \author P. Demongandin, Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_HEATEXCHANGER_H_
#define ALICEO2_MFT_HEATEXCHANGER_H_

#include "TNamed.h"
#include "TGeoVolume.h"
#include "TGeoMatrix.h"

namespace o2 {
namespace MFT {

class HeatExchanger : public TNamed {

 public:

  HeatExchanger();
  HeatExchanger(Double_t Rwater, Double_t DRPipe, Double_t HeatExchangerThickness, Double_t CarbonThickness);
  
  ~HeatExchanger() override = default;
  
  TGeoVolumeAssembly* create(Int_t kHalf, Int_t disk);

  void createHalfDisk0(Int_t half);
  void createHalfDisk1(Int_t half);
  void createHalfDisk2(Int_t half);
  void createHalfDisk3(Int_t half);
  void createHalfDisk4(Int_t half);
  void createManyfold(Int_t disk);

  Double_t getWaterRadius() { return mRWater; }
  void setWaterRadius(Double_t &Rwater) { mRWater = Rwater; }

  Double_t getPipeThickness() { return mDRPipe; }
  void setPipeThickness(Double_t &DRPipe) { mDRPipe = DRPipe; }

  Double_t getExchangerWidth() { return mHeatExchangerThickness; }
  void setExchangerWidth(Double_t &HeatExchangerThickness) { mHeatExchangerThickness = HeatExchangerThickness; }

  Double_t getCarbonThickness() { return mCarbonThickness; }
  void setCarbonThickness(Double_t &CarbonThickness) { mCarbonThickness = CarbonThickness; }

  TGeoMaterial *mMaterial;
  TGeoMedium *mWater;
  TGeoMedium *mPipe;
  TGeoMedium *mCarbon;
  TGeoMedium *mRohacell;

 private:

  void initParameters();

  const static Int_t sNMaxDisks;

  enum {Bottom, Top, NHalves};

  TGeoVolumeAssembly *mHalfDisk;
  
  TGeoRotation ***mHalfDiskRotation;
  TGeoCombiTrans ***mHalfDiskTransformation;

  Double_t mRWater;  // Radius of the water 
  Double_t mDRPipe;  // Thickness of the pipe
  Double_t mHeatExchangerThickness; //width of the heat exchanger
  Double_t mCarbonThickness; //thickness of carbon plate over 2
  Double_t mHalfDiskGap; //gap between half disks

  Double_t mRohacellThickness;

  //Parameters for carbon and Rohacell
  Int_t mNDisks;  //number of planes
  Int_t mNPart[5]; //number of part of each half-disk
  Double_t mRMin[5]; // radius of the central hole of each disk
  Double_t mZPlan[5]; //position on z axis of each plane

  //Dimensions of carbon and Rohacell planes
  Double_t **mSupportXDimensions;
  Double_t **mSupportYDimensions;

  //Parameters for disk0, disk1 and disk2
  Double_t mLWater; // Length of tube part
  Double_t mXPosition0[3]; //position on x axis of each tube for disk 0, 1 and 2 
  Double_t mAngle0; //angle of the sides torus part of each pipe for disk 0, 1 and 2
  Double_t mRadius0; // radius of the sides torus part for disk 0, 1 and 2
  Double_t mLpartial0; // length of partial tube part

   //Parameters for disk3 
  Double_t mLWater3[3]; // length of tube part for third plan 
  Double_t mXPosition3[4]; // tube position on x axe of each tube for disk 3
  Double_t mAngle3[3]; // angle of sides torus of each pipe for disk 3
  Double_t mRadius3[3]; // radius of the sides torus for disk 3
  Double_t mAngleThirdPipe3; // angle with x axe of tube part of third pipe
  Double_t mLpartial3[2]; // length of partial tube

  Double_t mRadius3fourth[4]; // radius of fourth pipe torus of fourth pipe
  Double_t mAngle3fourth[4]; // angle of fourth pipe torus of fourth pipe
  Double_t mBeta3fourth[3]; // shift angle of different torus part of fourth pipe of disk 3

  //Parameters for disk4
  Double_t mLWater4[3]; // length of tube part for fourth plan
  Double_t mXPosition4[5]; // tube position on x axe of each tube for disk 4
  Double_t mAngle4[6]; // angle of sides torus of each pipe for disk 4
  Double_t mRadius4[5]; // radius of the sides torus for disk 4
  Double_t mLpartial4[2]; // length of partial tube for first and second pipe of disk 4
  Double_t mAngle4fifth[4]; // angle of torus for fifth pipe of disk 4
  Double_t mRadius4fifth[4]; // radius of torus for fifth pipe of disk 4

  ClassDefOverride(HeatExchanger, 2);

};

}
}

#endif
