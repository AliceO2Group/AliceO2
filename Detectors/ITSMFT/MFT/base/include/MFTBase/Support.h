/// \file Support.h
/// \brief Class describing geometry of one MFT half-disk support
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#ifndef ALICEO2_MFT_SUPPORT_H_
#define ALICEO2_MFT_SUPPORT_H_

#include "TNamed.h"

#include "FairLogger.h"

class TGeoVolume;
class TGeoCompositeShape;

namespace o2 {
namespace MFT {

class Support : public TNamed {
  
public:
  
  Support();
  
  ~Support() override;
  
  TGeoVolumeAssembly* createVolume(Int_t half, Int_t disk);
  TGeoVolumeAssembly* createPCBs(Int_t half, Int_t disk);
  TGeoVolumeAssembly* createPCB_00_01(Int_t half, Int_t disk);
  TGeoVolumeAssembly* createPCB_02(Int_t half, Int_t disk);
  TGeoVolumeAssembly* createPCB_03(Int_t half, Int_t disk);
  TGeoVolumeAssembly* createPCB_04(Int_t half, Int_t disk);
  TGeoVolumeAssembly* createPCB_PSU(Int_t half, Int_t disk);
  TGeoVolume* createSupport(Int_t half, Int_t disk);
  TGeoVolume * createDisk_Support_00 ();
  TGeoVolume * createDisk_Support_01 ();
  TGeoVolume * createDisk_Support_02 ();
  TGeoVolume * createDisk_Support_03 ();
  TGeoVolume * createDisk_Support_04 ();

  TGeoCompositeShape * screw_array (Int_t N, Double_t gap = 1.7);
  TGeoCompositeShape * screw_C ();
  TGeoCompositeShape * screw_D ();
  TGeoCompositeShape * screw_E ();
  TGeoCompositeShape * through_hole_a (Double_t thickness=.8);
  TGeoCompositeShape * through_hole_b (Double_t thickness=.8);
  TGeoCompositeShape * through_hole_c (Double_t thickness=.8);
  TGeoCompositeShape * through_hole_d (Double_t thickness=.8);
  TGeoCompositeShape * through_hole_e (Double_t thickness=.8);
  
protected:
  
  TGeoVolumeAssembly * mSupportVolume;
  Double_t mSupportThickness;
  Double_t mPCBThickness;

private:
  
  ClassDefOverride(Support,1)
  
};

}
}

#endif

