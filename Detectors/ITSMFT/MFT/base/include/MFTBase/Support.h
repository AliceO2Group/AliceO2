// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Support.h
/// \brief Class describing geometry of one MFT half-disk support
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#ifndef ALICEO2_MFT_SUPPORT_H_
#define ALICEO2_MFT_SUPPORT_H_

#include "FairLogger.h"

class TGeoVolume;
class TGeoCompositeShape;

namespace o2
{
namespace mft
{

class Support
{

 public:
  Support();

  ~Support();

  TGeoVolumeAssembly* createVolume(Int_t half, Int_t disk);
  TGeoVolumeAssembly* createPCBs(Int_t half, Int_t disk);
  TGeoVolumeAssembly* createPCB_00_01(Int_t half, Int_t disk);
  TGeoVolumeAssembly* createPCB_02(Int_t half, Int_t disk);
  TGeoVolumeAssembly* createPCB_03(Int_t half, Int_t disk);
  TGeoVolumeAssembly* createPCB_04(Int_t half, Int_t disk);
  TGeoVolumeAssembly* createPCB_PSU(Int_t half, Int_t disk);
  TGeoVolume* createSupport(Int_t half, Int_t disk);
  TGeoVolume* createDisk_Support_00();
  TGeoVolume* createDisk_Support_01();
  TGeoVolume* createDisk_Support_02();
  TGeoVolume* createDisk_Support_03();
  TGeoVolume* createDisk_Support_04();

  TGeoCompositeShape* screw_array(Int_t N, Double_t gap = 1.7);
  TGeoCompositeShape* screw_C();
  TGeoCompositeShape* screw_D();
  TGeoCompositeShape* screw_E();
  TGeoCompositeShape* through_hole_a(Double_t thickness = .8);
  TGeoCompositeShape* through_hole_b(Double_t thickness = .8);
  TGeoCompositeShape* through_hole_c(Double_t thickness = .8);
  TGeoCompositeShape* through_hole_d(Double_t thickness = .8);
  TGeoCompositeShape* through_hole_e(Double_t thickness = .8);

 protected:
  TGeoVolumeAssembly* mSupportVolume;
  Double_t mSupportThickness;
  Double_t mPCBThickness;

 private:
  ClassDef(Support, 1)
};
} // namespace mft
} // namespace o2

#endif
