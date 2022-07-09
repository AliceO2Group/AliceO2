// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Screenshot.h
/// \brief Screenshot functionality
/// \author julian.myrcha@cern.ch
/// \author michal.chwesiuk@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_SCREENSHOT_H
#define ALICE_O2_EVENTVISUALISATION_SCREENSHOT_H

#include <TASImage.h>
#include <DetectorsCommonDataFormats/DetID.h>
#include <string>
namespace o2
{
namespace event_visualisation
{

class Screenshot
{
 private:
  static bool CopyImage(TASImage* dst, TASImage* src, Int_t x_dst, Int_t y_dst, Int_t x_src, Int_t y_src,
                        UInt_t w_src, UInt_t h_src);
  static TASImage* ScaleImage(TASImage* image, UInt_t desiredWidth, UInt_t desiredHeight, const std::string& backgroundColor);

 public:
  static void perform(o2::detectors::DetID::mask_t detectorsMask, int runNumber, int firstTForbit, std::string collisionTime);
};

} // namespace event_visualisation
} // namespace o2

#endif // ALICE_O2_EVENTVISUALISATION_SCREENSHOT_H