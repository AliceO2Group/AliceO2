// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_RAWDISPLAY_H_
#define ALICEO2_TRD_RAWDISPLAY_H_

///
/// \file   RawDisplay.h
/// \author Thomas Dietel, tom@dietel.net
///

#include "TRDQC/RawDataManager.h"

class TVirtualPad;
class TH2;

namespace o2::trd
{

/// RawDisplay: base class to visualize TRD raw and associated data
/// The display is based on a TH2 heat map of ADC values and tracklets as lines.
/// Clusters can be calculated and drawn based on the ADC values.
class RawDisplay
{
 public:
  RawDisplay(RawDataSpan& dataspan, TVirtualPad* pad = nullptr);
  void drawDigits(std::string opt = "colz");
  void drawTracklets();
  void drawClusters();
  void drawHits();
  void drawMCTrackSegments();

  void draw()
  {
    drawDigits();
    drawTracklets();
  }

 protected:
  RawDataSpan& mDataSpan;
  TVirtualPad* mPad{0};
  TH2* mDigitsHisto{0};
  std::string mName;
  std::string mDesc;
  int mFirstPad;
  int mLastPad;

  float mClusterThreshold{50}; /// threshold for drawing clusters
};

/// The MCM display is a raw display specialized to display data for a single MCM
class MCMDisplay : public RawDisplay
{
 public:
  MCMDisplay(RawDataSpan& mcmdata, TVirtualPad* pad = nullptr);
};

} // namespace o2::trd

#endif // ALICEO2_TRD_RAWDISPLAY_H_
