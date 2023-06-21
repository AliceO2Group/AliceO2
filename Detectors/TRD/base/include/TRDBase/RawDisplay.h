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


// #include "DataFormatsTRD/Digit.h"
// #include "DataFormatsTRD/Tracklet64.h"
// #include "DataFormatsTRD/TriggerRecord.h"
// #include "DataFormatsTRD/Hit.h"

#include "TRDBase/DataManager.h"

// #include "DataFormatsTRD/HelperMethods.h"


// #include "TRDBase/Geometry.h"

// #include "DetectorsBase/GeometryManager.h"
// #include "DetectorsBase/Propagator.h"
// #include "ReconstructionDataFormats/TrackTPCITS.h"
// #include <DataFormatsTPC/TrackTPC.h>
// #include "CommonDataFormat/TFIDInfo.h"

// #include <TTreeReaderArray.h>


// #include <vector>
// #include <array>
// #include <map>
// #include <ostream>
// #include <filesystem>

// class TFile;
// class TTreeReader;
// class TPad;
class TVirtualPad;
class TH2;

// class TPad;


// template<typename T>
// class TTreeReaderArray<T>;

namespace o2::trd
{

/// \namespace rawdisp
/// \brief Raw data display and analysis
///
/// This namespace provides helper classes to display low-level TRD data.
///
/// origin: TRD
/// \author Thomas Dietel, tom@dietel.net
namespace rawdisp
{


class RawDisplay
{
public:
  RawDisplay(RawDataSpan &dataspan, TVirtualPad *pad=NULL);
  void DrawDigits();
  void DrawTracklets();
  void DrawClusters();

  void Draw() { DrawDigits(); DrawTracklets(); }

protected:
  RawDataSpan &mDataSpan;
  TVirtualPad* mPad{0};
  TH2* mDigitsHisto{0};
  std::string mName;
  std::string mDesc;
  int mFirstPad;
  int mLastPad;
};

class MCMDisplay : public RawDisplay
{
public:
  MCMDisplay(RawDataSpan &mcmdata, TVirtualPad *pad=NULL);
};

// TPad *DrawMCM(RawDataSpan &mcm, TPad *pad=NULL);



} // namespace rawdisp

} // namespace o2::trd

#endif // ALICEO2_TRD_RAWDISPLAY_H_
