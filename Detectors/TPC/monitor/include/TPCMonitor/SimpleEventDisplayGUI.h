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

/// \file SimpleEventDisplayGUI.h
/// \brief GUI for raw data event display
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef TPC_SimpleEventDisplayGUI_H_
#define TPC_SimpleEventDisplayGUI_H_

#include <memory>

#include "TString.h"

#include "TPCMonitor/SimpleEventDisplay.h"

class TH2F;
class TH1F;
class TH1;
class TGTextEntry;
class TGTextButton;
class TGCheckButton;

namespace o2::tpc
{

class SimpleEventDisplayGUI
{
 public:
  enum RunMode {
    Offline = 0, ///< run offline from file
    Online = 1,  ///< run online from decoded digits
  };

  enum HistogramType {
    MaxValues = 0,
    Occupancy = 1,
  };

  void toggleFFT();
  void toggleOccupancy();
  void monitorGui();
  void exitRoot();
  void update(TString clist);
  void resetHists(int type, HistogramType histogramType);
  void drawPadSignal(int event, int x, int y, TObject* o);
  void fillHists(int type = 0, HistogramType histogramType = MaxValues);
  void selectSector(int sector);
  int FindROCFromXY(const float x, const float y, const int side);
  void selectSectorExec(int event, int x, int y, TObject* o);
  void initGUI();
  void next(int eventNumber = -1);
  void callEventNumber();
  void applySignalThreshold();

  void runSimpleEventDisplay(std::string_view fileInfo, std::string_view pedestalFile = "", int firstTimeBin = 0, int lastTimeBin = 500, int nTimeBinsPerCall = 500, uint32_t verbosity = 0, uint32_t debugLevel = 0, int selectedSector = 0, bool showSides = 1);

  SimpleEventDisplay& getEventDisplay() { return mEvDisp; }

  void setMode(RunMode mode) { mRunMode = mode; }

  // ===| for online processing |===
  void startGUI(int maxTimeBins = 114048);
  bool isStopRequested() const { return mStop; }
  bool isProcessingEvent() const { return mProcessingEvent; }
  bool isNextEventRequested() const { return mNextEvent; }
  bool isWaitingForDigitUpdate() const { return mUpdatingDigits; }
  void resetNextEventReqested() { mNextEvent = false; }
  void resetUpdatingDigits() { mUpdatingDigits = false; }
  void setDataAvailable(bool available) { mDataAvailable = available; }

 private:
  SimpleEventDisplay mEvDisp;

  int mOldHooverdSector = -1;
  int mSelectedSector = 0;
  int mMaxEvents = 100000000;
  bool mShowSides = true;

  // ===| for onine processing |===
  bool mNextEvent = false;
  bool mProcessingEvent = false;
  bool mUpdatingDigits = false;
  bool mStop = false;
  bool mDataAvailable = false;
  RunMode mRunMode;

  TH2F* mHMaxA = nullptr;
  TH2F* mHMaxC = nullptr;
  TH2F* mHMaxIROC = nullptr;
  TH2F* mHMaxOROC = nullptr;
  TH1* mHFFTO = nullptr;
  TH1* mHFFTI = nullptr;
  TH2F* mHOccupancyA = nullptr;
  TH2F* mHOccupancyC = nullptr;
  TH2F* mHOccupancyIROC = nullptr;
  TH2F* mHOccupancyOROC = nullptr;

  TGCheckButton* mCheckFFT = nullptr;
  TGCheckButton* mCheckOccupancy = nullptr;
  TGTextEntry* mEventNumber = nullptr;
  TGTextEntry* mSignalThresholdValue = nullptr;

  TH1* getBinInfoXY(int& binx, int& biny, float& bincx, float& bincy);

  void initOccupancyHists();
  void deleteOccupancyHists();

  ClassDefNV(SimpleEventDisplayGUI, 0);
};

} // namespace o2::tpc

#endif
