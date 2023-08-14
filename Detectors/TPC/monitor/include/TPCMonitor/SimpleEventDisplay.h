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

#ifndef ALICEO2_TPC_SIMPLEEVENTDISPLAY_H_
#define ALICEO2_TPC_SIMPLEEVENTDISPLAY_H_

/// \file   SimpleEventDisplay.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include "TH1.h"

#include "TPCBase/CalDet.h"
#include "TPCBase/CRU.h"
#include "TPCCalibration/CalibRawBase.h"

class TH2D;

namespace o2
{
namespace tpc
{

class Mapper;
/// \class SimpleEventDisplay
/// \brief Base of a simple event display for digits
///
/// This class is a base for a simple event display
/// It processes raw data and saves digit information for pad and row.
///
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
class SimpleEventDisplay : public CalibRawBase
{
 public:
  SimpleEventDisplay();

  ~SimpleEventDisplay() override = default;

  Int_t updateROC(const Int_t roc, const Int_t row, const Int_t pad,
                  const Int_t timeBin, const Float_t signal) final;

  /// not used
  Int_t updateCRU(const CRU& cru, const Int_t row, const Int_t pad,
                  const Int_t timeBin, const Float_t signal) final { return 0; }

  CalPad* getCalPadMax() { return &mPadMax; }

  CalPad* getCalPadOccupancy() { return &mPadOccupancy; }

  /// Set currently selected sector
  void setSelectedSector(Int_t selectedSector) { mSelectedSector = selectedSector; }

  /// Set last processed sector
  void setLastSector(Int_t lastSector) { mLastSector = lastSector; }

  /// Set last selected sector
  void setLastSelSector(Int_t lastSelSector) { mLastSelSector = lastSelSector; }

  void setPedstals(CalPad* pedestals) { mPedestals = pedestals; }

  void setSignalThreshold(UInt_t signalThreshold) { mSignalThreshold = signalThreshold; }

  void setShowOccupancy(bool showOccupancy) { mShowOccupancy = showOccupancy; }

  TH1D* makePadSignals(Int_t roc, Int_t row, Int_t pad);

  /// set time bin range
  void setTimeBinRange(int firstBin, int lastBin)
  {
    mFirstTimeBin = firstBin;
    mLastTimeBin = lastBin;
    initHistograms();
  }

  /// Dummy end event
  void endEvent() final{};

 private:
  CalPad mPadMax;       //!< Cal Pad with max Entry per channel
  CalPad mPadOccupancy; //!< Cal Pad with Occupancy per channel
  TH2D* mHSigIROC;      //!< iroc signals
  TH2D* mHSigOROC;      //!< oroc signals
  CalPad* mPedestals;   //!< Pedestal calibratino object

  Int_t mCurrentChannel;   //!< current channel processed
  Int_t mCurrentROC;       //!< current ROC processed
  Int_t mLastSector;       //!< Last sector processed
  Int_t mSelectedSector;   //!< Sector selected for processing
  Int_t mLastSelSector;    //!< Last sector selected for processing
  Int_t mCurrentRow;       //!< current row processed
  Int_t mCurrentPad;       //!< current pad processed
  Float_t mMaxPadSignal;   //!< maximum bin of current pad
  Int_t mMaxTimeBin;       //!< time bin with maximum value
  Bool_t mSectorLoop;      //!< only process one sector
  Int_t mFirstTimeBin;     //!< first time bin to accept
  Int_t mLastTimeBin;      //!< last time bin to accept
  UInt_t mSignalThreshold; //!< minimum adc value
  Bool_t mShowOccupancy;   //!< true iff occupancy should be calculated, false otherwise

  const Mapper& mTPCmapper; //! mapper

  void resetEvent() final;
  void initHistograms();
};

} // namespace tpc

} // namespace o2
#endif
