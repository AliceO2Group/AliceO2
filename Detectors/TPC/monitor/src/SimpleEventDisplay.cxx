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

/// \file   SimpleEventDisplay.cxx
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include "TH1D.h"
#include "TH2S.h"
#include "TROOT.h"
#include "TString.h"

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/CRU.h"
#include "TPCBase/Mapper.h"

#include "TPCMonitor/SimpleEventDisplay.h"

using namespace o2::tpc;

SimpleEventDisplay::SimpleEventDisplay()
  : CalibRawBase(),
    mPadMax("qMax", PadSubset::ROC),
    mPadOccupancy("occupancy", PadSubset::ROC),
    mHSigIROC(nullptr),
    mHSigOROC(nullptr),
    mPedestals(nullptr),
    mCurrentChannel(-1),
    mCurrentROC(-1),
    mLastSector(-1),
    mSelectedSector(-1),
    mLastSelSector(-1),
    mCurrentRow(-1),
    mCurrentPad(-1),
    mMaxPadSignal(-1),
    mMaxTimeBin(-1),
    mSectorLoop(kFALSE),
    mFirstTimeBin(0),
    mLastTimeBin(512),
    mTPCmapper(Mapper::instance()),
    mSignalThreshold(0),
    mShowOccupancy(kFALSE)
{
  initHistograms();
}

void SimpleEventDisplay::initHistograms()
{
  delete mHSigOROC;
  delete mHSigIROC;

  const int nPadsIROC = mTPCmapper.getPadsInIROC();
  const int nPadsOROC = mTPCmapper.getPadsInOROC();
  const int numberOfTimeBins = mLastTimeBin - mFirstTimeBin;

  mHSigIROC = new TH2D("PadSigIROC", "Pad Signals IROC", numberOfTimeBins, mFirstTimeBin, mLastTimeBin, nPadsIROC, 0, nPadsIROC);
  mHSigOROC = new TH2D("PadSigOROC", "Pad Signals OROC", numberOfTimeBins, mFirstTimeBin, mLastTimeBin, nPadsOROC, 0, nPadsOROC);
}

//_____________________________________________________________________
Int_t SimpleEventDisplay::updateROC(const Int_t roc,
                                    const Int_t row,
                                    const Int_t pad,
                                    const Int_t timeBin,
                                    const Float_t signal)
{
  //
  // Signal filling methode on the fly pedestal and time offset correction if necessary.
  // no extra analysis necessary. Assumes knowledge of the signal shape!
  // assumes that it is looped over consecutive time bins of one pad
  //
  // printf("update called: %d, %d, %d, %d, %.3f\n", roc, row, pad, timeBin, signal);
  if (row < 0) {
    return 0;
  }
  if (pad < 0) {
    return 0;
  }
  if (timeBin < 0) {
    return 0;
  }
  if ((timeBin > mLastTimeBin) || (timeBin < mFirstTimeBin)) {
    return 0;
  }
  if (mSectorLoop && roc % 36 != mSelectedSector % 36) {
    return 0;
  }

  if (row < 0 || pad < 0) {
    printf("Wrong Pad or Row number, skipping!");
    return 0;
  }

  float corrSignal = signal;

  // ===| get pedestal |========================================================
  if (mPedestals) {
    corrSignal -= mPedestals->getValue(ROC(roc), row, pad);
  }

  const int iChannel = mTPCmapper.getPadNumberInROC(PadROCPos(roc, row, pad));

  // init first pad and roc in this event
  if (mCurrentChannel == -1) {
    mCurrentChannel = iChannel;
    mCurrentROC = roc;
    mCurrentRow = row;
    mCurrentPad = pad;
  }

  // process last pad if we change to a new one
  if (iChannel != mCurrentChannel) {
    mLastSector = mCurrentROC;
    mCurrentChannel = iChannel;
    mCurrentROC = roc;
    mCurrentRow = row;
    mCurrentPad = pad;
    mMaxPadSignal = 0;
  }

  // fill signals for current pad
  if (mCurrentROC % 36 == mSelectedSector % 36) {
    const Int_t nbins = mLastTimeBin - mFirstTimeBin;
    const Int_t offset = (nbins + 2) * (iChannel + 1) + (timeBin - mFirstTimeBin) + 1;

    if ((UInt_t)roc < mTPCmapper.getNumberOfIROCs()) {
      mHSigIROC->GetArray()[offset] = corrSignal >= mSignalThreshold ? corrSignal : 0;
    } else {
      mHSigOROC->GetArray()[offset] = corrSignal >= mSignalThreshold ? corrSignal : 0;
    }
  }

  CalROC& calROC = mPadMax.getCalArray(mCurrentROC);
  auto val = calROC.getValue(row, pad);

  if (corrSignal > val && corrSignal >= mSignalThreshold) {
    calROC.setValue(row, pad, corrSignal);
    mMaxPadSignal = corrSignal;
    mMaxTimeBin = timeBin;
  }

  CalROC& calROCOccupancy = mPadOccupancy.getCalArray(mCurrentROC);
  const auto occupancy = calROCOccupancy.getValue(row, pad);

  if (corrSignal >= mSignalThreshold) {
    calROCOccupancy.setValue(row, pad, occupancy + 1.0f);
  }

  return 0;
}

//_____________________________________________________________________
TH1D* SimpleEventDisplay::makePadSignals(Int_t roc, Int_t row, Int_t pad)
{
  const int padOffset = (roc > 35) * Mapper::getPadsInIROC();
  const int channel = mTPCmapper.getPadNumberInROC(PadROCPos(roc, row, pad));

  const FECInfo& fecInfo = mTPCmapper.getFECInfo(PadROCPos(roc, row, pad));
  const int cruNumber = mTPCmapper.getCRU(ROC(roc).getSector(), channel + padOffset);
  const CRU cru(cruNumber);
  const PartitionInfo& partInfo = mTPCmapper.getMapPartitionInfo()[cru.partition()];
  const int nFECs = partInfo.getNumberOfFECs();
  const int fecOffset = (nFECs + 1) / 2;
  const int fecInPartition = fecInfo.getIndex() - partInfo.getSectorFECOffset();
  const int dataWrapperID = fecInPartition >= fecOffset;
  const int globalLinkID = (fecInPartition % fecOffset) + dataWrapperID * 12;

  mSelectedSector = roc;

  // attention change for if event has changed
  if (mSelectedSector % 36 != mLastSelSector % 36) {
    mSectorLoop = kTRUE;
    processEvent(getPresentEventNumber());
    mLastSelSector = mSelectedSector;
    mSectorLoop = kFALSE;
  }
  TH1D* h = nullptr;
  const Int_t nbins = mLastTimeBin - mFirstTimeBin;
  if (nbins <= 0) {
    return nullptr;
  }
  const Int_t offset = (nbins + 2) * (channel + 1);
  Double_t* arrP = nullptr;

  TString title("#splitline{#lower[.1]{#scale[.5]{");

  if (roc < (Int_t)mTPCmapper.getNumberOfIROCs()) {
    h = (TH1D*)gROOT->FindObject("PadSignals_IROC");
    if (!h) {
      h = new TH1D("PadSignals_IROC", "PadSignals IROC;time bins (200ns);amplitude (ADC counts)", nbins, mFirstTimeBin, mLastTimeBin);
    }
    h->SetFillColor(kBlue - 10);
    arrP = mHSigIROC->GetArray() + offset;
    //     title+="IROC ";
  } else {
    h = (TH1D*)gROOT->FindObject("PadSignals_OROC");
    if (!h) {
      h = new TH1D("PadSignals_OROC", "PadSignals OROC;time bins (200ns);amplitude (ADC counts)", nbins, mFirstTimeBin, mLastTimeBin);
    }
    h->SetFillColor(kBlue - 10);
    arrP = mHSigOROC->GetArray() + offset;
    title += "OROC ";
  }
  title += (roc / 18 % 2 == 0) ? "A" : "C";
  title += Form("%02d (%02d) row: %02d, pad: %03d, globalpad: %05d (in roc)}}}{#scale[.5]{FEC: %02d (%02d), Chip: %02d, Chn: %02d, CRU: %d, Link: %02d (%s%d)}}",
                roc % 18, roc, row, pad, channel, fecInfo.getIndex(), fecInPartition, fecInfo.getSampaChip(), fecInfo.getSampaChannel(), cruNumber % CRU::CRUperSector, globalLinkID, dataWrapperID ? "B" : "A", globalLinkID % 12);

  h->SetTitle(title.Data());
  Int_t entries = 0;
  for (Int_t i = 0; i < nbins; i++) {
    entries += (Int_t)arrP[i + 1];
  }
  h->Set(nbins + 2, arrP);
  h->SetEntries(entries);
  return h;
}
//_____________________________________________________________________
void SimpleEventDisplay::resetEvent()
{
  //
  //
  //
  if (!mSectorLoop) {
    mPadMax.multiply(0.);
    mPadOccupancy.multiply(0.);
  }
  mHSigIROC->Reset();
  mHSigOROC->Reset();
}
