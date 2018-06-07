// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digitizer.cxx
/// \brief Implementation of the ITS digitizer

#include "ITSMFTBase/Digit.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTSimulation/Digitizer.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include <TRandom.h>
#include <climits>
#include <vector>
#include <numeric>
#include "FairLogger.h" // for LOG

ClassImp(o2::ITSMFT::Digitizer)

  using o2::ITSMFT::Hit;
using o2::ITSMFT::Digit;
using Segmentation = o2::ITSMFT::SegmentationAlpide;

using namespace o2::ITSMFT;
// using namespace o2::Base;

//_______________________________________________________________________
void Digitizer::init()
{
  const Int_t numOfChips = mGeometry->getNumberOfChips();
  mChips.resize(numOfChips);
  for (int i = numOfChips; i--;) {
    mChips[i].setChipIndex(i);
  }
  if (!mParams.getAlpSimResponse()) {
    mAlpSimResp = std::make_unique<o2::ITSMFT::AlpideSimResponse>();
    mAlpSimResp->initData();
    mParams.setAlpSimResponse(mAlpSimResp.get());
  }

}

//_______________________________________________________________________
void Digitizer::process()
{
  // digitize single event

  const Int_t numOfChips = mGeometry->getNumberOfChips();

  // estimate the smalles RO Frame this event may have
  double hTime0 = mEventTime - mParams.getTimeOffset();
  if (hTime0 > UINT_MAX) {
    LOG(WARNING) << "min Hit RO Frame undefined: time: " << hTime0 << " is in far future: "
                 << " EventTime: " << mEventTime << " TimeOffset: " << mParams.getTimeOffset() << FairLogger::endl;
    return;
  }

  if (hTime0 < 0) {
    hTime0 = 0.;
  }
  UInt_t minNewROFrame = static_cast<UInt_t>(hTime0 / mParams.getROFrameLenght());

  LOG(INFO) << "Digitizing ITS event at time " << mEventTime << " (TOffset= " << mParams.getTimeOffset()
            << " ROFrame= " << minNewROFrame << ")"
            << " cont.mode: " << isContinuous() << " current Min/Max RO Frames " << mROFrameMin << "/" << mROFrameMax
            << FairLogger::endl;

  if (mParams.isContinuous() && minNewROFrame > mROFrameMin) {
    // if there are already digits cached for previous RO Frames AND the new event
    // cannot contribute to these digits, move them to the output container
    if (mROFrameMax < minNewROFrame) {
      mROFrameMax = minNewROFrame - 1;
    }
    for (auto rof = mROFrameMin; rof < minNewROFrame; rof++) {
      fillOutputContainer(rof);
    }
  }

  int nHits = mHits->size();
  std::vector<int> hitIdx(nHits);
  std::iota(std::begin(hitIdx), std::end(hitIdx), 0);
  // sort hits to improve memory access
  std::sort(hitIdx.begin(), hitIdx.end(),
            [&](auto lhs, auto rhs) {
              return (*mHits)[lhs].GetDetectorID() < (*mHits)[rhs].GetDetectorID();
            });
  for (int i : hitIdx) {
    processHit((*mHits)[i], mROFrameMax);
  }
  // in the triggered mode store digits after every MC event
  if (!mParams.isContinuous()) {
    fillOutputContainer(mROFrameMax);
  }
}

//_______________________________________________________________________
void Digitizer::setEventTime(double t)
{
  // assign event time, it should be in a strictly increasing order
  // convert to ns
  t *= mCoeffToNanoSecond;

  if (t < mEventTime && mParams.isContinuous()) {
    LOG(FATAL) << "New event time (" << t << ") is < previous event time (" << mEventTime << ")" << FairLogger::endl;
  }
  mEventTime = t;
  // to randomize the RO phase wrt the event time we use a random offset
  if (mParams.isContinuous()) {       // in continuous mode we set the offset only in the very beginning
    if (!mParams.isTimeOffsetSet()) { // offset is initially at -inf
      mParams.setTimeOffset(0);       ///*mEventTime + */ mParams.getROFrameLenght() * (gRandom->Rndm() - 0.5));
    }
  } else { // in the triggered mode we start from 0 ROFrame in every event
    mParams.setTimeOffset(mEventTime); // + mParams.getROFrameLenght() * (gRandom->Rndm() - 0.5));
    mROFrameMin = 0; // so we reset the frame counters
    mROFrameMax = 0;
  }
}

//_______________________________________________________________________
void Digitizer::fillOutputContainer(UInt_t maxFrame)
{
  // fill output with digits ready to be stored, generating the noise beforehand
  if (maxFrame > mROFrameMax) {
    maxFrame = mROFrameMax;
  }
  // make sure all buffers for extra digits are created
  getExtraDigBuffer(mROFrameMax);

  LOG(INFO) << "Filling ITS digits output for RO frames " << mROFrameMin << ":" << maxFrame << FairLogger::endl;

  // we have to write chips in RO increasing order, therefore have to loop over the frames here
  auto rof = mROFrameMin;
  for (; rof <= maxFrame; rof++) {
    auto& extra = *(mExtraBuff.front().get());
    for (auto& chip : mChips) {
      //      chip.fillOutputContainer(digits, rof, &mParams);
      chip.addNoise(rof, rof, &mParams);
      auto& buffer = chip.getPreDigits();
      if (buffer.empty()) {
        continue;
      }
      auto itBeg = buffer.begin();
      auto iter = itBeg;
      ULong64_t maxKey = chip.getOrderingKey(rof + 1, 0, 0); // fetch digits with key below that
      for (; iter != buffer.end(); ++iter) {
        if (iter->first > maxKey) {
          break; // is the digit ROFrame from the key > the max requested frame
        }
        auto& preDig = iter->second; // preDigit
        if (preDig.charge >= mParams.getChargeThreshold()) {
          int digID = mDigits->size();
          mDigits->emplace_back(chip.getChipIndex(), rof, preDig.row, preDig.col, preDig.charge);
          mMCLabels->addElement(digID, preDig.labelRef.label);
          auto& nextRef = preDig.labelRef; // extra contributors are in extra array
          while (nextRef.next >= 0) {
            nextRef = extra[nextRef.next];
            mMCLabels->addElement(digID, nextRef.label);
          }
        }
      }
      buffer.erase(itBeg, iter);
    }
    extra.clear(); // clear container for extra digits of the rof ROFrame
    // and move it as a new slot in the end
    mExtraBuff.emplace_back(mExtraBuff.front().release());
    mExtraBuff.pop_front();
    mROFrameMin++;
  }
  mROFrameMin = maxFrame + 1;
}

//_______________________________________________________________________
void Digitizer::setCurrSrcID(int v)
{
  // set current MC source ID
  if (v > MCCompLabel::maxSourceID()) {
    LOG(FATAL) << "MC source id " << v << " exceeds max storable in the label " << MCCompLabel::maxSourceID()
               << FairLogger::endl;
  }
  mCurrSrcID = v;
}

//_______________________________________________________________________
void Digitizer::setCurrEvID(int v)
{
  // set current MC event ID
  if (v > MCCompLabel::maxEventID()) {
    LOG(FATAL) << "MC event id " << v << " exceeds max storable in the label " << MCCompLabel::maxEventID()
               << FairLogger::endl;
  }
  mCurrEvID = v;
}

//_______________________________________________________________________
void Digitizer::processHit(const o2::ITSMFT::Hit& hit, UInt_t& maxFr)
{
  // conver single hit to digits

  double hTime0 = hit.GetTime() * sec2ns + mEventTime - mParams.getTimeOffset(); // time from the RO start, in ns
  if (hTime0 > UINT_MAX) {
    LOG(WARNING) << "Hit RO Frame undefined: time: " << hTime0 << " is in far future: hitTime: "
                 << hit.GetTime() << " EventTime: " << mEventTime << " ChipOffset: "
                 << mParams.getTimeOffset() << FairLogger::endl;
    return;
  }
  // calculate RO Frame for this hit
  if (hTime0 < 0) {
    hTime0 = 0.;
  }
  float tTot = mParams.getSignalShape().getDuration();
  UInt_t roFrame = UInt_t(hTime0 * mParams.getROFrameLenghtInv());
  std::array<float, maxROFPerHit> timesROF; // start/end time in every ROF covered
  // time till the end of the ROF where the hit was registered, used for signal overflow check
  float tInt = float((roFrame + 1) * mParams.getROFrameLenght() - hTime0);
  timesROF[0] = 0.f;
  timesROF[1] = tInt;
  int nROFs = 1;
  while (tInt < tTot) {
    tInt += mParams.getROFrameLenght();
    timesROF[++nROFs] = tInt;
  }
  UInt_t roFrameLast = roFrame + (nROFs - 1);
  if (roFrameLast > maxFr) {
    maxFr = roFrameLast;
  }

  float nStepsInv = mParams.getNSimStepsInv();
  int nSteps = mParams.getNSimSteps();

  const auto& matrix = mGeometry->getMatrixL2G(hit.GetDetectorID());

  Vector3D<float> xyzLocS(matrix ^ (hit.GetPosStart())); // start position in sensor frame
  Vector3D<float> xyzLocE(matrix ^ (hit.GetPos()));      // end position in sensor frame
  Vector3D<float> step(xyzLocE);
  step -= xyzLocS;
  step *= nStepsInv; // position increment at each step
  // the electrons will injected in the middle of each step
  Vector3D<float> stepH(step * 0.5);
  xyzLocS += stepH;
  xyzLocE -= stepH;

  int rowS = -1, colS = -1, rowE = -1, colE = -1, nSkip = 0;
  // get entrance pixel row and col
  while (!Segmentation::localToDetector(xyzLocS.X(), xyzLocS.Z(), rowS, colS)) { // guard-ring ?
    if (++nSkip >= nSteps) {
      return; // did not enter to sensitive matrix
    }
    xyzLocS += step;
  }
  // get exit pixel row and col
  while (!Segmentation::localToDetector(xyzLocE.X(), xyzLocE.Z(), rowE, colE)) { // guard-ring ?
    if (++nSkip >= nSteps) {
      return; // did not enter to sensitive matrix
    }
    xyzLocE -= step;
  }
  // estimate the limiting min/max row and col where the non-0 response is possible
  if (rowS > rowE) {
    std::swap(rowS, rowE);
  }
  if (colS > colE) {
    std::swap(colS, colE);
  }
  rowS -= AlpideRespSimMat::NPix / 2;
  rowE += AlpideRespSimMat::NPix / 2;
  if (rowS < 0) {
    rowS = 0;
  }
  if (rowE >= Segmentation::NRows) {
    rowE = Segmentation::NRows - 1;
  }
  colS -= AlpideRespSimMat::NPix / 2;
  colE += AlpideRespSimMat::NPix / 2;
  if (colS < 0) {
    colS = 0;
  }
  if (colE >= Segmentation::NCols) {
    colE = Segmentation::NCols - 1;
  }
  int rowSpan = rowE - rowS + 1, colSpan = colE - colS + 1; // size of plaquet where some response is expected

  float respMatrix[rowSpan][colSpan]; // response accumulated here
  std::fill(&respMatrix[0][0], &respMatrix[0][0] + rowSpan * colSpan, 0.f);

  float nElectrons = hit.GetEnergyLoss() * mParams.getEnergyToNElectrons(); // total number of deposited electrons
  nElectrons *= nStepsInv;                                                  // N electrons injected per step
  if (nSkip) {
    nSteps -= nSkip;
  }
  //
  int rowPrev = -1, colPrev = -1, row, col;
  float cRowPix = 0.f, cColPix = 0.f; // local coordinated of the current pixel center

  const o2::ITSMFT::AlpideSimResponse* resp = mParams.getAlpSimResponse();

  // take into account that the AlpideSimResponse has min/max thickness non-symmetric around 0
  xyzLocS.SetY(xyzLocS.Y() + resp->getDepthShift());

  // collect charge in evey pixel which might be affected by the hit
  for (int iStep = nSteps; iStep--;) {
    // Get the pixel ID
    Segmentation::localToDetector(xyzLocS.X(), xyzLocS.Z(), row, col);
    if (row != rowPrev || col != colPrev) { // update pixel and coordinates of its center
      if (!Segmentation::detectorToLocal(row, col, cRowPix, cColPix)) {
        continue; // should not happen
      }
      rowPrev = row;
      colPrev = col;
    }
    bool flipCol, flipRow;
    // note that response needs coordinates along column row (locX) (locZ) then depth (locY)
    auto rspmat = resp->getResponse(xyzLocS.X() - cRowPix, xyzLocS.Z() - cColPix, xyzLocS.Y(), flipRow, flipCol);

    xyzLocS += step;
    if (!rspmat) {
      continue;
    }

    for (int irow = AlpideRespSimMat::NPix; irow--;) {
      int rowDest = row + irow - AlpideRespSimMat::NPix / 2 - rowS; // destination row in the respMatrix
      if (rowDest < 0 || rowDest >= rowSpan) {
        continue;
      }
      for (int icol = AlpideRespSimMat::NPix; icol--;) {
        int colDest = col + icol - AlpideRespSimMat::NPix / 2 - colS; // destination column in the respMatrix
        if (colDest < 0 || colDest >= colSpan) {
          continue;
        }
        respMatrix[rowDest][colDest] += rspmat->getValue(irow, icol, flipRow, flipCol);
      }
    }
  }

  // fire the pixels assuming Poisson(n_response_electrons)
  o2::MCCompLabel lbl(hit.GetTrackID(), mCurrEvID, mCurrSrcID);
  auto& chip = mChips[hit.GetDetectorID()];
  for (int irow = rowSpan; irow--;) {
    UShort_t rowIS = irow + rowS;
    for (int icol = colSpan; icol--;) {
      float nEleResp = respMatrix[irow][icol];
      if (!nEleResp) {
        continue;
      }
      int nEle = gRandom->Poisson(nElectrons * nEleResp); // total charge in given pixel
      // ignore charge which have no chance to fire the pixel
      if (nEle < mParams.getMinChargeToAccount()) {
        continue;
      }
      UShort_t colIS = icol + colS;
      //
      registerDigits(chip, roFrame, nROFs, rowIS, colIS, nEle, lbl, timesROF);
    }
  }
}
