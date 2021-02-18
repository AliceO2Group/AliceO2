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
/// \brief Implementation of the ITS/MFT digitizer

#include "DataFormatsITSMFT/Digit.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTSimulation/Digitizer.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include <TRandom.h>
#include <climits>
#include <vector>
#include <numeric>
#include "FairLogger.h" // for LOG

using o2::itsmft::Digit;
using o2::itsmft::Hit;
using Segmentation = o2::itsmft::SegmentationAlpide;

using namespace o2::itsmft;
// using namespace o2::base;

//_______________________________________________________________________
void Digitizer::init()
{
  const Int_t numOfChips = mGeometry->getNumberOfChips();
  mChips.resize(numOfChips);
  for (int i = numOfChips; i--;) {
    mChips[i].setChipIndex(i);
  }
  if (!mParams.getAlpSimResponse()) {
    mAlpSimResp = std::make_unique<o2::itsmft::AlpideSimResponse>();
    mAlpSimResp->initData();
    mParams.setAlpSimResponse(mAlpSimResp.get());
  }
  mParams.print();
}

//_______________________________________________________________________
void Digitizer::process(const std::vector<Hit>* hits, int evID, int srcID)
{
  // digitize single event, the time must have been set beforehand

  LOG(INFO) << "Digitizing " << mGeometry->getName() << " hits of entry " << evID << " from source "
            << srcID << " at time " << mEventTime << " ROFrame= " << mNewROFrame << ")"
            << " cont.mode: " << isContinuous()
            << " Min/Max ROFrames " << mROFrameMin << "/" << mROFrameMax;

  // is there something to flush ?
  if (mNewROFrame > mROFrameMin) {
    fillOutputContainer(mNewROFrame - 1); // flush out all frame preceding the new one
  }

  int nHits = hits->size();
  std::vector<int> hitIdx(nHits);
  std::iota(std::begin(hitIdx), std::end(hitIdx), 0);
  // sort hits to improve memory access
  std::sort(hitIdx.begin(), hitIdx.end(),
            [hits](auto lhs, auto rhs) {
              return (*hits)[lhs].GetDetectorID() < (*hits)[rhs].GetDetectorID();
            });
  for (int i : hitIdx) {
    processHit((*hits)[i], mROFrameMax, evID, srcID);
  }
  // in the triggered mode store digits after every MC event
  // TODO: in the real triggered mode this will not be needed, this is actually for the
  // single event processing only
  if (!mParams.isContinuous()) {
    fillOutputContainer(mROFrameMax);
  }
}

//_______________________________________________________________________
void Digitizer::setEventTime(const o2::InteractionTimeRecord& irt)
{
  // assign event time in ns
  mEventTime = irt;
  if (!mParams.isContinuous()) {
    mROFrameMin = 0; // in triggered mode reset the frame counters
    mROFrameMax = 0;
  }
  // RO frame corresponding to provided time
  mCollisionTimeWrtROF = mEventTime.timeInBCNS; // in triggered mode the ROF starts at BC (is there a delay?)
  if (mParams.isContinuous()) {
    auto nbc = mEventTime.toLong();
    if (mCollisionTimeWrtROF < 0 && nbc > 0) {
      nbc--;
    }
    mNewROFrame = nbc / mParams.getROFrameLengthInBC();
    // in continuous mode depends on starts of periodic readout frame
    mCollisionTimeWrtROF += (nbc % mParams.getROFrameLengthInBC()) * o2::constants::lhc::LHCBunchSpacingNS;
  } else {
    mNewROFrame = 0;
  }

  if (mNewROFrame < mROFrameMin) {
    LOG(ERROR) << "New ROFrame " << mNewROFrame << " (" << irt << ") precedes currently cashed " << mROFrameMin;
    throw std::runtime_error("deduced ROFrame precedes already processed one");
  }

  if (mParams.isContinuous() && mROFrameMax < mNewROFrame) {
    mROFrameMax = mNewROFrame - 1; // all frames up to this are finished
  }
}

//_______________________________________________________________________
void Digitizer::fillOutputContainer(UInt_t frameLast)
{
  // fill output with digits from min.cached up to requested frame, generating the noise beforehand
  if (frameLast > mROFrameMax) {
    frameLast = mROFrameMax;
  }
  // make sure all buffers for extra digits are created up to the maxFrame
  getExtraDigBuffer(mROFrameMax);

  LOG(INFO) << "Filling " << mGeometry->getName() << " digits output for RO frames " << mROFrameMin << ":"
            << frameLast;

  o2::itsmft::ROFRecord rcROF;

  // we have to write chips in RO increasing order, therefore have to loop over the frames here
  for (; mROFrameMin <= frameLast; mROFrameMin++) {
    rcROF.setROFrame(mROFrameMin);
    rcROF.setFirstEntry(mDigits->size()); // start of current ROF in digits

    auto& extra = *(mExtraBuff.front().get());
    for (auto& chip : mChips) {
      chip.addNoise(mROFrameMin, mROFrameMin, &mParams);
      auto& buffer = chip.getPreDigits();
      if (buffer.empty()) {
        continue;
      }
      auto itBeg = buffer.begin();
      auto iter = itBeg;
      ULong64_t maxKey = chip.getOrderingKey(mROFrameMin + 1, 0, 0) - 1; // fetch digits with key below that
      for (; iter != buffer.end(); ++iter) {
        if (iter->first > maxKey) {
          break; // is the digit ROFrame from the key > the max requested frame
        }
        auto& preDig = iter->second; // preDigit
        if (preDig.charge >= mParams.getChargeThreshold()) {
          int digID = mDigits->size();
          mDigits->emplace_back(chip.getChipIndex(), preDig.row, preDig.col, preDig.charge);
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
    // finalize ROF record
    rcROF.setNEntries(mDigits->size() - rcROF.getFirstEntry()); // number of digits
    if (isContinuous()) {
      rcROF.getBCData().setFromLong(mROFrameMin * mParams.getROFrameLengthInBC());
    } else {
      rcROF.getBCData() = mEventTime; // RSTODO do we need to add trigger delay?
    }
    if (mROFRecords) {
      mROFRecords->push_back(rcROF);
    }
    extra.clear(); // clear container for extra digits of the mROFrameMin ROFrame
    // and move it as a new slot in the end
    mExtraBuff.emplace_back(mExtraBuff.front().release());
    mExtraBuff.pop_front();
  }
}

//_______________________________________________________________________
void Digitizer::processHit(const o2::itsmft::Hit& hit, UInt_t& maxFr, int evID, int srcID)
{
  // convert single hit to digits
  float timeInROF = hit.GetTime() * sec2ns;
  if (timeInROF > 20e3) {
    const int maxWarn = 10;
    static int warnNo = 0;
    if (warnNo < maxWarn) {
      LOG(WARNING) << "Ignoring hit with time_in_event = " << timeInROF << " ns"
                   << ((++warnNo < maxWarn) ? "" : " (suppressing further warnings)");
    }
    return;
  }
  if (isContinuous()) {
    timeInROF += mCollisionTimeWrtROF;
  }
  // calculate RO Frame for this hit
  if (timeInROF < 0) {
    timeInROF = 0.;
  }
  float tTot = mParams.getSignalShape().getMaxDuration();
  // frame of the hit signal start wrt event ROFrame
  int roFrameRel = int(timeInROF * mParams.getROFrameLengthInv());
  // frame of the hit signal end  wrt event ROFrame: in the triggered mode we read just 1 frame
  UInt_t roFrameRelMax = mParams.isContinuous() ? (timeInROF + tTot) * mParams.getROFrameLengthInv() : roFrameRel;
  int nFrames = roFrameRelMax + 1 - roFrameRel;
  UInt_t roFrameMax = mNewROFrame + roFrameRelMax;
  if (roFrameMax > maxFr) {
    maxFr = roFrameMax; // if signal extends beyond current maxFrame, increase the latter
  }

  // here we start stepping in the depth of the sensor to generate charge diffision
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

  const o2::itsmft::AlpideSimResponse* resp = mParams.getAlpSimResponse();

  // take into account that the AlpideSimResponse depth defintion has different min/max boundaries
  // although the max should coincide with the surface of the epitaxial layer, which in the chip
  // local coordinates has Y = +SensorLayerThickness/2

  xyzLocS.SetY(xyzLocS.Y() + resp->getDepthMax() - Segmentation::SensorLayerThickness / 2.);

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
  o2::MCCompLabel lbl(hit.GetTrackID(), evID, srcID, false);
  auto& chip = mChips[hit.GetDetectorID()];
  auto roFrameAbs = mNewROFrame + roFrameRel;
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
      registerDigits(chip, roFrameAbs, timeInROF, nFrames, rowIS, colIS, nEle, lbl);
    }
  }
}

//________________________________________________________________________________
void Digitizer::registerDigits(ChipDigitsContainer& chip, UInt_t roFrame, float tInROF, int nROF,
                               UShort_t row, UShort_t col, int nEle, o2::MCCompLabel& lbl)
{
  // Register digits for given pixel, accounting for the possible signal contribution to
  // multiple ROFrame. The signal starts at time tInROF wrt the start of provided roFrame
  // In every ROFrame we check the collected signal during strobe

  float tStrobe = mParams.getStrobeDelay() - tInROF; // strobe start wrt signal start
  for (int i = 0; i < nROF; i++) {
    UInt_t roFr = roFrame + i;
    int nEleROF = mParams.getSignalShape().getCollectedCharge(nEle, tStrobe, tStrobe + mParams.getStrobeLength());
    tStrobe += mParams.getROFrameLength(); // for the next ROF

    // discard too small contributions, they have no chance to produce a digit
    if (nEleROF < mParams.getMinChargeToAccount()) {
      continue;
    }
    if (roFr > mEventROFrameMax)
      mEventROFrameMax = roFr;
    if (roFr < mEventROFrameMin)
      mEventROFrameMin = roFr;
    auto key = chip.getOrderingKey(roFr, row, col);
    PreDigit* pd = chip.findDigit(key);
    if (!pd) {
      chip.addDigit(key, roFr, row, col, nEleROF, lbl);
    } else { // there is already a digit at this slot, account as PreDigitExtra contribution
      pd->charge += nEleROF;
      if (pd->labelRef.label == lbl) { // don't store the same label twice
        continue;
      }
      ExtraDig* extra = getExtraDigBuffer(roFr);
      int& nxt = pd->labelRef.next;
      bool skip = false;
      while (nxt >= 0) {
        if ((*extra)[nxt].label == lbl) { // don't store the same label twice
          skip = true;
          break;
        }
        nxt = (*extra)[nxt].next;
      }
      if (skip) {
        continue;
      }
      // new predigit will be added in the end of the chain
      nxt = extra->size();
      extra->emplace_back(lbl);
    }
  }
}
