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

/// \file Digitizer.cxx

#include "DataFormatsITSMFT/Digit.h"
#include "TRKBase/SegmentationChip.h"
#include "TRKSimulation/DPLDigitizerParam.h"
#include "TRKSimulation/TRKLayer.h"
#include "TRKSimulation/Digitizer.h"
#include "DetectorsRaw/HBFUtils.h"

#include <TRandom.h>
// #include <climits>
#include <vector>
#include <iostream>
#include <numeric>
#include <fairlogger/Logger.h> // for LOG

using o2::itsmft::Digit;
using o2::trk::Hit;
using Segmentation = o2::trk::SegmentationChip;

using namespace o2::trk;
using namespace o2::itsmft;
// using namespace o2::base;
//_______________________________________________________________________
void Digitizer::init()
{
  LOG(info) << "Initializing digitizer";
  mNumberOfChips = mGeometry->getNumberOfChips();
  mChips.resize(mNumberOfChips); /// temporary, to not make it crash
  for (int i = mNumberOfChips; i--;) {
    mChips[i].setChipIndex(i);
    if (mNoiseMap) {
      mChips[i].setNoiseMap(mNoiseMap);
    }
    if (mDeadChanMap) {
      mChips[i].disable(mDeadChanMap->isFullChipMasked(i));
      mChips[i].setDeadChanMap(mDeadChanMap);
    }
  }

  // setting the correct response function (for the moment, for both VD and MLOT the same response function is used)
  mChipSimResp = mParams.getAlpSimResponse();
  mChipSimRespVD = mChipSimResp;   /// for the moment considering the same response
  mChipSimRespMLOT = mChipSimResp; /// for the moment considering the same response

  /// setting scale factors to adapt to the APTS response function (adjusting pitch and Y shift)
  // TODO: adjust Y shift when the geometry is improved
  LOG(info) << " Depth max VD: " << mChipSimRespVD->getDepthMax();
  LOG(info) << " Depth min VD: " << mChipSimRespVD->getDepthMin();

  LOG(info) << " Depth max MLOT: " << mChipSimRespMLOT->getDepthMax();
  LOG(info) << " Depth min MLOT: " << mChipSimRespMLOT->getDepthMin();

  float thicknessVD = 0.0095;                                            // cm --- hardcoded based on geometry currently present
  float thicknessMLOT = o2::trk::SegmentationChip::SiliconThicknessMLOT; // 0.01 cm = 100 um --- based on geometry currently present

  LOG(info) << "Using response name: " << mRespName;
  mSimRespOrientation = false;

  if (mRespName == "APTS") { // default
    mSimRespVDScaleX = o2::trk::constants::apts::pitchX / o2::trk::SegmentationChip::PitchRowVD;
    mSimRespVDScaleZ = o2::trk::constants::apts::pitchZ / o2::trk::SegmentationChip::PitchColVD;
    mSimRespVDShift = mChipSimRespVD->getDepthMax(); // the curved, rescaled, sensors have a width from 0 to -45. Must add ~10 um (= max depth) to match the APTS response.
    mSimRespMLOTScaleX = o2::trk::constants::apts::pitchX / o2::trk::SegmentationChip::PitchRowMLOT;
    mSimRespMLOTScaleZ = o2::trk::constants::apts::pitchZ / o2::trk::SegmentationChip::PitchColMLOT;
    mSimRespOrientation = true; /// APTS response function is flipped along x wrt the ones of ALPIDE and ALICE3
  } else if (mRespName == "ALICE3") {
    mSimRespVDScaleX = o2::trk::constants::alice3resp::pitchX / o2::trk::SegmentationChip::PitchRowVD;
    mSimRespVDScaleZ = o2::trk::constants::alice3resp::pitchZ / o2::trk::SegmentationChip::PitchColVD;
    mSimRespVDShift = mChipSimRespVD->getDepthMax(); // the curved, rescaled, sensors have a width from 0 to -95 um. Must align the start of epi layer with the response function.
    mSimRespMLOTScaleX = o2::trk::constants::alice3resp::pitchX / o2::trk::SegmentationChip::PitchRowMLOT;
    mSimRespMLOTScaleZ = o2::trk::constants::alice3resp::pitchZ / o2::trk::SegmentationChip::PitchColMLOT;
  } else {
    LOG(fatal) << "Unknown response name: " << mRespName;
  }

  mSimRespMLOTShift = mChipSimRespMLOT->getDepthMax() - thicknessMLOT / 2.f; // the shift should be done considering the rescaling done to adapt to the wrong silicon thickness. TODO: remove the scaling factor for the depth when the silicon thickness match the simulated response

  // importing the parameters from DPLDigitizerParam.h
  auto& dOptTRK = DPLDigitizerParam<o2::detectors::DetID::TRK>::Instance();

  LOGP(info, "TRK Digitizer is initialised.");
  mParams.print();
  LOGP(info, "VD shift = {}  ; ML/OT shift = {} = {} - {}", mSimRespVDShift, mSimRespMLOTShift, mChipSimRespMLOT->getDepthMax(), thicknessMLOT / 2.f);
  LOGP(info, "VD pixel scale on x = {} ; z = {}", mSimRespVDScaleX, mSimRespVDScaleZ);
  LOGP(info, "ML/OT pixel scale on x = {} ; z = {}", mSimRespMLOTScaleX, mSimRespMLOTScaleZ);
  LOGP(info, "Response orientation: {}", mSimRespOrientation ? "flipped" : "normal");

  mIRFirstSampledTF = o2::raw::HBFUtils::Instance().getFirstSampledTFIR();
}

const o2::trk::ChipSimResponse* Digitizer::getChipResponse(int chipID)
{
  if (mGeometry->getSubDetID(chipID) == 0) { /// VD
    return mChipSimRespVD;
  }

  else if (mGeometry->getSubDetID(chipID) == 1) { /// ML/OT
    return mChipSimRespMLOT;
  }
  return nullptr;
};

//_______________________________________________________________________
void Digitizer::process(const std::vector<Hit>* hits, int evID, int srcID)
{
  // digitize single event, the time must have been set beforehand

  LOG(info) << " Digitizing " << mGeometry->getName() << " (ID: " << mGeometry->getDetID()
            << ") hits of event " << evID << " from source " << srcID
            << " at time " << mEventTime.getTimeNS() << " ROFrame = " << mNewROFrame
            << " cont.mode: " << isContinuous()
            << " Min/Max ROFrames " << mROFrameMin << "/" << mROFrameMax;

  std::cout << "Printing segmentation info: " << std::endl;
  SegmentationChip::Print();

  // // is there something to flush ?
  if (mNewROFrame > mROFrameMin) {
    fillOutputContainer(mNewROFrame - 1); // flush out all frames preceding the new one
  }

  int nHits = hits->size();
  std::vector<int> hitIdx(nHits);
  std::iota(std::begin(hitIdx), std::end(hitIdx), 0);
  // sort hits to improve memory access
  std::sort(hitIdx.begin(), hitIdx.end(),
            [hits](auto lhs, auto rhs) {
              return (*hits)[lhs].GetDetectorID() < (*hits)[rhs].GetDetectorID();
            });
  LOG(info) << "Processing " << nHits << " hits";
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
  LOG(info) << "Setting event time to " << irt.getTimeNS() << " ns after orbit 0 bc 0";
  // assign event time in ns
  mEventTime = irt;
  if (!mParams.isContinuous()) {
    mROFrameMin = 0; // in triggered mode reset the frame counters
    mROFrameMax = 0;
  }
  // RO frame corresponding to provided time
  mCollisionTimeWrtROF = mEventTime.timeInBCNS; // in triggered mode the ROF starts at BC (is there a delay?)
  if (mParams.isContinuous()) {
    auto nbc = mEventTime.differenceInBC(mIRFirstSampledTF);

    if (mCollisionTimeWrtROF < 0 && nbc > 0) {
      nbc--;
    }

    mNewROFrame = nbc / mParams.getROFrameLengthInBC();

    LOG(debug) << " NewROFrame " << mNewROFrame << " = " << nbc << "/" << mParams.getROFrameLengthInBC() << " (nbc/mParams.getROFrameLengthInBC()";

    // in continuous mode depends on starts of periodic readout frame
    mCollisionTimeWrtROF += (nbc % mParams.getROFrameLengthInBC()) * o2::constants::lhc::LHCBunchSpacingNS;
  } else {
    mNewROFrame = 0;
  }

  if (mNewROFrame < mROFrameMin) {
    LOG(error) << "New ROFrame " << mNewROFrame << " (" << irt << ") precedes currently cashed " << mROFrameMin;
    throw std::runtime_error("deduced ROFrame precedes already processed one");
  }

  if (mParams.isContinuous() && mROFrameMax < mNewROFrame) {
    mROFrameMax = mNewROFrame - 1; // all frames up to this are finished
  }
}

//_______________________________________________________________________
void Digitizer::fillOutputContainer(uint32_t frameLast)
{
  // // fill output with digits from min.cached up to requested frame, generating the noise beforehand
  if (frameLast > mROFrameMax) {
    frameLast = mROFrameMax;
  }
  // // make sure all buffers for extra digits are created up to the maxFrame
  getExtraDigBuffer(mROFrameMax);
  LOG(info) << "Filling " << mGeometry->getName() << " digits output for RO frames " << mROFrameMin << ":"
            << frameLast;

  o2::itsmft::ROFRecord rcROF; /// using temporarly itsmft::ROFRecord

  // we have to write chips in RO increasing order, therefore have to loop over the frames here
  for (; mROFrameMin <= frameLast; mROFrameMin++) {
    rcROF.setROFrame(mROFrameMin);
    rcROF.setFirstEntry(mDigits->size()); // start of current ROF in digits

    auto& extra = *(mExtraBuff.front().get());
    for (auto& chip : mChips) {
      if (chip.isDisabled()) {
        continue;
      }
      chip.addNoise(mROFrameMin, mROFrameMin, &mParams, mGeometry->getSubDetID(chip.getChipIndex()), mGeometry->getLayer(chip.getChipIndex())); /// TODO: add noise
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
          LOG(debug) << "Adding digit ID: " << digID << " with chipID: " << chip.getChipIndex() << ", row: " << preDig.row << ", col: " << preDig.col << ", charge: " << preDig.charge;
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
      rcROF.getBCData().setFromLong(mIRFirstSampledTF.toLong() + mROFrameMin * mParams.getROFrameLengthInBC());
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
void Digitizer::processHit(const o2::trk::Hit& hit, uint32_t& maxFr, int evID, int srcID)
{
  int chipID = hit.GetDetectorID(); //// the chip ID at the moment is not referred to the chip but to a wider detector element (e.g. quarter of layer or disk in VD, stave in ML, half stave in OT)
  int subDetID = mGeometry->getSubDetID(chipID);

  int layer = mGeometry->getLayer(chipID);
  int disk = mGeometry->getDisk(chipID);

  if (disk != -1) {
    LOG(debug) << "Skipping disk " << disk;
    return; // skipping hits on disks for the moment
  }

  LOG(debug) << "Processing hit for chip " << chipID;
  auto& chip = mChips[chipID];
  if (chip.isDisabled()) {
    LOG(debug) << "Skipping disabled chip " << chipID;
    return;
  }
  float timeInROF = hit.GetTime() * sec2ns;
  LOG(debug) << "Hit time: " << timeInROF << " ns";
  if (timeInROF > 20e3) {
    const int maxWarn = 10;
    static int warnNo = 0;
    if (warnNo < maxWarn) {
      LOG(warning) << "Ignoring hit with time_in_event = " << timeInROF << " ns"
                   << ((++warnNo < maxWarn) ? "" : " (suppressing further warnings)");
    }
    return;
  }
  if (isContinuous()) {
    timeInROF += mCollisionTimeWrtROF;
  }
  if (mIsBeforeFirstRO && timeInROF < 0) {
    // disregard this hit because it comes from an event byefore readout starts and it does not effect this RO
    LOG(debug) << "Ignoring hit with timeInROF = " << timeInROF;
    return;
  }

  // calculate RO Frame for this hit
  if (timeInROF < 0) {
    timeInROF = 0.;
  }
  float tTot = mParams.getSignalShape().getMaxDuration();
  // frame of the hit signal start wrt event ROFrame
  int roFrameRel = int(timeInROF * mParams.getROFrameLengthInv());
  // frame of the hit signal end  wrt event ROFrame: in the triggered mode we read just 1 frame
  uint32_t roFrameRelMax = mParams.isContinuous() ? (timeInROF + tTot) * mParams.getROFrameLengthInv() : roFrameRel;
  int nFrames = roFrameRelMax + 1 - roFrameRel;
  uint32_t roFrameMax = mNewROFrame + roFrameRelMax;
  if (roFrameMax > maxFr) {
    maxFr = roFrameMax; // if signal extends beyond current maxFrame, increase the latter
  }

  // here we start stepping in the depth of the sensor to generate charge diffusion
  float nStepsInv = mParams.getNSimStepsInv();
  int nSteps = mParams.getNSimSteps();

  const auto& matrix = mGeometry->getMatrixL2G(hit.GetDetectorID());
  // matrix.print();

  /// transorm from the global detector coordinates to the local detector coordinates
  math_utils::Vector3D<float> xyzLocS(matrix ^ (hit.GetPosStart())); // start position in sensor frame
  math_utils::Vector3D<float> xyzLocE(matrix ^ (hit.GetPos()));      // end position in sensor frame

  if (subDetID == 0) { // VD - need to take into account for the curved layers. TODO: consider the disks
    // transform the point on the curved surface to a flat one
    math_utils::Vector2D<float> xyFlatS = Segmentation::curvedToFlat(layer, xyzLocS.x(), xyzLocS.y());
    math_utils::Vector2D<float> xyFlatE = Segmentation::curvedToFlat(layer, xyzLocE.x(), xyzLocE.y());
    LOG(debug) << "Called curved to flat: " << xyzLocS.x() << " -> " << xyFlatS.x() << ", " << xyzLocS.y() << " -> " << xyFlatS.y();
    // update the local coordinates with the flattened ones
    xyzLocS.SetXYZ(xyFlatS.x(), xyFlatS.y(), xyzLocS.Z());
    xyzLocE.SetXYZ(xyFlatE.x(), xyFlatE.y(), xyzLocE.Z());
  }

  // std::cout<<"Printing example of point in 0.35 0.35 0 in global frame: "<<std::endl;
  // math_utils::Point3D<float> examplehitGlob(0.35, 0.35, 0);
  // math_utils::Vector3D<float> exampleLoc(matrix ^ (examplehitGlob)); // start position in sensor frame
  // std::cout<< "Example hit in local frame: " << exampleLoc << std::endl;
  // std::cout<<"Going back to glob coordinates: " << (matrix * exampleLoc) << std::endl;

  math_utils::Vector3D<float> step(xyzLocE);
  step -= xyzLocS;
  step *= nStepsInv; // position increment at each step
  // the electrons will injected in the middle of each step
  // starting from the middle of the first step
  math_utils::Vector3D<float> stepH(step * 0.5);
  xyzLocS += stepH;
  xyzLocE -= stepH;

  LOG(debug) << "Step into the sensitive volume: " << step << ".  Number of steps: " << nSteps;
  int rowS = -1, colS = -1, rowE = -1, colE = -1, nSkip = 0;

  /// here it is the control whether the hit is in the sensitive matrix based on the segmentation
  // get entrance pixel row and col
  while (!Segmentation::localToDetector(xyzLocS.X(), xyzLocS.Z(), rowS, colS, subDetID, layer, disk)) { // guard-ring ?
    if (++nSkip >= nSteps) {
      LOG(debug) << "Did not enter to sensitive matrix, " << nSkip << " >= " << nSteps;
      return; // did not enter to sensitive matrix
    }
    xyzLocS += step;
  }

  // get exit pixel row and col
  while (!Segmentation::localToDetector(xyzLocE.X(), xyzLocE.Z(), rowE, colE, subDetID, layer, disk)) { /// for the moment chipID = bigger element
    if (++nSkip >= nSteps) {
      LOG(debug) << "Did not enter to sensitive matrix, " << nSkip << " >= " << nSteps;
      return; // did not enter to sensitive matrix
    }
    xyzLocE -= step;
  }

  int nCols = getNCols(subDetID, layer);
  int nRows = getNRows(subDetID, layer);

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
  if (rowE >= nRows) {
    rowE = nRows - 1;
  }
  colS -= AlpideRespSimMat::NPix / 2;
  colE += AlpideRespSimMat::NPix / 2;
  if (colS < 0) {
    colS = 0;
  }
  if (colE >= nCols) {
    colE = nCols - 1;
  }
  int rowSpan = rowE - rowS + 1, colSpan = colE - colS + 1; // size of plaquet where some response is expected

  float respMatrix[rowSpan][colSpan]; // response accumulated here
  std::fill(&respMatrix[0][0], &respMatrix[0][0] + rowSpan * colSpan, 0.f);

  float nElectrons = hit.GetEnergyLoss() * mParams.getEnergyToNElectrons(); // total number of deposited electrons
  nElectrons *= nStepsInv;                                                  // N electrons injected per step
  if (nSkip) {
    nSteps -= nSkip;
  }

  int rowPrev = -1, colPrev = -1, row, col;
  float cRowPix = 0.f, cColPix = 0.f; // local coordinate of the current pixel center

  const o2::trk::ChipSimResponse* resp = getChipResponse(chipID);
  // std::cout << "Printing chip response:" << std::endl;
  // resp->print();

  // take into account that the ChipSimResponse depth defintion has different min/max boundaries
  // although the max should coincide with the surface of the epitaxial layer, which in the chip
  // local coordinates has Y = +SensorLayerThickness/2
  // LOG(info)<<"SubdetID = " << subDetID<< " shift: "<<mSimRespVDShift<<" or "<<mSimRespMLOTShift;
  // LOG(info)<< " Before shift: S = " << xyzLocS.Y()*1e4 << "  E = " << xyzLocE.Y()*1e4;
  xyzLocS.SetY(xyzLocS.Y() + ((subDetID == 0) ? mSimRespVDShift : mSimRespMLOTShift));
  // LOG(info)<< " After shift: S = " << xyzLocS.Y()*1e4 << "  E = " << xyzLocE.Y()*1e4;

  // collect charge in every pixel which might be affected by the hit
  for (int iStep = nSteps; iStep--;) {
    // Get the pixel ID
    Segmentation::localToDetector(xyzLocS.X(), xyzLocS.Z(), row, col, subDetID, layer, disk);
    if (row != rowPrev || col != colPrev) { // update pixel and coordinates of its center
      if (!Segmentation::detectorToLocal(row, col, cRowPix, cColPix, subDetID, layer, disk)) {
        continue; // should not happen
      }
      rowPrev = row;
      colPrev = col;
    }
    bool flipCol = false, flipRow = false;
    // note that response needs coordinates along column row (locX) (locZ) then depth (locY)
    float rowMax{}, colMax{};
    const AlpideRespSimMat* rspmat{nullptr};
    if (subDetID == 0) { // VD
      rowMax = 0.5f * Segmentation::PitchRowVD * mSimRespVDScaleX;
      colMax = 0.5f * Segmentation::PitchColVD * mSimRespVDScaleZ;
      rspmat = resp->getResponse(mSimRespVDScaleX * (xyzLocS.X() - cRowPix), mSimRespVDScaleZ * (xyzLocS.Z() - cColPix), xyzLocS.Y(), flipRow, flipCol, rowMax, colMax);
    } else { // ML/OT
      rowMax = 0.5f * Segmentation::PitchRowMLOT * mSimRespMLOTScaleX;
      colMax = 0.5f * Segmentation::PitchColMLOT * mSimRespMLOTScaleZ;
      rspmat = resp->getResponse(mSimRespMLOTScaleX * (xyzLocS.X() - cRowPix), mSimRespMLOTScaleZ * (xyzLocS.Z() - cColPix), xyzLocS.Y(), flipRow, flipCol, rowMax, colMax);
    }

    xyzLocS += step;

    if (rspmat == nullptr) {
      LOG(debug) << "Error in rspmat for step " << iStep << " / " << nSteps;
      continue;
    }
    // LOG(info) << "rspmat valid! for step " << iStep << " / " << nSteps << ", (row,col) = (" << row << "," << col << ")";
    // LOG(info) << "rspmat valid! for step " << iStep << " / " << nSteps << " Y= " << xyzLocS.Y()*1e4 << " , (row,col) = (" << row << "," << col << ")";
    // rspmat->print(); // print the response matrix for debugging

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
        respMatrix[rowDest][colDest] += rspmat->getValue(irow, icol, mSimRespOrientation ? !flipRow : flipRow, flipCol);
      }
    }
  }

  // fire the pixels assuming Poisson(n_response_electrons)
  o2::MCCompLabel lbl(hit.GetTrackID(), evID, srcID, false);
  auto roFrameAbs = mNewROFrame + roFrameRel;
  LOG(debug) << "\nSpanning through rows and columns; rowspan = " << rowSpan << " colspan = " << colSpan << " = " << colE << " - " << colS << " +1 ";
  for (int irow = rowSpan; irow--;) {          // irow ranging from 4 to 0
    uint16_t rowIS = irow + rowS;              // row distant irow from the row of the hit start
    for (int icol = colSpan; icol--;) {        // icol ranging from 4 to 0
      float nEleResp = respMatrix[irow][icol]; // value of the probability of the response in this pixel
      if (nEleResp <= 1.e-36) {
        continue;
      }
      LOG(debug) << "nEleResp: value " << nEleResp << " for pixel " << irow << " " << icol;
      int nEle = gRandom->Poisson(nElectrons * nEleResp); // total charge in given pixel = number of electrons generated in the hit multiplied by the probability of being detected in their position
      LOG(debug) << "Charge detected in the pixel: " << nEle << " for pixel " << irow << " " << icol;
      // ignore charge which have no chance to fire the pixel
      if (nEle < mParams.getMinChargeToAccount()) { /// TODO: substitute with the threshold?
        LOG(debug) << "Ignoring pixel with nEle = " << nEle << " < min charge to account "
                   << mParams.getMinChargeToAccount() << " for pixel " << irow << " " << icol;
        continue;
      }

      uint16_t colIS = icol + colS; // col distant icol from the col of the hit start
      if (mNoiseMap && mNoiseMap->isNoisy(chipID, rowIS, colIS)) {
        continue;
      }
      if (mDeadChanMap && mDeadChanMap->isNoisy(chipID, rowIS, colIS)) {
        continue;
      }
      registerDigits(chip, roFrameAbs, timeInROF, nFrames, rowIS, colIS, nEle, lbl);
    }
  }
}

//________________________________________________________________________________
void Digitizer::registerDigits(o2::trk::ChipDigitsContainer& chip, uint32_t roFrame, float tInROF, int nROF,
                               uint16_t row, uint16_t col, int nEle, o2::MCCompLabel& lbl)
{
  // Register digits for given pixel, accounting for the possible signal contribution to
  // multiple ROFrame. The signal starts at time tInROF wrt the start of provided roFrame
  // In every ROFrame we check the collected signal during strobe
  LOG(debug) << "Registering digits for chip " << chip.getChipIndex() << " at ROFrame " << roFrame
             << " row " << row << " col " << col << " nEle " << nEle << " label " << lbl;
  float tStrobe = mParams.getStrobeDelay() - tInROF; // strobe start wrt signal start
  for (int i = 0; i < nROF; i++) {                   // loop on all the ROFs occupied by the same signal to calculate the charge accumulated in that ROF
    uint32_t roFr = roFrame + i;
    int nEleROF = mParams.getSignalShape().getCollectedCharge(nEle, tStrobe, tStrobe + mParams.getStrobeLength());
    tStrobe += mParams.getROFrameLength(); // for the next ROF

    // discard too small contributions, they have no chance to produce a digit
    if (nEleROF < mParams.getMinChargeToAccount()) { /// use threshold instead?
      continue;
    }
    if (roFr > mEventROFrameMax) {
      mEventROFrameMax = roFr;
    }
    if (roFr < mEventROFrameMin) {
      mEventROFrameMin = roFr;
    }
    auto key = chip.getOrderingKey(roFr, row, col);
    o2::itsmft::PreDigit* pd = chip.findDigit(key);
    if (!pd) {
      chip.addDigit(key, roFr, row, col, nEleROF, lbl);
      LOG(debug) << "Added digit with key: " << key << "  ROF: " << roFr << "  row: " << row << "  col: " << col << "  charge: " << nEleROF;
    } else { // there is already a digit at this slot, account as PreDigitExtra contribution
      LOG(debug) << "Added to pre-digit with key: " << key << "  ROF: " << roFr << "  row: " << row << "  col: " << col << "  charge: " << nEleROF;
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
