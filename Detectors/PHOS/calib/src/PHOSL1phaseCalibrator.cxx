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

#include "PHOSCalibWorkflow/PHOSL1phaseCalibrator.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CCDB/BasicCCDBManager.h"
#include "Framework/ControlService.h"
#include "PHOSBase/Geometry.h"

#include "FairLogger.h"

using namespace o2::phos;

using Slot = o2::calibration::TimeSlot<o2::phos::PHOSL1phaseSlot>;

PHOSL1phaseSlot::PHOSL1phaseSlot()
{
  for (int d = mDDL; d--;) {
    for (int b = 4; b--;) {
      mMean[d][b] = 0.;
      mRMS[d][b] = 0;
    }
    mNorm[d] = 0.;
  }
  for (int bc = 0; bc < 4; ++bc) {
    mQcHisto[bc].fill(0);
  }
}
PHOSL1phaseSlot::PHOSL1phaseSlot(const PHOSL1phaseSlot& other)
{
  mRunStartTime = other.mRunStartTime;
  mBadMap = nullptr;
}
void PHOSL1phaseSlot::print() const
{
  LOG(info) << "Collected statistics per ddl: " << mNorm[1] << " " << mNorm[2] << " " << mNorm[3] << " " << mNorm[4] << " " << mNorm[5] << " " << mNorm[6] << " " << mNorm[7] << " " << mNorm[8] << " " << mNorm[9] << " " << mNorm[10] << " " << mNorm[11] << " " << mNorm[12] << " " << mNorm[13];
}

void PHOSL1phaseSlot::fill(const gsl::span<const Cell>& cells, const gsl::span<const TriggerRecord>& trs)
{
  if (!mBadMap) {
    LOG(info) << "Retrieving BadMap from CCDB";
    auto& ccdbManager = o2::ccdb::BasicCCDBManager::instance();
    ccdbManager.setURL(o2::base::NameConf::getCCDBServer());
    LOG(info) << " set-up CCDB " << o2::base::NameConf::getCCDBServer();
    mBadMap = ccdbManager.get<o2::phos::BadChannelsMap>("PHS/Calib/BadMap");
    LOG(info) << "Retrieving Calibration from CCDB";
    mCalibParams = ccdbManager.get<o2::phos::CalibParams>("PHS/Calib/CalibParams");
  }

  for (auto& tr : trs) {

    int firstCellInEvent = tr.getFirstEntry();
    int lastCellInEvent = firstCellInEvent + tr.getNumberOfObjects();

    for (int i = firstCellInEvent; i < lastCellInEvent; i++) {
      const Cell& c = cells[i];

      if (c.getTRU()) {
        continue;
      }
      short absId = c.getAbsId();
      if (!mBadMap->isChannelGood(absId)) {
        continue;
      }

      float e = 0., t = 0.;
      if (c.getHighGain()) {
        e = c.getEnergy() * mCalibParams->getGain(absId);
        t = c.getTime() - mCalibParams->getHGTimeCalib(absId);
      } else {
        e = c.getEnergy() * mCalibParams->getGain(absId) * mCalibParams->getHGLGRatio(absId);
        t = c.getTime() - mCalibParams->getLGTimeCalib(absId);
      }
      if (e > mEmin && t > mTimeMin && t < mTimeMax) {
        char relid[3];
        o2::phos::Geometry::absToRelNumbering(absId, relid);
        int ddl = (relid[0] - 1) * 4 + (relid[1] - 1) / 16 - 2;
        for (int b = 0; b < 4; b++) {
          int timeshift = tr.getBCData().bc % 4 - b;
          if (timeshift < 0) {
            timeshift += 4;
          }
          float tcorr = t - timeshift * 25e-9;
          mMean[ddl][b] += tcorr;
          mRMS[ddl][b] += tcorr * tcorr;
          int it = (tcorr + 200.e-9) / 4.e-9;
          if (it >= 0 && it < 100) {
            mQcHisto[b][ddl * 100 + it]++;
          }
        }
        mNorm[ddl] += 1.;
      }
    }
  }
}
void PHOSL1phaseSlot::addMeanRms(std::array<std::array<float, 4>, 14>& sumMean,
                                 std::array<std::array<float, 4>, 14>& sumRMS,
                                 std::array<float, 14>& sumNorm)
{
  for (int d = mDDL; d--;) {
    for (int b = 4; b--;) {
      sumMean[d][b] += mMean[d][b];
      sumRMS[d][b] += mRMS[d][b];
    }
    sumNorm[d] += mNorm[d];
  }
}
void PHOSL1phaseSlot::addQcHistos(std::array<unsigned int, 1400> (&sum)[4])
{
  for (int bc = 4; bc--;) {
    for (int it = 1400; it--;) {
      sum[bc][it] += mQcHisto[bc][it];
    }
  }
}
void PHOSL1phaseSlot::merge(const PHOSL1phaseSlot* prev)
{
  for (int d = mDDL; d--;) {
    for (int b = 4; b--;) {
      mMean[d][b] += prev->mMean[d][b];
      mRMS[d][b] += prev->mRMS[d][b];
    }
    mNorm[d] += prev->mNorm[d];
  }
  for (int bc = 4; bc--;) {
    for (int it = 1400; it--;) {
      mQcHisto[bc][it] += prev->mQcHisto[bc][it];
    }
  }
}
void PHOSL1phaseSlot::clear()
{
  for (int d = mDDL; d--;) {
    for (int b = 4; b--;) {
      mMean[d][b] = 0.;
      mRMS[d][b] = 0;
    }
    mNorm[d] = 0.;
  }
  for (int bc = 4; bc--;) {
    mQcHisto[bc].fill(0);
  }
}

//==============================================================

PHOSL1phaseCalibrator::PHOSL1phaseCalibrator()
{
  for (int d = mDDL; d--;) {
    for (int b = 4; b--;) {
      mMean[d][b] = 0.;
      mRMS[d][b] = 0;
    }
    mNorm[d] = 0.;
  }
  for (int bc = 0; bc < 4; ++bc) {
    mQcHisto[bc].fill(0);
  }
}
bool PHOSL1phaseCalibrator::hasEnoughData(const Slot& slot) const
{
  // otherwize will be merged with next Slot
  return true;
}
void PHOSL1phaseCalibrator::initOutput()
{
}
void PHOSL1phaseCalibrator::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  PHOSL1phaseSlot* ct = slot.getContainer();
  LOG(info) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();
  ct->addMeanRms(mMean, mRMS, mNorm);
  ct->addQcHistos(mQcHisto);
  ct->clear();
}

Slot& PHOSL1phaseCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{

  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<PHOSL1phaseSlot>());
  return slot;
}

bool PHOSL1phaseCalibrator::process(TFType tf, const gsl::span<const Cell>& cells, const gsl::span<const TriggerRecord>& tr)
{

  // process current TF
  auto& slotTF = getSlotForTF(tf);
  slotTF.getContainer()->setRunStartTime(tf);
  slotTF.getContainer()->fill(cells, tr);

  return true;
}
void PHOSL1phaseCalibrator::endOfStream()
{

  auto& cont = getSlots();
  for (auto& slot : cont) {
    finalizeSlot(slot);
  }

  // Calculate L1phases: First evaluate RMS and mean for 4 possible values of shift b
  // then find the smallest RMS, if two RMS are close, choose those b which produces mean closer to 0
  mL1phase = 0;
  for (int d = 0; d < mDDL; d++) {
    int iMinRMS = 0, iMinMean = 0;
    if (mNorm[d] == 0) { // leave phase=0
      continue;
    }
    float minMean = 0, minRMS = 0, subminRMS = 0;
    for (int b = 0; b < 4; b++) {
      mMean[d][b] /= mNorm[d];
      mRMS[d][b] /= mNorm[d];
      mRMS[d][b] -= mMean[d][b] * mMean[d][b];
      mMean[d][b] = abs(mMean[d][b]);
      if (b == 0) {
        minRMS = mRMS[d][b];
        minMean = mMean[d][b];
      } else {
        if (minRMS > mRMS[d][b]) {
          subminRMS = minRMS;
          minRMS = mRMS[d][b];
          iMinRMS = b;
        } else {
          if (subminRMS == 0) {
            subminRMS = mRMS[d][b];
          } else {
            if (mRMS[d][b] < subminRMS) {
              subminRMS = mRMS[d][b];
            }
          }
        }
        if (minMean > mMean[d][b]) {
          minMean = mMean[d][b];
          iMinMean = b;
        }
      }
    }

    // select b for this ddl
    int bestB = 0;
    const float eps = 1.e-2;                 // rel. accuracy of RMS calculation
    if (minRMS < subminRMS - eps * minRMS) { // clear minimum
      bestB = iMinRMS;
    } else { // RMS are slimilar, chose closer to zero Mean
      bestB = iMinMean;
    }
    if (bestB != 0) { // copy the histogram content to final histo
      for (int it = 100; it--;) {
        mQcHisto[0][d * 100 + it] = mQcHisto[bestB][d * 100 + it];
      }
    }
    mL1phase |= (bestB << (2 * d));
  }
  LOG(info) << "Calculated L1phase=" << mL1phase;
}