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

#include "PHOSCalibWorkflow/PHOSEnergyCalibrator.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/ControlService.h"

#include "FairLogger.h"
#include <fstream> // std::ifstream

using namespace o2::phos;

PHOSEnergySlot::PHOSEnergySlot()
{
  mHistos = std::make_unique<ETCalibHistos>();
  mBuffer = std::make_unique<RingBuffer>();
  mGeom = Geometry::GetInstance();
}
PHOSEnergySlot::PHOSEnergySlot(const PHOSEnergySlot& other)
{
  mRunStartTime = other.mRunStartTime;
  mBuffer = std::make_unique<RingBuffer>();
  mEvBC = other.mEvBC;
  mEvOrbit = other.mEvOrbit;
  mEvent = 0;
  mPtMin = other.mPtMin;
  mEminHGTime = other.mEminHGTime;
  mEminLGTime = other.mEminLGTime;
  mDigits.clear();
  mHistos = std::make_unique<ETCalibHistos>();
}

void PHOSEnergySlot::print() const
{
  LOG(info) << "Collected " << mDigits.size() << " CalibDigits";
}

void PHOSEnergySlot::fill(const gsl::span<const Cluster>& clusters, const gsl::span<const CluElement>& cluelements, const gsl::span<const TriggerRecord>& cluTR)
{
  // Scan current list of clusters
  // Fill time, non-linearity and mgg histograms
  // Fill list of re-calibraiable digits
  mDigits.clear();
  for (auto& tr : cluTR) {
    // Mark new event
    // First goes new event marker + BC (16 bit), next word orbit (32 bit)
    EventHeader h = {0};
    h.mMarker = 16383;
    h.mBC = tr.getBCData().bc;
    mDigits.push_back(h.mDataWord);
    mDigits.push_back(tr.getBCData().orbit);

    int firstCluInEvent = tr.getFirstEntry();
    int lastCluInEvent = firstCluInEvent + tr.getNumberOfObjects();

    mBuffer->startNewEvent(); // mark stored clusters to be used for Mixing
    for (int i = firstCluInEvent; i < lastCluInEvent; i++) {
      const Cluster& clu = clusters[i];
      if (clu.getEnergy() < mClusterEmin) { // There was problem in unfolding and cluster parameters not calculated
        continue;
      }
      fillTimeMassHisto(clu, cluelements);

      uint32_t firstCE = clu.getFirstCluEl();
      uint32_t lastCE = clu.getLastCluEl();
      for (uint32_t idig = firstCE; idig < lastCE; idig++) {
        const CluElement& ce = cluelements[idig];
        if (ce.energy < mDigitEmin) {
          continue;
        }
        short absId = ce.absId;
        // Fill cells from cluster for next iterations
        short adcCounts = ce.energy / mCalibParams->getGain(absId);
        // Need to chale LG gain too to fit dynamic range
        if (!ce.isHG) {
          adcCounts /= mCalibParams->getHGLGRatio(absId);
        }
        CalibDigit d = {0};
        d.mAddress = absId;
        d.mAdcAmp = adcCounts;
        d.mHgLg = ce.isHG;
        d.mCluster = (i - firstCluInEvent) % kMaxCluInEvent;
        mDigits.push_back(d.mDataWord);
        if (i - firstCluInEvent > kMaxCluInEvent) {
          // Normally this is not critical as indexes are used "locally", i.e. are compared to previous/next
          LOG(important) << "Too many clusters per event:" << i - firstCluInEvent << ", apply more strict selection; clusters with same indexes will appear";
        }
      }
    }
  }
}
void PHOSEnergySlot::clear()
{
  mHistos->reset();
  mDigits.clear();
}

void PHOSEnergySlot::fillTimeMassHisto(const Cluster& clu, const gsl::span<const CluElement>& cluelements)
{
  // Fill time distributions only for cells in cluster
  uint32_t firstCE = clu.getFirstCluEl();
  uint32_t lastCE = clu.getLastCluEl();

  for (uint32_t idig = firstCE; idig < lastCE; idig++) {
    const CluElement& ce = cluelements[idig];
    short absId = ce.absId;
    if (ce.isHG) {
      if (ce.energy > mEminHGTime) {
        mHistos->fill(ETCalibHistos::kTimeHGPerCell, absId, ce.time);
      }
      mHistos->fill(ETCalibHistos::kTimeHGSlewing, ce.time, ce.energy);
    } else {
      if (ce.energy > mEminLGTime) {
        mHistos->fill(ETCalibHistos::kTimeLGPerCell, absId, ce.time);
      }
      mHistos->fill(ETCalibHistos::kTimeLGSlewing, ce.time, ce.energy);
    }
  }

  // Real and Mixed inv mass distributions
  //  prepare TLorentsVector
  float posX, posZ;
  clu.getLocalPosition(posX, posZ);
  TVector3 vec3;
  mGeom->local2Global(clu.module(), posX, posZ, vec3);
  float e = clu.getEnergy();
  short absId;
  mGeom->relPosToAbsId(clu.module(), posX, posZ, absId);

  vec3 *= 1. / vec3.Mag();
  TLorentzVector v(vec3.X() * e, vec3.Y() * e, vec3.Z() * e, e);
  // Fill calibration histograms for all cells, even bad, but partners in inv, mass should be good
  bool isGood = checkCluster(clu);
  for (short ip = mBuffer->size(); ip--;) {
    const TLorentzVector& vp = mBuffer->getEntry(ip);
    TLorentzVector sum = v + vp;
    if (mBuffer->isCurrentEvent(ip)) { // same (real) event
      if (isGood) {
        mHistos->fill(ETCalibHistos::kReInvMassNonlin, e, sum.M());
      }
      if (sum.Pt() > mPtMin) {
        mHistos->fill(ETCalibHistos::kReInvMassPerCell, absId, sum.M());
      }
    } else { // Mixed
      if (isGood) {
        mHistos->fill(ETCalibHistos::kMiInvMassNonlin, e, sum.M());
      }
      if (sum.Pt() > mPtMin) {
        mHistos->fill(ETCalibHistos::kMiInvMassPerCell, absId, sum.M());
      }
    }
  }

  // Add to list ot partners only if cluster is good
  if (isGood) {
    mBuffer->addEntry(v);
  }
}

bool PHOSEnergySlot::checkCluster(const Cluster& clu)
{
  // First check BadMap
  float posX, posZ;
  clu.getLocalPosition(posX, posZ);
  short absId;
  Geometry::relPosToAbsId(clu.module(), posX, posZ, absId);
  if (!mBadMap->isChannelGood(absId)) {
    return false;
  }

  return (clu.getEnergy() > 0.3 && clu.getMultiplicity() > 1);
}

//==================================================

using es = o2::phos::PHOSEnergySlot;
using Slot = o2::calibration::TimeSlot<o2::phos::PHOSEnergySlot>;
PHOSEnergyCalibrator::PHOSEnergyCalibrator()
{
  // create final histos
  mHistos = std::make_unique<ETCalibHistos>();
}

void PHOSEnergyCalibrator::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  es* c = slot.getContainer();
  LOG(debug) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();
  // Add histos
  mHistos->merge(c->getCollectedHistos());
}

Slot& PHOSEnergyCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<es>());
  slot.getContainer()->setBadMap(mBadMap);
  slot.getContainer()->setCalibration(mCalibParams);
  slot.getContainer()->setCuts(mPtMin, mEminHGTime, mEminLGTime, mDigitEmin, mClusterEmin);
  return slot;
}

bool PHOSEnergyCalibrator::process(uint64_t tf, const gsl::span<const Cluster>& clusters,
                                   const gsl::span<const CluElement>& cluelements,
                                   const gsl::span<const TriggerRecord>& cluTR,
                                   std::vector<uint32_t>& outputDigits)
{
  // process current TF
  // First receive bad map and calibration if not received yet
  auto& slotTF = getSlotForTF(tf);
  slotTF.getContainer()->setRunStartTime(tf);
  slotTF.getContainer()->fill(clusters, cluelements, cluTR);
  // Add collected Digits
  auto tmpD = slotTF.getContainer()->getCollectedDigits();
  outputDigits.insert(outputDigits.end(), tmpD.begin(), tmpD.end());
  return true;
}

void PHOSEnergyCalibrator::endOfStream()
{
}
