// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  mHistos.reset();
  mBuffer.reset(new RingBuffer());
  mGeom = Geometry::GetInstance();
}

void PHOSEnergySlot::print() const
{
  LOG(INFO) << "Collected " << mDigits.size() << " CalibDigits";
}

void PHOSEnergySlot::fill(const gsl::span<const Cluster>& clusters, const gsl::span<const CluElement>& cluelements, const gsl::span<const TriggerRecord>& cluTR)
{
  //Scan current list of clusters
  //Fill time, non-linearity and mgg histograms
  //Fill list of re-calibraiable digits
  for (auto& tr : cluTR) {

    //Mark new event
    //First goes new event marker + BC (16 bit), next word orbit (32 bit)
    EventHeader h = {0};
    h.mMarker = 16383;
    h.mBC = tr.getBCData().bc;
    mDigits.push_back(h.mDataWord);
    mDigits.push_back(tr.getBCData().orbit);

    int iclu = 0;
    int firstCluInEvent = tr.getFirstEntry();
    int lastCluInEvent = firstCluInEvent + tr.getNumberOfObjects();

    mBuffer->startNewEvent(); // mark stored clusters to be used for Mixing
    for (int i = firstCluInEvent; i < lastCluInEvent; i++) {
      const Cluster& clu = clusters[i];

      fillTimeMassHisto(clu, cluelements);
      // bool isGood = checkCluster(clu);

      uint32_t firstCE = clu.getFirstCluEl();
      uint32_t lastCE = clu.getLastCluEl();
      for (int idig = firstCE; idig < lastCE; idig++) {
        const CluElement& ce = cluelements[idig];
        short absId = ce.absId;
        //Fill cells from cluster for next iterations
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
          //Normally this is not critical as indexes are used "locally", i.e. are compared to previous/next
          LOG(INFO) << "Too many clusters per event:" << i - firstCluInEvent << ", apply more strict selection; clusters with same indexes will appear";
        }
      }
    }
  }
}
void PHOSEnergySlot::clear()
{
  mHistos.reset();
  mDigits.clear();
}

void PHOSEnergySlot::fillTimeMassHisto(const Cluster& clu, const gsl::span<const CluElement>& cluelements)
{
  // Fill time distributions only for cells in cluster
  uint32_t firstCE = clu.getFirstCluEl();
  uint32_t lastCE = clu.getLastCluEl();

  for (int idig = firstCE; idig < lastCE; idig++) {
    const CluElement& ce = cluelements[idig];
    short absId = ce.absId;
    if (ce.isHG) {
      if (ce.energy > mEminHGTime) {
        mHistos.fill(ETCalibHistos::kTimeHGPerCell, absId, ce.time);
      }
      mHistos.fill(ETCalibHistos::kTimeHGSlewing, ce.time, ce.energy);
    } else {
      if (ce.energy > mEminLGTime) {
        mHistos.fill(ETCalibHistos::kTimeLGPerCell, absId, ce.time);
      }
      mHistos.fill(ETCalibHistos::kTimeLGSlewing, ce.time, ce.energy);
    }
  }

  //Real and Mixed inv mass distributions
  // prepare TLorentsVector
  float posX, posZ;
  clu.getLocalPosition(posX, posZ);
  TVector3 vec3;
  mGeom->local2Global(clu.module(), posX, posZ, vec3);
  vec3 -= mVertex;
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
    if (mBuffer->isCurrentEvent(ip)) { //same (real) event
      if (isGood) {
        mHistos.fill(ETCalibHistos::kReInvMassNonlin, e, sum.M());
      }
      if (sum.Pt() > mPtMin) {
        mHistos.fill(ETCalibHistos::kReInvMassPerCell, absId, sum.M());
      }
    } else { //Mixed
      if (isGood) {
        mHistos.fill(ETCalibHistos::kMiInvMassNonlin, e, sum.M());
      }
      if (sum.Pt() > mPtMin) {
        mHistos.fill(ETCalibHistos::kMiInvMassPerCell, absId, sum.M());
      }
    }
  }

  //Add to list ot partners only if cluster is good
  if (isGood) {
    mBuffer->addEntry(v);
  }
}

bool PHOSEnergySlot::checkCluster(const Cluster& clu)
{
  //First check BadMap
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
  mHistos.reset();
}
PHOSEnergySlot::PHOSEnergySlot(const PHOSEnergySlot& other)
{
  mRunStartTime = other.mRunStartTime;
  mBuffer.reset(new RingBuffer());
  mCalibParams.reset(new CalibParams(*(other.mCalibParams)));
  mBadMap.reset(new BadChannelsMap(*(other.mBadMap)));
  mEvBC = other.mEvBC;
  mEvOrbit = other.mEvOrbit;
  mEvent = 0;
  mPtMin = other.mPtMin;
  mEminHGTime = other.mEminHGTime;
  mEminLGTime = other.mEminLGTime;
  mDigits.clear();
  mHistos.reset();
}

void PHOSEnergyCalibrator::finalizeSlot(Slot& slot)
{

  // Extract results for the single slot
  es* c = slot.getContainer();
  LOG(INFO) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();
  //Add histos
  mHistos.merge(c->getCollectedHistos());
  //Add collected Digits
  auto tmpD = c->getCollectedDigits();
  //Add to list or write to file directly?
  if (!mFout) { //not open yet?
    LOG(INFO) << "Writing CalibDigits to file " << mdigitsfilename.data();
    mFout.reset(TFile::Open(mdigitsfilename.data(), "recreate"));
  }
  int nbites = mFout->WriteObjectAny(&tmpD, "std::vector<uint32_t>", Form("Digits%d", mChank++));
  LOG(INFO) << "Writing " << tmpD.size() << " CalibDigits, wrote " << nbites << "bytes";
  c->clear();
}

Slot& PHOSEnergyCalibrator::emplaceNewSlot(bool front, uint64_t tstart, uint64_t tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<es>());
  slot.getContainer()->setBadMap(*mBadMap);
  slot.getContainer()->setCalibration(*mCalibParams);
  slot.getContainer()->setCuts(mPtMin, mEminHGTime, mEminLGTime);
  return slot;
}

bool PHOSEnergyCalibrator::process(uint64_t tf, const gsl::span<const Cluster>& clusters,
                                   const gsl::span<const CluElement>& cluelements,
                                   const gsl::span<const TriggerRecord>& cluTR)
{
  // process current TF
  //First receive bad map and calibration if not received yet

  auto& slotTF = getSlotForTF(tf);
  slotTF.getContainer()->setRunStartTime(tf);
  slotTF.getContainer()->fill(clusters, cluelements, cluTR);
  return true;
}

void PHOSEnergyCalibrator::endOfStream()
{
}
