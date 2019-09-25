// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ZDCSimulation/Digitizer.h"
#include "ZDCSimulation/Digit.h"
#include "ZDCSimulation/Hit.h"
#include <FairLogger.h>
#include <vector>

using namespace o2::zdc;

ClassImp(Digitizer);

// this will process hits and fill the digit vector with digits which are finalized
void Digitizer::process(const std::vector<o2::zdc::Hit>& hits,
                        std::vector<o2::zdc::Digit>& digits,
                        o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>& labels)
{
  // loop over all hits and produce digits

  flush(digits, labels); // flush cached signal which cannot be affect by new event

  for (auto& hit : hits) {

    // hit of given IR can con
    // for each hit find out sector + detector information
    int detID = hit.GetDetectorID();
    int secID = hit.getSector();

    auto channel = toChannel(detID, secID);

    // o2::InteractionRecord assumes that the BC happens in the middle of 25ns interval, ZDC defines it in the beginning
    double hTime = hit.GetTime() - getTOFCorrection(detID); // account for TOF to detector
    hTime += mIR.timeNS;
    //
    o2::InteractionRecord irHit(hTime); // BC in which the hit appears (might be different from interaction BC for slow particles)

    // nominal time of the BC to which the hit will be attributed
    double bcTime = o2::InteractionRecord::bc2ns(irHit.bc, irHit.orbit);
    double tDiff = hTime - bcTime; // hit time wrt the BC

    auto& cachedBC = getCreateBCCache(irHit); // add hit data to cached BC data
    float nPhotons;
    if (detID == ZEM) { // TODO: ZEMCh1 and Common are both 0, could skip the check for detID
      nPhotons = (secID == ZEMCh1) ? hit.getPMCLightYield() : hit.getPMQLightYield();
    } else {
      nPhotons = (secID == Common) ? hit.getPMCLightYield() : hit.getPMQLightYield();
    }

    phe2Sample(nPhotons, tDiff, cachedBC.bcdata[channel]);
    cachedBC.labels.emplace_back(hit.getParentID(), mEventID, mSrcID, channel);
    const auto& lbLast = cachedBC.labels.back();

    // the photons signal is spread over 24ns, check if we can have contribution to next BC
    while ((tDiff -= o2::constants::lhc::LHCBunchSpacingNS) > -SampleLenghtNS) { // hit time wrt the next BC
      hTime += o2::constants::lhc::LHCBunchSpacingNS;                            // just to go to the next BC
      o2::InteractionRecord irHitNext(hTime);
      auto& cachedBCNext = getCreateBCCache(irHitNext); // NOTE: if new cached BC was created, this may invalidate cachedBC reference!
      phe2Sample(nPhotons, tDiff, cachedBCNext.bcdata[channel]);
      cachedBCNext.labels.push_back(lbLast); // same label is assigned to the overflow signal
    }
    // if digit for this sector does not exist, create one otherwise add to it
  }
}

void Digitizer::flush(std::vector<o2::zdc::Digit>& digits,
                      o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>& labels)
{
  // create digits from cached BC data which is more than 2 BCs past wrt the BC of currently processed event
  int nCached = mCache.size();
  int firstUnused = 0;
  for (int ib = 0; ib < nCached; ib++) {
    const auto& cachedIR = mCache[ib].intRecord;
    int bcDiff = mIR.differenceInBC(cachedIR);
    if (bcDiff > NBCAfter) {
      firstUnused = createDigit(digits, labels, ib); // entry in the cache last used to create a digit (can be up to ib+2)
      continue;
    }
    break;
  }

  for (int ib = firstUnused; ib--;) { // clear cached BCs wich are not needed anymore
    mCache.pop_front();
  }
}

void Digitizer::phe2Sample(int nphe, double timeWrtSample, ChannelBCDataF& sample) const
{
  //function to simulate the waveform from no. of photoelectrons seen in a given sample
  // for electrons at timeInSample wrt beginning of the sample

  // approximate with a triangular shape
  const float tRiseNS = 4.0; // RS: fix these params

  float maxAmplitude = 2. * nphe; // 2 -> put in constant!

  float tBin = 0.5 * ChannelTimeBinNS - timeWrtSample; // distance from 1st time-bin center to beginning of the signal
  if (tBin > SampleLenghtNS || tBin < -SampleLenghtNS) {
    return; // no contribution to this sample
  }
  for (int i = 0; i < sample.data.size(); i++) {
    if (tBin > 0.) {
      if (tBin > SampleLenghtNS) {
        break; // no point in continuing
      }
      float amp = (tBin < tRiseNS) ? tBin / tRiseNS : 1.f - (tBin - tRiseNS) / (SampleLenghtNS - tRiseNS);
      sample.data[i] += amp * maxAmplitude;
    }
    tBin += ChannelTimeBinNS;
  }
}

int Digitizer::createDigit(std::vector<o2::zdc::Digit>& digits, o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>& labels, int cachedID)
{
  // if the cached data for this bunch crossing provides a trigger, create a digit. Return the 1st cached BC id which is not contributing to this digit
  const BCCache empty;
  int fstUnused = cachedID + 1;
  const auto& cachedBC = mCache[cachedID];
  if (!NeedToTrigger(cachedBC)) {
    return fstUnused;
  }

  std::array<const BCCache*, NBCReadOut> slots = {&empty, &cachedBC, &empty, &empty};
  slots[1] = &cachedBC;
  if (cachedID > 0 && cachedBC.intRecord.differenceInBC(mCache[cachedID - 1].intRecord) == 1) { // digits stores 1 BC before the trigger
    slots[0] = &mCache[cachedID - 1];
  }
  if (cachedID + 1 < mCache.size()) { // and 2 BCs after the trigger
    int difBC = mCache[cachedID + 1].intRecord.differenceInBC(cachedBC.intRecord);
    if (difBC == 1) {
      slots[2] = &mCache[cachedID + 1];
      fstUnused++;
      // if next cached BC differs by one, then next to next has a chance to differ by 2 BCs
      if (cachedID + 2 < mCache.size() && mCache[cachedID + 2].intRecord.differenceInBC(cachedBC.intRecord) == 2) {
        slots[3] = &mCache[cachedID + 2];
        fstUnused++;
      }
    } else if (difBC == 2) { // it might be that there was an empty BC between 2 cached
      slots[3] = &mCache[cachedID + 1];
      fstUnused++;
    }
  }

  int digID = digits.size();
  auto& newdig = digits.emplace_back();
  newdig.getInteractionRecord() = cachedBC.intRecord;
  LOG(INFO) << "Adding new ZDC digit with InteractionRecord " << cachedBC.intRecord;
  for (int ich = NChannels; ich--;) {
    for (int isl = 4; isl--;) {
      const auto& src = slots[isl]->bcdata[ich].data;
      auto& dest = newdig.getChannel(ich, isl).data;
      for (int ib = NTimeBinsPerBC; ib--;) {
        dest[ib] = uint16_t(src[ib]); // accumulation is done in floats, digit holds 16bit ADC values
      }
    }
  }
  // Hadronic ZDCs also store the SUM of 4 towers
  for (int id = 0; id < 5; id++) {
    int idet = id + DetIDOffs; // once we start detector from 0, this should be removed
    if (idet != ZEM) {
      for (int isl = 4; isl--;) {
        auto& dest = newdig.getChannel(toChannel(idet, Sum), isl).data;
        for (int it = Ch1; it <= Ch4; it++) {
          const auto& src = newdig.getChannel(toChannel(idet, it), isl).data;
          for (int ib = NTimeBinsPerBC; ib--;) {
            dest[ib] += src[ib];
          } // loop over ADC bins
        }   // loop over towers
      }     // loop over 4 stored BC
    }
  } // loop over detectors

  // register labels of principal BC

  for (const auto& lbl : cachedBC.labels) {
    labels.addElement(digID, lbl);
  }
  return fstUnused;
}

o2::zdc::Digitizer::BCCache& Digitizer::getCreateBCCache(const o2::InteractionRecord& ir)
{
  if (mCache.empty() || mCache.back().intRecord < ir) {
    mCache.emplace_back();
    auto& cb = mCache.back();
    cb.intRecord = ir;
    return cb;
  }
  if (mCache.front().intRecord > ir) {
    mCache.emplace_front();
    auto& cb = mCache.front();
    cb.intRecord = ir;
    return cb;
  }

  for (auto cb = mCache.begin(); cb != mCache.end(); cb++) {
    if ((*cb).intRecord == ir) {
      return *cb;
    }
    if (ir < (*cb).intRecord) {
      auto cbnew = mCache.emplace(cb); // insert new element before cb
      (*cbnew).intRecord = ir;
      return (*cbnew);
    }
  }
  return mCache.front();
}

bool Digitizer::NeedToTrigger(const o2::zdc::Digitizer::BCCache& bc) const
{
  return true; // placeholder
}
