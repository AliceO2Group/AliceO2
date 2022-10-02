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

#include <fairlogger/Logger.h>

#include <fmt/core.h>
#include <gsl/span>
#include <TSystem.h>
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsCPV/RawFormats.h"
#include "CPVSimulation/RawWriter.h"
#include "CPVBase/CPVSimParams.h"
#include "CPVBase/Geometry.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCTP/TriggerOffsetsParam.h"
#include "DetectorsRaw/HBFUtils.h"

using namespace o2::cpv;

void RawWriter::init()
{
  mRawWriter = std::make_unique<o2::raw::RawFileWriter>(o2::header::gDataOriginCPV, true); // true = cru detector
  mRawWriter->setCarryOverCallBack(this);
  mRawWriter->setApplyCarryOverToLastPage(true);

  // register all cpv links
  for (auto&& link : links) {
    std::string rawFileName = mOutputLocation + "/CPV_" + link.flpId + "_cru" + std::to_string(link.cruId) + "_" + std::to_string(link.endPointId);
    if (mFileFor == FileFor_t::kLink) {
      rawFileName += Form("link%d", link.linkId);
    }
    rawFileName += ".raw";
    mRawWriter->registerLink(link.feeId, link.cruId, link.linkId, link.endPointId, rawFileName.data());
  }

  // CCDB setup
  const auto& hbfutils = o2::raw::HBFUtils::Instance();
  LOG(info) << "CCDB Url: " << mCcdbUrl;
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbMgr.setURL(mCcdbUrl);
  bool isCcdbReachable = ccdbMgr.isHostReachable(); // if host is not reachable we can use only dummy calibration
  if (!isCcdbReachable) {                           // dummy calibration
    if (mCcdbUrl.compare("localtest") != 0) {
      LOG(error) << "Host " << mCcdbUrl << " is not reachable!!!";
    }
    LOG(info) << "Using dummy calibration and default Lm-L0 delay";
    mLM_L0_delay = o2::ctp::TriggerOffsetsParam::Instance().LM_L0;
    mCalibParams = new o2::cpv::CalibParams(1);
    mBadMap = new o2::cpv::BadChannelMap(1);
    mPedestals = new o2::cpv::Pedestals(1);
  } else {                                        // read ccdb
    ccdbMgr.setCaching(true);                     // make local cache of remote objects
    ccdbMgr.setLocalObjectValidityChecking(true); // query objects from remote site only when local one is not valid
    LOG(info) << "Successfully initializated BasicCCDBManager with caching option";

    // read calibration from ccdb (for now do it only at the beginning of dataprocessing)
    // setup timestamp according to anchors
    ccdbMgr.setTimestamp(hbfutils.startTime);
    LOG(info) << "Using time stamp " << ccdbMgr.getTimestamp();

    // Lm-L0 delay
    mLM_L0_delay = ccdbMgr.get<o2::ctp::TriggerOffsetsParam>("CTP/Config/TriggerOffsets")->LM_L0;

    // gains
    LOG(info) << "CCDB: Reading o2::cpv::CalibParams from CPV/Calib/Gains";
    mCalibParams = ccdbMgr.get<o2::cpv::CalibParams>("CPV/Calib/Gains");
    if (!mCalibParams) {
      LOG(error) << "Cannot get o2::cpv::CalibParams from CCDB. using dummy calibration!";
      mCalibParams = new o2::cpv::CalibParams(1);
    }

    // no need to mask bad channels -> they will be thrown away at reconstruntion anyway
    /*
    LOG(info) << "CCDB: Reading o2::cpv::BadChannelMap from CPV/Calib/BadChannelMap";
    mBadMap = ccdbMgr.get<o2::cpv::BadChannelMap>("CPV/Calib/BadChannelMap"));
    if (!mBadMap) {
      LOG(error) << "Cannot get o2::cpv::BadChannelMap from CCDB. using dummy calibration!";
      mBadMap = new o2::cpv::BadChannelMap(1);
    }
    */

    // pedestals
    LOG(info) << "CCDB: Reading o2::cpv::Pedestals from CPV/Calib/Pedestals";
    mPedestals = ccdbMgr.get<o2::cpv::Pedestals>("CPV/Calib/Pedestals");
    if (!mPedestals) {
      LOG(error) << "Cannot get o2::cpv::Pedestals from CCDB. using dummy calibration!";
      mPedestals = new o2::cpv::Pedestals(1);
    }
    LOG(info) << "Task configuration is done.";
  }
}

void RawWriter::digitsToRaw(gsl::span<o2::cpv::Digit> digitsbranch, gsl::span<o2::cpv::TriggerRecord> triggerbranch)
{
  if (triggerbranch.begin() == triggerbranch.end()) { // do we have any data?
    return;
  }

  // process digits which belong to same orbit (taking into account )
  int iFirstTrgInCurrentOrbit = 0;
  unsigned int currentOrbit = (triggerbranch[0].getBCData() + mLM_L0_delay).orbit;
  int nTrgsInCurrentOrbit = 1;
  for (unsigned int iTrg = 1; iTrg < triggerbranch.size(); iTrg++) {
    if ((triggerbranch[iTrg].getBCData() + mLM_L0_delay).orbit != currentOrbit) { // if orbit changed, write previous orbit to file
      processOrbit(digitsbranch, triggerbranch.subspan(iFirstTrgInCurrentOrbit, nTrgsInCurrentOrbit));
      iFirstTrgInCurrentOrbit = iTrg; // orbit changed
      nTrgsInCurrentOrbit = 1;
      currentOrbit = (triggerbranch[iTrg].getBCData() + mLM_L0_delay).orbit;
    } else {
      nTrgsInCurrentOrbit++;
    }
  }
  processOrbit(digitsbranch, triggerbranch.subspan(iFirstTrgInCurrentOrbit, nTrgsInCurrentOrbit)); // process last orbit
}

// prepare preformatted data for one orbit and send it to RawFileWriter
bool RawWriter::processOrbit(const gsl::span<o2::cpv::Digit> digitsbranch, const gsl::span<o2::cpv::TriggerRecord> trgs)
{
  static int nMaxGbtWordsPerPage = o2::raw::RDHUtils::MAXCRUPage / o2::raw::RDHUtils::GBTWord - 4; // 512*16/16 - 4 = 508;
                                                                                                   // 4 gbt words are reserved for RDH

  // clear payloads of all links
  for (auto& payload : mPayload) {
    payload.clear();
  }

  // we're going to prepare preformatted pages
  bool preformatted = true;

  int gbtWordCounter[kNGBTLinks] = {0, 0, 0};
  int gbtWordCounterBeforeCPVTrailer[kNGBTLinks] = {0, 0, 0};
  bool isHeaderClosedWithTrailer[kNGBTLinks] = {false, false, false};
  for (auto& trg : trgs) {
    o2::InteractionRecord currentIR = trg.getBCData();
    currentIR += mLM_L0_delay;
    LOG(debug) << "RawWriter::processOrbit() : "
               << "I start to process trigger record (orbit = " << currentIR.orbit
               << ", BC = " << currentIR.bc << ")";
    LOG(debug) << "First entry = " << trg.getFirstEntry() << ", Number of objects = " << trg.getNumberOfObjects();

    // Clear array which is used to store digits
    for (int i = kNcc; i--;) {
      for (int j = kNDilogic; j--;) {
        for (int k = kNGasiplex; k--;) {
          mPadCharge[i][j][k].clear();
        }
      }
    }

    // make payload for current trigger
    int nDigsInTrg[kNGBTLinks] = {0, 0, 0};
    for (auto& dig : gsl::span(digitsbranch.data() + trg.getFirstEntry(), trg.getNumberOfObjects())) {

      short absId = dig.getAbsId();
      short ccId, dil, gas, pad;
      o2::cpv::Geometry::absIdToHWaddress(absId, ccId, dil, gas, pad);

      // Convert Amp to ADC counts
      short charge = std::round(dig.getAmplitude() / mCalibParams->getGain(absId) + mPedestals->getPedestal(absId));
      if (charge > 4095) {
        charge = 4095;
      }
      mPadCharge[ccId][dil][gas].emplace_back(charge, pad);
      nDigsInTrg[ccId / (kNcc / kNGBTLinks)]++; // linkId = ccId/8 or absId/7680
    }
    LOG(debug) << "I produced " << nDigsInTrg << " digits for this trigger record";

    // we need to write header + at least 1 payload word + trailer
    for (int iLink = 0; iLink < kNGBTLinks; iLink++) { // looping links
      gbtWordCounterBeforeCPVTrailer[iLink] = 0;
      if (nMaxGbtWordsPerPage - gbtWordCounter[iLink] < 3) { // otherwise flush already prepared data to file
        LOG(debug) << "RawWriter::processOrbit() : before header: adding preformatted dma page of size " << mPayload[iLink].size();
        mRawWriter->addData(links[iLink].feeId, links[iLink].cruId, links[iLink].linkId, links[iLink].endPointId, currentIR,
                            gsl::span<char>(mPayload[iLink].data(), mPayload[iLink].size()), preformatted);
        mPayload[iLink].clear();
        gbtWordCounter[iLink] = 0;
        gbtWordCounterBeforeCPVTrailer[iLink] = 0;
      }

      // first, header goes
      CpvHeader header(currentIR, false, false);
      for (int i = 0; i < 16; i++) {
        mPayload[iLink].push_back(header.mBytes[i]);
      }
      isHeaderClosedWithTrailer[iLink] = false;
      LOG(debug) << "RawWriter::processOrbit() : "
                 << "I wrote cpv header for orbit = " << currentIR.orbit
                 << " and BC = " << currentIR.bc;

      gbtWordCounter[iLink]++;
      gbtWordCounterBeforeCPVTrailer[iLink]++;

      int nDigsToWriteLeft = nDigsInTrg[iLink];

      for (char ccId = iLink * (kNcc / kNGBTLinks); (ccId < (iLink + 1) * (kNcc / kNGBTLinks)) && (ccId < kNcc); ccId++) {
        int ccWordCounter = 0;
        for (char dil = 0; dil < kNDilogic; dil++) {
          for (char gas = 0; gas < kNGasiplex; gas++) {
            for (padCharge& pc : mPadCharge[int(ccId)][int(dil)][int(gas)]) {
              // Generate 3 CC words, add CC header and empty bits to complete 128 bits;
              PadWord currentword = {0};
              currentword.charge = pc.charge;
              currentword.address = pc.pad;
              currentword.gas = gas;
              currentword.dil = dil;
              mPayload[iLink].push_back(currentword.mBytes[0]);
              mPayload[iLink].push_back(currentword.mBytes[1]);
              mPayload[iLink].push_back(currentword.mBytes[2]);
              ccWordCounter++;
              nDigsToWriteLeft--;
              if (ccWordCounter % 3 == 0) { // complete 3 channels (72 bit) + CC index (8 bits) + 6 empty bits = Generate 128 bits of data
                mPayload[iLink].push_back(ccId);
                for (int i = 6; i--;) {
                  mPayload[iLink].push_back(char(0));
                }
                gbtWordCounter[iLink]++;
                gbtWordCounterBeforeCPVTrailer[iLink]++;
                if (nMaxGbtWordsPerPage - gbtWordCounter[iLink] == 1) {                                      // the only space for trailer left on current page
                  CpvTrailer tr(gbtWordCounterBeforeCPVTrailer[iLink], currentIR.bc, nDigsToWriteLeft == 0); // add trailer and flush page to file
                  for (int i = 0; i < 16; i++) {
                    mPayload[iLink].push_back(tr.mBytes[i]);
                  }
                  isHeaderClosedWithTrailer[iLink] = true;
                  LOG(debug) << "RawWriter::processOrbit() : middle of payload: adding preformatted dma page of size " << mPayload[iLink].size();
                  mRawWriter->addData(links[iLink].feeId, links[iLink].cruId, links[iLink].linkId, links[iLink].endPointId, currentIR,
                                      gsl::span<char>(mPayload[iLink].data(), mPayload[iLink].size()), preformatted);

                  mPayload[iLink].clear();
                  gbtWordCounter[iLink] = 0;
                  gbtWordCounterBeforeCPVTrailer[iLink] = 0;
                  if (nDigsToWriteLeft) { // some digits left for writing
                    CpvHeader newHeader(currentIR, false, true);
                    for (int i = 0; i < 16; i++) { // so put a new header and continue
                      mPayload[iLink].push_back(newHeader.mBytes[i]);
                    }
                    isHeaderClosedWithTrailer[iLink] = false;
                    gbtWordCounter[iLink]++;
                    gbtWordCounterBeforeCPVTrailer[iLink]++;
                  }
                }
              }
            }
          }
        } // end of dil cycle
        if (ccWordCounter % 3 != 0) {
          while (ccWordCounter % 3 != 0) {
            mPayload[iLink].push_back(char(255));
            mPayload[iLink].push_back(char(255));
            mPayload[iLink].push_back(char(255));
            ccWordCounter++;
          }
          mPayload[iLink].push_back(ccId);
          for (int i = 6; i--;) {
            mPayload[iLink].push_back(char(0));
          }
          gbtWordCounter[iLink]++;
          gbtWordCounterBeforeCPVTrailer[iLink]++;
          if (nMaxGbtWordsPerPage - gbtWordCounter[iLink] == 1) {                                      // the only space for trailer left on current page
            CpvTrailer tr(gbtWordCounterBeforeCPVTrailer[iLink], currentIR.bc, nDigsToWriteLeft == 0); // add trailer and flush page to file
            for (int i = 0; i < 16; i++) {
              mPayload[iLink].push_back(tr.mBytes[i]);
            }
            isHeaderClosedWithTrailer[iLink] = true;
            LOG(debug) << "RawWriter::processOrbit() : middle of payload (after filling empty words): adding preformatted dma page of size " << mPayload[iLink].size();
            mRawWriter->addData(links[iLink].feeId, links[iLink].cruId, links[iLink].linkId, links[iLink].endPointId, currentIR,
                                gsl::span<char>(mPayload[iLink].data(), mPayload[iLink].size()), preformatted);
            mPayload[iLink].clear();
            gbtWordCounter[iLink] = 0;
            gbtWordCounterBeforeCPVTrailer[iLink] = 0;
            if (nDigsToWriteLeft) {          // some digits left for writing
              for (int i = 0; i < 16; i++) { // so put a new header and continue
                mPayload[iLink].push_back(header.mBytes[i]);
              }
              isHeaderClosedWithTrailer[iLink] = false;
              gbtWordCounter[iLink]++;
              gbtWordCounterBeforeCPVTrailer[iLink]++;
            }
          }
        }
      } // end of ccId cycle
      if (!isHeaderClosedWithTrailer[iLink]) {
        CpvTrailer tr(gbtWordCounterBeforeCPVTrailer[iLink], currentIR.bc, true);
        for (int i = 0; i < 16; i++) {
          mPayload[iLink].push_back(tr.mBytes[i]);
        }
        isHeaderClosedWithTrailer[iLink] = true;
        gbtWordCounterBeforeCPVTrailer[iLink] = 0;
        gbtWordCounter[iLink]++;
      }
    } // end of iLink cycle
  }   // end of "for (auto& trg : trgs)""

  // flush payload to file (if any)
  for (int iLink = 0; iLink < kNGBTLinks; iLink++) {
    if (mPayload[iLink].size()) {
      LOG(debug) << "RawWriter::processOrbit() : final payload: adding preformatted dma page of size " << mPayload[iLink].size();
      mRawWriter->addData(links[iLink].feeId, links[iLink].cruId, links[iLink].linkId, links[iLink].endPointId,
                          trgs.back().getBCData(), gsl::span<char>(mPayload[iLink].data(), mPayload[iLink].size()), preformatted);
      mPayload[iLink].clear();
    }
  }
  return true;
}
// carryover method is not used as we write preformatted pages
int RawWriter::carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                               const char* ptr, int maxSize, int splitID,
                               std::vector<char>& trailer, std::vector<char>& header) const
{

  constexpr int cpvTrailerSize = 36;
  int offs = ptr - &data[0];                                  // offset wrt the head of the payload
  assert(offs >= 0 && size_t(offs + maxSize) <= data.size()); // make sure ptr and end of the suggested block are within the payload
  int leftBefore = data.size() - offs;                        // payload left before this splitting
  int leftAfter = leftBefore - maxSize;                       // what would be left after the suggested splitting
  int actualSize = maxSize;
  if (leftAfter && leftAfter <= cpvTrailerSize) {   // avoid splitting the trailer or writing only it.
    actualSize -= (cpvTrailerSize - leftAfter) + 4; // (as we work with int, not char in decoding)
  }
  return actualSize;
}
