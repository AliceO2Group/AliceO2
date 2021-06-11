// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FairLogger.h"

#include <fmt/core.h>
#include <gsl/span>
#include <TSystem.h>
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsCPV/RawFormats.h"
#include "CPVSimulation/RawWriter.h"
#include "CPVBase/CPVSimParams.h"
#include "CPVBase/Geometry.h"
#include "CCDB/CcdbApi.h"

using namespace o2::cpv;

void RawWriter::init()
{
  mRawWriter = std::make_unique<o2::raw::RawFileWriter>(o2::header::gDataOriginCPV, true); //true = cru detector
  mRawWriter->setCarryOverCallBack(this);
  mRawWriter->setApplyCarryOverToLastPage(true);

  // Set output file and register link
  std::string rawfilename = mOutputLocation;
  rawfilename += "/cpv.raw";

  //ddl,crorc, link,...
  mRawWriter->registerLink(0, 0, 0, 0, rawfilename.data());
}

void RawWriter::digitsToRaw(gsl::span<o2::cpv::Digit> digitsbranch, gsl::span<o2::cpv::TriggerRecord> triggerbranch)
{
  if (triggerbranch.begin() == triggerbranch.end()) { //do we have any data?
    return;
  }
  if (!mCalibParams) {
    if (o2::cpv::CPVSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
      mCalibParams = std::make_unique<o2::cpv::CalibParams>(1); // test default calibration
      LOG(INFO) << "[RawWriter] No reading calibration from ccdb requested, set default";
    } else {
      LOG(INFO) << "[RawWriter] getting calibration object from ccdb";
      o2::ccdb::CcdbApi ccdb;
      std::map<std::string, std::string> metadata;
      ccdb.init("http://ccdb-test.cern.ch:8080"); // or http://localhost:8080 for a local installation
      auto tr = triggerbranch.begin();
      double eventTime = -1;
      // if(tr!=triggerbranch.end()){
      //   eventTime = (*tr).getBCData().getTimeNS() ;
      // }
      //add copy constructor if necessary
      //      mCalibParams = std::make_unique<o2::cpv::CalibParams>(ccdb.retrieveFromTFileAny<o2::cpv::CalibParams>("CPV/Calib", metadata, eventTime));
      if (!mCalibParams) {
        LOG(FATAL) << "[RawWriter] can not get calibration object from ccdb";
      }
    }
  }

  if (!mPedestals) {
    if (o2::cpv::CPVSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
      mPedestals = std::make_unique<o2::cpv::Pedestals>(1); // test default calibration
      LOG(INFO) << "[RawWriter] No reading calibration from ccdb requested, set default";
    } else {
      LOG(INFO) << "[RawWriter] getting calibration object from ccdb";
      o2::ccdb::CcdbApi ccdb;
      std::map<std::string, std::string> metadata;
      ccdb.init("http://ccdb-test.cern.ch:8080"); // or http://localhost:8080 for a local installation
      auto tr = triggerbranch.begin();
      double eventTime = -1;
      // if(tr!=triggerbranch.end()){
      //   eventTime = (*tr).getBCData().getTimeNS() ;
      // }
      //add copy constructor if necessary
      //      mPedestals = std::make_unique<o2::cpv::Pedestals>(ccdb.retrieveFromTFileAny<o2::cpv::Pedestals>("CPV/Calib", metadata, eventTime));
      if (!mPedestals) {
        LOG(FATAL) << "[RawWriter] can not get calibration object from ccdb";
      }
    }
  }

  //process digits which belong to same orbit
  int iFirstTrgInCurrentOrbit = 0;
  int currentOrbit = triggerbranch[0].getBCData().orbit;
  int nTrgsInCurrentOrbit = 1;
  for (int iTrg = 1; iTrg < triggerbranch.size(); iTrg++) {
    if (triggerbranch[iTrg].getBCData().orbit != currentOrbit) { //if orbit changed, write previous orbit to file
      processOrbit(digitsbranch, triggerbranch.subspan(iFirstTrgInCurrentOrbit, nTrgsInCurrentOrbit));
      iFirstTrgInCurrentOrbit = iTrg; //orbit changed
      nTrgsInCurrentOrbit = 1;
      currentOrbit = triggerbranch[iTrg].getBCData().orbit;
    } else {
      nTrgsInCurrentOrbit++;
    }
  }
  processOrbit(digitsbranch, triggerbranch.subspan(iFirstTrgInCurrentOrbit, nTrgsInCurrentOrbit)); //process last orbit
}

//prepare preformatted data for one orbit and send it to RawFileWriter
bool RawWriter::processOrbit(const gsl::span<o2::cpv::Digit> digitsbranch, const gsl::span<o2::cpv::TriggerRecord> trgs)
{
  //Array used to sort properly digits
  for (int i = kNcc; i--;) {
    for (int j = kNDilogic; j--;) {
      for (int k = kNGasiplex; k--;) {
        mPadCharge[i][j][k].clear();
      }
    }
  }
  //clear payload
  mPayload.clear();

  //we're going to prepare preformatted pages
  int feeid = 0, cru = 0, link = 0, endpoint = 0;
  bool preformatted = true;

  int gbtWordCounter = 0;
  int gbtWordCounterBeforeCPVTrailer = 0;
  int nMaxGbtWordsPerPage = o2::raw::RDHUtils::MAXCRUPage / o2::raw::RDHUtils::GBTWord; //512*16/16 = 512;
  nMaxGbtWordsPerPage -= 4;                                                             //rdh size = 4 gbt words
  for (auto& trg : trgs) {
    gbtWordCounterBeforeCPVTrailer = 0; //every trigger we start this counter from 0

    //make payload for current trigger
    int nDigsInTrg = 0;
    for (auto& dig : gsl::span(digitsbranch.data() + trg.getFirstEntry(), trg.getNumberOfObjects())) {

      short absId = dig.getAbsId();
      short ccId, dil, gas, pad;
      o2::cpv::Geometry::absIdToHWaddress(absId, ccId, dil, gas, pad);

      //Convert Amp to ADC counts
      short charge = dig.getAmplitude() / mCalibParams->getGain(absId) + mPedestals->getPedestal(absId);
      if (charge > 4095) {
        charge = 4095;
      }
      mPadCharge[ccId][dil][gas].emplace_back(charge, pad);
      nDigsInTrg++;
    }

    //we need to write header + trailer + at least 1 payload word
    if (nMaxGbtWordsPerPage - gbtWordCounter < 3) { //otherwise flush already prepared data to file
      LOG(DEBUG) << "RawWriter::processOrbit() : before header: adding preformatted dma page of size " << mPayload.size();
      mRawWriter->addData(feeid, cru, link, endpoint, trg.getBCData(), gsl::span<char>(mPayload.data(), mPayload.size()), preformatted);
      mPayload.clear();
      gbtWordCounter = 0;
      gbtWordCounterBeforeCPVTrailer = 0;
    }

    //first, header goes
    CpvHeader header(trg.getBCData(), false, false);
    for (int i = 0; i < 16; i++) {
      mPayload.push_back(header.mBytes[i]);
    }
    gbtWordCounter++;
    gbtWordCounterBeforeCPVTrailer++;

    int ccWordCounter = 0;
    for (char ccId = 0; ccId < kNcc; ccId++) {
      for (char dil = 0; dil < kNDilogic; dil++) {
        for (char gas = 0; gas < kNGasiplex; gas++) {
          for (padCharge& pc : mPadCharge[ccId][dil][gas]) {
            // Generate 3 CC words, add CC header and empty bits to complete 128 bits;
            PadWord currentword = {0};
            currentword.charge = pc.charge;
            currentword.address = pc.pad;
            currentword.gas = gas;
            currentword.dil = dil;
            mPayload.push_back(currentword.mBytes[0]);
            mPayload.push_back(currentword.mBytes[1]);
            mPayload.push_back(currentword.mBytes[2]);
            ccWordCounter++;
            if (ccWordCounter % 3 == 0) { // complete 3 channels (72 bit) + CC index (8 bits) + 6 empty bits = Generate 128 bits of data
              mPayload.push_back(ccId);
              for (int i = 6; i--;) {
                mPayload.push_back(char(0));
              }
              gbtWordCounter++;
              gbtWordCounterBeforeCPVTrailer++;
              if (nMaxGbtWordsPerPage - gbtWordCounter < 2) {                            //the only space for trailer left on current page
                CpvTrailer tr(gbtWordCounterBeforeCPVTrailer, trg.getBCData().bc, true); //add trailer and flush page to file
                for (int i = 0; i < 16; i++) {
                  mPayload.push_back(tr.mBytes[i]);
                }
                LOG(DEBUG) << "RawWriter::processOrbit() : middle of payload: adding preformatted dma page of size " << mPayload.size();
                mRawWriter->addData(feeid, cru, link, endpoint, trg.getBCData(), gsl::span<char>(mPayload.data(), mPayload.size()), preformatted);
                mPayload.clear();
                gbtWordCounter = 0;
                gbtWordCounterBeforeCPVTrailer = 0;
                if (nDigsInTrg > gbtWordCounterBeforeCPVTrailer - 1) { //some digits left for writing
                  for (int i = 0; i < 16; i++) {                       //so put a new header and continue
                    mPayload.push_back(header.mBytes[i]);
                  }
                  gbtWordCounter++;
                  gbtWordCounterBeforeCPVTrailer++;
                }
              }
            }
          }
          if (ccWordCounter % 3 != 0) {
            while (ccWordCounter % 3 != 0) {
              mPayload.push_back(char(255));
              mPayload.push_back(char(255));
              mPayload.push_back(char(255));
              ccWordCounter++;
            }
            mPayload.push_back(ccId);
            for (int i = 6; i--;) {
              mPayload.push_back(char(0));
            }
            gbtWordCounter++;
            gbtWordCounterBeforeCPVTrailer++;
            if (nMaxGbtWordsPerPage - gbtWordCounter < 2) {                            //the only space for trailer left on current page
              CpvTrailer tr(gbtWordCounterBeforeCPVTrailer, trg.getBCData().bc, true); //add trailer and flush page to file
              for (int i = 0; i < 16; i++) {
                mPayload.push_back(tr.mBytes[i]);
              }
              LOG(DEBUG) << "RawWriter::processOrbit() : middle of payload (after filling empty words): adding preformatted dma page of size " << mPayload.size();
              mRawWriter->addData(feeid, cru, link, endpoint, trg.getBCData(), gsl::span<char>(mPayload.data(), mPayload.size()), preformatted);
              mPayload.clear();
              gbtWordCounter = 0;
              gbtWordCounterBeforeCPVTrailer = 0;
              if (nDigsInTrg > gbtWordCounterBeforeCPVTrailer - 1) { //some digits left for writing
                for (int i = 0; i < 16; i++) {                       //so put a new header and continue
                  mPayload.push_back(header.mBytes[i]);
                }
                gbtWordCounter++;
                gbtWordCounterBeforeCPVTrailer++;
              }
            }
          }
        }
      }
    }
    CpvTrailer tr(gbtWordCounterBeforeCPVTrailer, trg.getBCData().bc, true); //cout GBT words
    for (int i = 0; i < 16; i++) {
      mPayload.push_back(tr.mBytes[i]);
    }
    gbtWordCounterBeforeCPVTrailer++;
    gbtWordCounter++;
  }                      //end of for (auto& trg : trgs)
  if (mPayload.size()) { //flush payload to file (if any)
    LOG(DEBUG) << "RawWriter::processOrbit() : final payload: adding preformatted dma page of size " << mPayload.size();
    mRawWriter->addData(feeid, cru, link, endpoint, trgs.back().getBCData(), gsl::span<char>(mPayload.data(), mPayload.size()), preformatted);
    mPayload.clear();
  }
  return true;
}
//carryover method is not used as we write preformatted pages
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
