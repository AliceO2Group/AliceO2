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
#include "DataFormatsCPV/RawFormats.h"
#include "CPVSimulation/RawWriter.h"
#include "CPVBase/CPVSimParams.h"
#include "CPVBase/RCUTrailer.h"
#include "CPVBase/Geometry.h"
#include "CCDB/CcdbApi.h"

using namespace o2::cpv;

void RawWriter::init()
{
  mRawWriter = std::make_unique<o2::raw::RawFileWriter>(o2::header::gDataOriginCPV, false);
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

  for (auto trg : triggerbranch) {
    processTrigger(digitsbranch, trg);
  }
}

bool RawWriter::processTrigger(const gsl::span<o2::cpv::Digit> digitsbranch, const o2::cpv::TriggerRecord& trg)
{

  //Array used to sort properly digits
  for (int j = kNRow; j--;) {
    for (int k = kNDilogic; k--;) {
      mPadCharge[j][k].clear();
    }
  }

  for (auto& dig : gsl::span(digitsbranch.data() + trg.getFirstEntry(), trg.getNumberOfObjects())) {

    short absId = dig.getAbsId();
    short mod, dilogic, row, hwAddr;
    o2::cpv::Geometry::absIdToHWaddress(absId, mod, row, dilogic, hwAddr);

    //Convert Amp to ADC counts
    short charge = dig.getAmplitude() / mCalibParams->getGain(absId);
    if (charge > 2047) {
      charge = 2047;
    }
    mPadCharge[row][dilogic].emplace_back(charge, hwAddr);
  }

  //Do through DLLs and fill raw words in proper order
  mPayload.clear();
  //write empty header, later will be updated ?

  int nwInSegment = 0;
  int posRowMarker = 0;
  for (int row = 0; row < kNRow; row++) {
    //reserve place for row header
    mPayload.emplace_back(uint32_t(0));
    posRowMarker = mPayload.size() - 1;
    nwInSegment++;
    int nwRow = 0;
    for (Int_t dilogic = 0; dilogic < kNDilogic; dilogic++) {
      int nPad = 0;
      for (padCharge& pc : mPadCharge[row][dilogic]) {
        PadWord currentword = {0};
        currentword.charge = pc.charge;
        currentword.address = pc.pad;
        currentword.dilogic = dilogic;
        currentword.row = row;
        mPayload.push_back(currentword.mDataWord);
        nwInSegment++;
        nPad++;
        nwRow++;
      }
      EoEWord we = {0};
      we.nword = nPad;
      we.dilogic = dilogic;
      we.row = row;
      we.checkbit = 1;
      mPayload.push_back(we.mDataWord);
      nwInSegment++;
      nwRow++;
    }
    if (row % 2 == 1) {
      SegMarkerWord w = {0};
      w.row = row;
      w.nwords = nwInSegment;
      w.marker = 2736;
      mPayload.push_back(w.mDataWord);
      nwInSegment = 0;
      nwRow++;
    }
    //Now we know number of words, update rawMarker
    RowMarkerWord wr = {0};
    wr.marker = 13992;
    wr.nwords = nwRow - 1;
    mPayload[posRowMarker] = wr.mDataWord;
  }

  // register output data
  LOG(DEBUG1) << "Adding payload with size " << mPayload.size() << " (" << mPayload.size() << " ALTRO words)";
  mRawWriter->addData(0, 0, 0, 0, trg.getBCData(), gsl::span<char>(reinterpret_cast<char*>(mPayload.data()), mPayload.size() * sizeof(uint32_t)));
  return true;
}
int RawWriter::carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                               const char* ptr, int maxSize, int splitID,
                               std::vector<char>& trailer, std::vector<char>& header) const
{

  constexpr int phosTrailerSize = 36;
  int offs = ptr - &data[0];                                  // offset wrt the head of the payload
  assert(offs >= 0 && size_t(offs + maxSize) <= data.size()); // make sure ptr and end of the suggested block are within the payload
  int leftBefore = data.size() - offs;                        // payload left before this splitting
  int leftAfter = leftBefore - maxSize;                       // what would be left after the suggested splitting
  int actualSize = maxSize;
  if (leftAfter && leftAfter <= phosTrailerSize) {   // avoid splitting the trailer or writing only it.
    actualSize -= (phosTrailerSize - leftAfter) + 4; // (as we work with int, not char in decoding)
  }
  return actualSize;
}
