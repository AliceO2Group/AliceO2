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
#include "PHOSSimulation/RawWriter.h"
#include "PHOSBase/PHOSSimParams.h"
#include "CCDB/CcdbApi.h"

using namespace o2::phos;

void RawWriter::init()
{
  mRawWriter = std::make_unique<o2::raw::RawFileWriter>(o2::header::gDataOriginPHS, false);
  mRawWriter->setCarryOverCallBack(this);
  mRawWriter->setApplyCarryOverToLastPage(true);

  // initialize mapping
  if (!mMapping) {
    mMapping = std::make_unique<o2::phos::Mapping>();
    if (!mMapping) {
      LOG(ERROR) << "Failed to initialize mapping";
    }
    if (mMapping->setMapping() != o2::phos::Mapping::kOK) {
      LOG(ERROR) << "Failed to construct mapping";
    }
  }

  for (auto iddl = 0; iddl < o2::phos::Mapping::NDDL; iddl++) {
    // For PHOS set
    std::string rawfilename = mOutputLocation;
    switch (mFileFor) {
      case FileFor_t::kFullDet:
        rawfilename += "/phos.raw";
        break;
      case FileFor_t::kLink:
        rawfilename += fmt::format("/phos_{:d}.raw", iddl);
    }
    short crorc, link;
    mMapping->ddlToCrorcLink(iddl, crorc, link);
    mRawWriter->registerLink(iddl, crorc, link, 0, rawfilename.data());
  }

  // initialize containers for SRU
  for (auto isru = 0; isru < o2::phos::Mapping::NDDL; isru++) {
    SRUDigitContainer srucont;
    srucont.mSRUid = isru;
    mSRUdata.push_back(srucont);
  }
}

void RawWriter::digitsToRaw(gsl::span<o2::phos::Digit> digitsbranch, gsl::span<o2::phos::TriggerRecord> triggerbranch)
{
  if (!mCalibParams) {
    if (o2::phos::PHOSSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
      mCalibParams = std::make_unique<CalibParams>(1); // test default calibration
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
      // mCalibParams = ccdb.retrieveFromTFileAny<o2::phos::CalibParams>("PHOS/Calib", metadata, eventTime);
      if (!mCalibParams) {
        LOG(FATAL) << "[RawWriter] can not get calibration object from ccdb";
      }
    }
  }

  for (auto trg : triggerbranch) {
    processTrigger(digitsbranch, trg);
  }
}

bool RawWriter::processTrigger(const gsl::span<o2::phos::Digit> digitsbranch, const o2::phos::TriggerRecord& trg)
{
  auto srucont = mSRUdata.begin();
  while (srucont != mSRUdata.end()) {
    srucont->mChannels.clear();
    srucont++;
  }
  std::vector<o2::phos::Digit*>* digitsList;
  for (auto& dig : gsl::span(digitsbranch.data() + trg.getFirstEntry(), trg.getNumberOfObjects())) {
    short absId = dig.getAbsId();
    short ddl, hwAddrHG;
    //get ddl and High Gain hw addresses
    if (mMapping->absIdTohw(absId, 0, ddl, hwAddrHG) != o2::phos::Mapping::kOK) {
      LOG(ERROR) << "Wrong AbsId" << absId;
    }

    //Collect possible several digits (signal+pileup) into one map record
    auto celldata = mSRUdata[ddl].mChannels.find(absId);
    if (celldata == mSRUdata[ddl].mChannels.end()) {
      const auto it = mSRUdata[ddl].mChannels.insert(celldata, {absId, std::vector<o2::phos::Digit*>()});
      it->second.push_back(&dig);
    } else {
      celldata->second.push_back(&dig);
    }
  }

  // Create and fill DMA pages for each channel
  std::vector<uint32_t> rawbunches;
  std::vector<char> payload;
  std::vector<AltroBunch> rawbunchesHG, rawbunchesLG;

  for (srucont = mSRUdata.begin(); srucont != mSRUdata.end(); srucont++) {
    short ddl = srucont->mSRUid;
    payload.clear();

    for (auto ch = srucont->mChannels.cbegin(); ch != srucont->mChannels.cend(); ch++) {
      // Find out hardware address of the channel
      bool isLGfilled = 0;
      createRawBunches(ch->first, ch->second, rawbunchesHG, rawbunchesLG, isLGfilled);

      short hwAddrHG; //High gain always filled
      if (mMapping->absIdTohw(ch->first, 0, ddl, hwAddrHG) != o2::phos::Mapping::kOK) {
        LOG(ERROR) << "Wrong AbsId" << ch->first;
      }
      rawbunches.clear();
      for (auto& bunch : rawbunchesHG) {
        rawbunches.push_back(bunch.mADCs.size() + 2);
        rawbunches.push_back(bunch.mStarttime);
        for (auto adc : bunch.mADCs) {
          rawbunches.push_back(adc);
        }
      }
      if (rawbunches.size() == 0) {
        continue;
      }
      auto encodedbunches = encodeBunchData(rawbunches);
      ChannelHeader chanhead = {0};
      chanhead.mHardwareAddress = hwAddrHG;
      chanhead.mPayloadSize = rawbunches.size();
      chanhead.mMark = 1; //mark channel header
      char* chanheadwords = reinterpret_cast<char*>(&chanhead.mDataWord);
      for (int iword = 0; iword < sizeof(ChannelHeader) / sizeof(char); iword++) {
        payload.emplace_back(chanheadwords[iword]);
      }

      char* channelwords = reinterpret_cast<char*>(encodedbunches.data());
      for (auto iword = 0; iword < encodedbunches.size() * sizeof(int) / sizeof(char); iword++) {
        payload.emplace_back(channelwords[iword]);
      }

      if (isLGfilled) { //fill both HighGain, and LowGain channels in case of saturation
        short hwAddrLG; //High gain always filled
        if (mMapping->absIdTohw(ch->first, 1, ddl, hwAddrLG) != o2::phos::Mapping::kOK) {
          LOG(ERROR) << "Wrong AbsId" << ch->first;
        }

        rawbunches.clear();
        for (auto& bunch : rawbunchesLG) {
          rawbunches.push_back(bunch.mADCs.size() + 2);
          rawbunches.push_back(bunch.mStarttime);
          for (auto adc : bunch.mADCs) {
            rawbunches.push_back(adc);
          }
        }

        encodedbunches = encodeBunchData(rawbunches);
        ChannelHeader chanheadLG = {0};
        chanheadLG.mHardwareAddress = hwAddrLG;
        chanheadLG.mPayloadSize = rawbunches.size();
        chanheadLG.mMark = 1; //mark channel header

        chanheadwords = reinterpret_cast<char*>(&chanheadLG.mDataWord);
        for (int iword = 0; iword < sizeof(ChannelHeader) / sizeof(char); iword++) {
          payload.emplace_back(chanheadwords[iword]);
        }
        channelwords = reinterpret_cast<char*>(encodedbunches.data());
        for (auto iword = 0; iword < encodedbunches.size() * sizeof(int) / sizeof(char); iword++) {
          payload.emplace_back(channelwords[iword]);
        }
      }
    }

    // Create RCU trailer
    auto trailerwords = createRCUTrailer(payload.size() / 4, 16, 16, 100., 0.);
    for (auto word : trailerwords) {
      payload.emplace_back(word);
    }

    // register output data
    LOG(DEBUG1) << "Adding payload with size " << payload.size() << " (" << payload.size() / 4 << " ALTRO words)";

    short crorc, link;
    mMapping->ddlToCrorcLink(ddl, crorc, link);
    mRawWriter->addData(ddl, crorc, link, 0, trg.getBCData(), payload);
  }
  return true;
}

void RawWriter::createRawBunches(short absId, const std::vector<o2::phos::Digit*>& channelDigits, std::vector<o2::phos::AltroBunch>& bunchHG,
                                 std::vector<o2::phos::AltroBunch>& bunchLG, bool& isLGFilled)
{

  isLGFilled = false;
  short samples[kNPHOSSAMPLES] = {0};
  float hglgratio = mCalibParams->getHGLGRatio(absId);
  for (auto dig : channelDigits) {
    //Convert energy and time to ADC counts and time ticks
    float ampADC = dig->getAmplitude();                             // Digits amplitude already in ADC channels
    if (ampADC > o2::phos::PHOSSimParams::Instance().mMCOverflow) { //High Gain in saturation, fill also Low Gain
      isLGFilled = true;
    }
    float timeTicks = dig->getTime();                           //time in ns
    timeTicks /= o2::phos::PHOSSimParams::Instance().mTimeTick; //time in PHOS ticks
    //Add to current sample contribution from digit
    fillGamma2(ampADC, timeTicks, samples);
  }

  //reduce samples below ZS and fill output
  short zs = (short)o2::phos::PHOSSimParams::Instance().mZSthreshold;
  bunchHG.clear();
  AltroBunch currentBunch;
  //Note reverse time order
  for (int i = kNPHOSSAMPLES; i--;) {
    if (samples[i] > zs) {
      currentBunch.mADCs.emplace_back(std::min(o2::phos::PHOSSimParams::Instance().mMCOverflow, samples[i]));
    } else { //end of sample?
      if (currentBunch.mADCs.size()) {
        currentBunch.mStarttime = i + 1;
        bunchHG.push_back(currentBunch);
        currentBunch.mADCs.clear();
      }
    }
  }
  if (currentBunch.mADCs.size()) {
    bunchHG.push_back(currentBunch);
    currentBunch.mADCs.clear();
  }
  if (isLGFilled) {
    bunchLG.clear();
    currentBunch.mADCs.clear();
    for (int i = kNPHOSSAMPLES; i--;) {
      if (samples[i] > zs * hglgratio) {
        currentBunch.mADCs.emplace_back(std::min(o2::phos::PHOSSimParams::Instance().mMCOverflow, short(samples[i] / hglgratio)));
      } else { //end of sample?
        if (currentBunch.mADCs.size()) {
          currentBunch.mStarttime = i + 1;
          bunchLG.push_back(currentBunch);
          currentBunch.mADCs.clear();
        }
      }
    }
    if (currentBunch.mADCs.size()) {
      bunchLG.push_back(currentBunch);
    }
  }
}

void RawWriter::fillGamma2(float amp, float time, short* samples)
{
  //Simulate Gamma2 signal added to current sample in PHOS
  float alpha = o2::phos::PHOSSimParams::Instance().mSampleDecayTime;
  amp += 0.5; //rounding err
  for (int i = 0; i < kNPHOSSAMPLES; i++) {
    if (i - o2::phos::PHOSSimParams::Instance().mPreSamples < time) {
      continue;
    }
    float x = alpha * (i - o2::phos::PHOSSimParams::Instance().mPreSamples - time);
    float y = 0.25 * amp * x * x * std::exp(2. - x); //0.25*exp(-2) normalization to unity
    samples[i] += short(y);
  }
}

std::vector<uint32_t> RawWriter::encodeBunchData(const std::vector<uint32_t>& data)
{
  std::vector<uint32_t> encoded;
  CaloBunchWord currentword;
  currentword.mDataWord = 0;
  int wordnumber = 0;
  for (auto adc : data) {
    switch (wordnumber) {
      case 0:
        currentword.mWord0 = adc;
        break;
      case 1:
        currentword.mWord1 = adc;
        break;
      case 2:
        currentword.mWord2 = adc;
        break;
    };
    wordnumber++;
    if (wordnumber == 3) {
      // start new word;
      encoded.push_back(currentword.mDataWord);
      currentword.mDataWord = 0;
      wordnumber = 0;
    }
  }
  if (wordnumber) {
    encoded.push_back(currentword.mDataWord);
  }
  return encoded;
}

std::vector<char> RawWriter::createRCUTrailer(int payloadsize, int feca, int fecb, double timesample, double l1phase)
{
  RCUTrailer trailer;
  trailer.setActiveFECsA(feca);
  trailer.setActiveFECsB(fecb);
  trailer.setPayloadSize(payloadsize);
  trailer.setL1Phase(l1phase);
  trailer.setTimeSample(timesample);
  auto trailerwords = trailer.encode();
  std::vector<char> encoded(trailerwords.size() * sizeof(uint32_t));
  memcpy(encoded.data(), trailerwords.data(), trailerwords.size() * sizeof(uint32_t));
  return encoded;
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
