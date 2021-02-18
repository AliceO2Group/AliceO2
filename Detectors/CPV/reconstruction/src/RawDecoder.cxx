// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <FairLogger.h>
#include "CPVReconstruction/RawReaderMemory.h"
#include "CPVReconstruction/RawDecoder.h"
#include "DataFormatsCPV/RawFormats.h"
#include "InfoLogger/InfoLogger.hxx"
#include "DetectorsRaw/RDHUtils.h"
#include "CPVBase/Geometry.h"

using namespace o2::cpv;

RawDecoder::RawDecoder(RawReaderMemory& reader) : mRawReader(reader),
                                                  mRCUTrailer(),
                                                  mChannelsInitialized(false)
{
}

RawErrorType_t RawDecoder::decode()
{

  auto& header = mRawReader.getRawHeader();
  short ddl = o2::raw::RDHUtils::getFEEID(header);
  mDigits.clear();

  auto payloadwords = mRawReader.getPayload();
  if (payloadwords.size() == 0) {
    mErrors.emplace_back(ddl, 0, 0, 0, kNO_PAYLOAD); //add error
    LOG(ERROR) << "Empty payload for DDL=" << ddl;
    return kNO_PAYLOAD;
  }

  // if(readRCUTrailer()!=kOK){
  //   LOG(ERROR) << "can not read RCU trailer for DDL " << ddl ;
  //   return kRCU_TRAILER_ERROR;
  // }

  return readChannels();
}

RawErrorType_t RawDecoder::readRCUTrailer()
{
  gsl::span<const char> payload(reinterpret_cast<const char*>(mRawReader.getPayload().data()), mRawReader.getPayload().size() * sizeof(uint32_t));
  mRCUTrailer.constructFromRawPayload(payload);
  return kOK;
}

RawErrorType_t RawDecoder::readChannels()
{
  mChannelsInitialized = false;
  auto& header = mRawReader.getRawHeader();
  // short ddl = o2::raw::RDHUtils::getLinkID(header) ; //Current module

  auto& payloadwords = mRawReader.getPayload();
  uint32_t wc = 0;
  auto b = payloadwords.cbegin();
  auto e = payloadwords.cend();
  while (b != e) {
    cpvword w(b, e);
    if (w.isOK()) {
      for (int i = 0; i < 3; i++) {
        PadWord pw = {w.cpvPadWord(i)};
        if (pw.zero == 0) {
          addDigit(pw.mDataWord, w.ccId());
        } else {
          if (pw.mDataWord != 0xffffff) { //not empty pad
            LOG(ERROR) << "no zero bit in data word " << pw.mDataWord;
            mErrors.emplace_back(w.ccId(), 0, 0, 0, kPADERROR); //add error for non-existing row
          }
        }
      }
    } else { //this may be trailer
      cpvtrailer tr(b, e);
      if (tr.isOK()) {
        if (tr.wordCounter() != wc) {
          //some words lost?
          LOG(ERROR) << "Read " << wc << " words, expected " << tr.wordCounter();
          mErrors.emplace_back(w.ccId(), 0, 0, 0, kPAYLOAD_DECODING);
          //TODO! should we continuew or brake?
        }
      } else {
        //error
        LOG(ERROR) << "Read neither data nor trailer word";
        mErrors.emplace_back(w.ccId(), 0, 0, 0, kPADERROR); //add error for non-existing row
      }
    }
    b += 16;
    wc++;
  }

  mChannelsInitialized = true;
  return kOK;
}

const RCUTrailer& RawDecoder::getRCUTrailer() const
{
  if (!mRCUTrailer.isInitialized()) {
    LOG(ERROR) << "RCU trailer not initialized";
  }
  return mRCUTrailer;
}

const std::vector<uint32_t>& RawDecoder::getDigits() const
{
  if (!mChannelsInitialized) {
    LOG(ERROR) << "Channels not initialized";
  }
  return mDigits;
}

void RawDecoder::addDigit(uint32_t w, short ccId)
{

  PadWord pad = {w};
  unsigned short absId;
  o2::cpv::Geometry::hwaddressToAbsId(ccId, pad.dil, pad.gas, pad.address, absId);

  AddressCharge ac = {0};
  ac.Address = absId;
  ac.Charge = pad.charge;
  mDigits.push_back(ac.mDataWord);
}