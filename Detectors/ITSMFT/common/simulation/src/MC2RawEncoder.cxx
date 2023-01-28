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

#include "ITSMFTSimulation/MC2RawEncoder.h"
#include "CommonConstants/Triggers.h"
#include "ITSMFTReconstruction/GBTLink.h"
#include "Framework/Logger.h"

using namespace o2::itsmft;
using namespace o2::raw;
using RDHUtils = o2::raw::RDHUtils;

///______________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::init()
{
  assert(mWriter.isReadOutModeSet());
  mWriter.setCarryOverCallBack(this);
  mWriter.setNewRDHCallBack(this);

  // limit RUs to convert to existing ones
  mRUSWMax = (mRUSWMax < uint8_t(mMAP.getNRUs())) ? mRUSWMax : mMAP.getNRUs() - 1;
  //
  mNLinks = 0;
  for (uint8_t ru = mRUSWMin; ru <= mRUSWMax; ru++) {
    auto& ruData = getCreateRUDecode(ru);
    int nLinks = 0;
    for (int il = 0; il < RUDecodeData::MaxLinksPerRU; il++) {
      auto* link = getGBTLink(ruData.links[il]);
      if (link) {
        auto subspec = RDHUtils::getSubSpec(link->cruID, link->idInCRU, link->endPointID, link->feeID);
        if (!mWriter.isLinkRegistered(subspec)) {
          LOGF(info, "RU%3d FEEId 0x%04x Link %02d of CRU=0x%94x will be writing to default sink %s",
               int(ru), link->feeID, link->idInCRU, link->cruID, mDefaultSinkName);
          mWriter.registerLink(link->feeID, link->cruID, link->idInCRU, link->endPointID, mDefaultSinkName);
        }
        nLinks++;
        if (link->packetCounter < 0) {
          link->packetCounter = 0; // reset only once
        }
      }
    }
    mNLinks += nLinks;
    if (!nLinks) {
      LOG(warning) << "No GBT links were defined for RU " << int(ru) << " defining automatically";
      ruData.links[0] = addGBTLink();
      auto* link = getGBTLink(ruData.links[0]);
      link->lanes = mMAP.getCablesOnRUType(ruData.ruInfo->ruType);
      link->idInCRU = 0;
      link->idInRU = 0;
      link->cruID = ru;
      link->feeID = mMAP.RUSW2FEEId(ruData.ruInfo->idSW, 0);
      link->endPointID = 0;
      link->packetCounter = 0;
      mWriter.registerLink(link->feeID, link->cruID, link->idInCRU, link->endPointID, mDefaultSinkName);
      LOGF(info, "RU%3d FEEId 0x%04x Link %02d of CRU=0x%04x Lanes: %s -> %s", int(ru), link->feeID,
           link->idInCRU, link->cruID, std::bitset<28>(link->lanes).to_string(), mDefaultSinkName);
      mNLinks++;
    }
  }
  assert(mNLinks > 0);
  // create mapping from feeID to links for fast access
  for (int i = 0; i < mNLinks; i++) {
    const auto* lnk = getGBTLink(i);
    mFEEId2Link[lnk->feeID] = lnk;
    mFEEId2GBTHeader[lnk->feeID].activeLanes = lnk->lanes;
  }
}

///______________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::finalize()
{
  mWriter.close();
}

///______________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::digits2raw(gsl::span<const Digit> digits, const o2::InteractionRecord& bcData)
{
  // Convert digits to raw data
  // The digits in the span/vector must be in increasing chipID order
  // Return the number of pages in the link with smallest amount of pages

  if (!mNLinks) {
    init();
  }
  int nDigTot = digits.size();
  mCurrIR = bcData;
  // place digits into corresponding chip buffers
  ChipPixelData* curChipData = nullptr;
  ChipInfo chInfo;
  UShort_t curChipID = 0xffff; // currently processed SW chip id
  for (const auto& dig : digits) {
    if (curChipID != dig.getChipIndex()) {
      mMAP.getChipInfoSW(dig.getChipIndex(), chInfo);
      if (chInfo.ru < mRUSWMin || chInfo.ru > mRUSWMax) { // ignore this chip?
        continue;
      }
      auto* ru = getRUDecode(chInfo.ru);
      curChipID = dig.getChipIndex();
      curChipData = &ru->chipsData[ru->nChipsFired++];
      curChipData->setChipID(chInfo.chOnRU->id); // set chip ID within the RU
    }
    curChipData->getData().emplace_back(&dig); // add new digit to the container
  }

  // convert digits to alpide data in the per-cable buffers
  for (int iru = int(mRUSWMin); iru <= int(mRUSWMax); iru++) {
    auto& ru = *getRUDecode(iru);
    uint16_t next2Proc = 0, nchTot = mMAP.getNChipsOnRUType(ru.ruInfo->ruType);
    for (int ich = 0; ich < ru.nChipsFired; ich++) {
      auto& chipData = ru.chipsData[ich];
      convertEmptyChips(next2Proc, chipData.getChipID(), ru); // if needed store EmptyChip flags for the empty chips
      next2Proc = chipData.getChipID() + 1;
      convertChip(chipData, ru);
      chipData.clear();
    }
    convertEmptyChips(next2Proc, nchTot, ru); // if needed store EmptyChip flags
    fillGBTLinks(ru);                         // flush per-lane buffers to link buffers
  }
}

//___________________________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::convertChip(ChipPixelData& chipData, RUDecodeData& ru)
{
  ///< convert digits of single chip to Alpide format.
  const auto& chip = *mMAP.getChipOnRUInfo(ru.ruInfo->ruType, chipData.getChipID());
  ru.cableHWID[chip.cableHWPos] = chip.cableHW; // register the cable HW ID
  auto& pixels = chipData.getData();
  std::sort(pixels.begin(), pixels.end(),
            [](auto lhs, auto rhs) {
              return (lhs.getRow() < rhs.getRow()) ? true : ((lhs.getRow() > rhs.getRow()) ? false : (lhs.getCol() < rhs.getCol()));
            });
  ru.cableData[chip.cableHWPos].ensureFreeCapacity(40 * (2 + pixels.size())); // make sure buffer has enough capacity
  mCoder.encodeChip(ru.cableData[chip.cableHWPos], chipData, chip.chipOnModuleHW, mCurrIR.bc);
}

//______________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::convertEmptyChips(int fromChip, int uptoChip, RUDecodeData& ru)
{
  // add empty chip words to respective cable's buffers for all chips of the current RU container
  for (int chipIDSW = fromChip; chipIDSW < uptoChip; chipIDSW++) { // flag chips w/o data
    const auto& chip = *mMAP.getChipOnRUInfo(ru.ruInfo->ruType, chipIDSW);
    ru.cableHWID[chip.cableHWPos] = chip.cableHW; // register the cable HW ID
    ru.cableData[chip.cableHWPos].ensureFreeCapacity(100);
    mCoder.addEmptyChip(ru.cableData[chip.cableHWPos], chip.chipOnModuleHW, mCurrIR.bc);
  }
}

//___________________________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::fillGBTLinks(RUDecodeData& ru)
{
  // fill data of the RU to links buffer, return the number of pages in the link with smallest amount of pages
  constexpr uint8_t zero16[GBTPaddedWordLength] = {0}; // to speedup padding
  ru.nCables = ru.ruInfo->nCables;

  // trigger word
  GBTTrigger gbtTrigger;
  gbtTrigger.bc = mCurrIR.bc;
  gbtTrigger.orbit = mCurrIR.orbit;
  gbtTrigger.internal = isContinuousReadout();
  gbtTrigger.triggerType = 0; //TODO

  for (int il = 0; il < RUDecodeData::MaxLinksPerRU; il++) {
    auto link = getGBTLink(ru.links[il]);
    if (!link) {
      continue;
    }
    // estimate real payload size in GBT words
    int nPayLoadWordsNeeded = 0;           // number of payload words filled to link buffer (RDH not included) for current IR
    for (int icab = ru.nCables; icab--;) { // calculate number of GBT words per link
      if ((link->lanes & (0x1 << mMAP.cablePos(ru.ruInfo->ruType, icab)))) {
        int nb = ru.cableData[mMAP.cablePos(ru.ruInfo->ruType, icab)].getSize();
        nPayLoadWordsNeeded += nb ? 1 + (nb - 1) / 9 : 0; // single GBT word carries at most 9 payload bytes
      }
    }
    // reserve space for payload + trigger + header + trailer
    link->data.ensureFreeCapacity((3 + nPayLoadWordsNeeded) * link->wordLength);

    link->data.addFast(gbtTrigger.getW8(), link->wordLength); // write GBT trigger

    // now loop over the lanes served by this link, writing each time at most 9 bytes, untill all lanes are copied
    bool hasData = true;
    while (hasData) {
      hasData = false;
      for (int icab = 0; icab < ru.nCables; icab++) {
        if ((link->lanes & (0x1 << mMAP.cablePos(ru.ruInfo->ruType, icab)))) {
          auto& cableData = ru.cableData[mMAP.cablePos(ru.ruInfo->ruType, icab)];
          int nb = cableData.getUnusedSize();
          if (!nb) {
            continue; // write 80b word only if there is something to write
          }
          if (nb > 9) {
            nb = 9;
          }
          int gbtWordStart = link->data.getSize();                                                                                         // beginning of the current GBT word in the link
          link->data.addFast(cableData.getPtr(), nb);                                                                                      // fill payload of cable
          link->data.addFast(zero16, link->wordLength - nb);                                                                               // fill the rest of the GBT word by 0
          link->data[gbtWordStart + 9] = mMAP.getGBTHeaderRUType(ru.ruInfo->ruType, ru.cableHWID[mMAP.cablePos(ru.ruInfo->ruType, icab)]); // set cable flag
          cableData.setPtr(cableData.getPtr() + nb);
          hasData = true;
        } // storing data of single cable
      }   // loop over cables of this link
    };    // loop until all links are w/o data

    // all payload was dumped, write final trailer
    GBTDataTrailer gbtTrailer; // lanes will be set on closing the trigger
    //    gbtTrailer.lanesStops = link->lanes; // RS CURRENTLY NOT USED
    gbtTrailer.packetDone = true;                                // RS CURRENTLY NOT USED
    link->data.addFast(gbtTrailer.getW8(), link->wordLength);    // write GBT trailer for the last packet
    LOGF(debug, "Filled %s with %d GBT words", link->describe(), nPayLoadWordsNeeded + 3);

    // flush to writer
    mWriter.addData(link->feeID, link->cruID, link->idInCRU, link->endPointID, mCurrIR, gsl::span((char*)link->data.data(), link->data.getSize()));
    link->data.clear();
    //
  } // loop over links of RU
  ru.clear();
}

///______________________________________________________________________
template <class Mapping>
RUDecodeData& MC2RawEncoder<Mapping>::getCreateRUDecode(int ruSW)
{
  assert(ruSW < mMAP.getNRUs());
  if (mRUEntry[ruSW] < 0) {
    mRUEntry[ruSW] = mNRUs++;
    mRUDecodeVec[mRUEntry[ruSW]].ruInfo = mMAP.getRUInfoSW(ruSW); // info on the stave/RU
    mRUDecodeVec[mRUEntry[ruSW]].chipsData.resize(mMAP.getNChipsOnRUType(mMAP.getRUInfoSW(ruSW)->ruType));
    LOG(info) << "Defining container for RU " << ruSW << " at slot " << mRUEntry[ruSW];
  }
  return mRUDecodeVec[mRUEntry[ruSW]];
}

///______________________________________________________________________
template <class Mapping>
int MC2RawEncoder<Mapping>::carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                                            const char* ptr, int maxSize, int splitID,
                                            std::vector<char>& trailer, std::vector<char>& header) const
{
  // The RawFileWriter receives from the encoder the payload to format according to the CRU format
  // In case this payload size does not fit into the CRU page (this may happen even if it is
  // less than 8KB, since it might be added to already partially populated CRU page of the HBF)
  // it will write on the page only part of the payload and carry over the rest on extra page(s).
  // By default the RawFileWriter will simply chunk payload as is considers necessary, but some
  // detectors want their CRU pages to be self-consistent and in case of payload splitting they
  // add in the end of page to be closed and beginning of the new page to be opened
  // (right after the RDH) detector-specific trailer and header respectively.
  //
  // The role of this method is to suggest to writer how to split the payload:
  // If this method was set to the RawFileWriter using
  // RawFileWriter::setCarryOverCallBack(pointer_on_the_converter_class);
  // then the RawFileWriter will call it before splitting.
  //
  // It provides to carryOverMethod method the following info:
  // rdh     : RDH of the CRU page being written
  // data    : original payload received by the RawFileWriter
  // ptr     : pointer on the data in the payload which was not yet added to the link CRU pages
  // maxSize : maximum size (multiple of 16 bytes) of the bloc starting at ptr which it can
  //           accomodate at the current CRU page (i.e. what it would write by default)
  // splitID : number of times this payload was already split, 0 at 1st call
  // trailer : what it wants to add in the end of the CRU page where the data starting from ptr
  //           will be added. The trailer is supplied as an empy vector, which carryOverMethod
  //           may populate, but its size must be multiple of 16 bytes.
  // header  : what it wants to add right after the RDH of the new CRU page before the rest of
  //           the payload (starting at ptr+actualSize) will be written
  //
  // The method must return actual size of the bloc which can be written (<=maxSize).
  // If this method populates the trailer, it must ensure that it returns the actual size such that
  // actualSize + trailer.size() <= maxSize
  // In case returned actualSize == 0, current CRU page will be closed w/o adding anything, and new
  // query of this method will be done on the new CRU page

  // During the carry-over ITS needs to repeat the GBTTrigger and GBTDataTrailer words.
  // Also the GBTDataHeader needs to be repeated right after continuation RDH, but it will be provided in the newRDHMethod
  const int wordSize = RDHUtils::getDataFormat(rdh) == 0 ? o2::itsmft::GBTPaddedWordLength : o2::itsmft::GBTWordLength;

  int offs = ptr - &data[0]; // offset wrt the head of the payload
  // make sure ptr and end of the suggested block are within the payload
  assert(offs >= 0 && size_t(offs + maxSize) <= data.size());

  // this is where we would usually split: account for the trailer to add
  int actualSize = maxSize - wordSize;
  char* trailPtr = &data[data.size() - wordSize]; // pointer on the payload trailer

  if ((maxSize <= 2 * wordSize)) {   // we cannot split trigger+header
    actualSize = 0;                  // suggest moving the whole payload to the new CRU page
    if (offs == 0) {                 // just carry over everything, trigger+header was not yet written
      return actualSize;
    }
  } else {
    if (ptr + actualSize >= trailPtr) { // we need to split at least 1 GBT word before the trailer
      actualSize = trailPtr - ptr - wordSize;
    }
  }
  // copy the GBTTrigger and GBTHeader from the head of the payload
  header.resize(wordSize);
  memcpy(header.data(), &data[0], wordSize);

  // copy the GBTTrailer from the end of the payload
  trailer.resize(wordSize);
  memcpy(trailer.data(), trailPtr, wordSize);
  GBTDataTrailer& gbtTrailer = *reinterpret_cast<GBTDataTrailer*>(&trailer[0]);
  gbtTrailer.packetDone = false; // intermediate trailers should not have done=true
  //  gbtTrailer.lanesStops = 0;     // intermediate trailers should not have lanes closed // RS CURRENTLY NOT USED

  return actualSize;
}

///______________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::newRDHMethod(const header::RDHAny* rdh, bool empty, std::vector<char>& toAdd) const
{
  // these method is called by the writer when it opens a new RDH page to fill some data.
  // empty tells if the previous RDH page had some payload (to differentiate between automatic open/close of empty RDH pages, handled by
  // the emptyHBFFunc and read data filling
  const int wordSize = RDHUtils::getDataFormat(rdh) == 0 ? o2::itsmft::GBTPaddedWordLength : o2::itsmft::GBTWordLength;
  if (RDHUtils::getStop(rdh)) { // the RDH was added to close previous HBF, do we want to add diagnostic data?
    if (!empty) {
      GBTDiagnostic diag;
      toAdd.resize(wordSize);
      memcpy(toAdd.data(), diag.getW8(), wordSize);
    }
  } else { // we need to add GBTDataHeader
    toAdd.resize(wordSize);
    memcpy(toAdd.data(), mFEEId2GBTHeader.find(RDHUtils::getFEEID(rdh))->second.getW8(), wordSize);
  }
}

template class o2::itsmft::MC2RawEncoder<o2::itsmft::ChipMappingITS>;
template class o2::itsmft::MC2RawEncoder<o2::itsmft::ChipMappingMFT>;
