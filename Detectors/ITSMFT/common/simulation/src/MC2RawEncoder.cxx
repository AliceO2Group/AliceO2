// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITSMFTSimulation/MC2RawEncoder.h"
#include "CommonConstants/Triggers.h"

using namespace o2::itsmft;

///______________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::init()
{
  if (mROMode == NotSet) {
    LOG(FATAL) << "Readout Mode must be set explicitly via setContinuousReadout(bool)";
  }
  // limit RUs to convert to existing ones
  mRUSWMax = (mRUSWMax < uint8_t(mMAP.getNRUs())) ? mRUSWMax : mMAP.getNRUs() - 1;
  //
  mNLinks = 0;
  for (uint8_t ru = mRUSWMin; ru <= mRUSWMax; ru++) {
    auto& ruData = getCreateRUDecode(ru);
    int nLinks = 0;
    for (int il = 0; il < MaxLinksPerRU; il++) {
      if (ruData.links[il]) {
        nLinks++;
        if (ruData.links[il]->packetCounter < 0) {
          ruData.links[il]->packetCounter = 0; // reset only once
        }
      }
    }
    mNLinks += nLinks;
    if (!nLinks) {
      LOG(INFO) << "Imposing single link readout for RU " << int(ru);
      ruData.links[0] = std::make_unique<GBTLink>();
      ruData.links[0]->lanes = mMAP.getCablesOnRUType(ruData.ruInfo->ruType);
      ruData.links[0]->id = 0;
      ruData.links[0]->cruID = ru;
      ruData.links[0]->feeID = mMAP.RUSW2FEEId(ruData.ruInfo->idSW, 0);
      ruData.links[0]->packetCounter = 0;
      mNLinks++;
    }
  }

  mLastIR = mHBFUtils.getFirstIR(); // TFs are counted from this IR

  assert(mNLinks > 0);
}

///______________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::finalize()
{
  // close open HBFs, write empty HBFs until the end of the TF and close all streams

  int tf = mHBFUtils.getTF(mCurrIR);
  mCurrIR = mHBFUtils.getIRTF(tf + 1) - 1; // last IR of the current TF
  mHBFUtils.fillHBIRvector(mHBIRVec, mLastIR, mCurrIR);
  for (const auto& ir : mHBIRVec) {
    mLastIR = ir;
    mRDH = mHBFUtils.createRDH<RDH>(mLastIR);
    openHBF(); // open new HBF for all links
  }
  mLastIR++;       // next call to fillHBIRvector should start from yet non-processed IR
  closeHBF();      // close all HBFs
  flushAllLinks(); // write data remaining in the link
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

  // get list of IR's for which HBF should be generated
  mHBFUtils.fillHBIRvector(mHBIRVec, mLastIR, mCurrIR);
  for (const auto& ir : mHBIRVec) {
    mLastIR = ir;
    mRDH = mHBFUtils.createRDH<RDH>(mLastIR);
    openHBF(); // open new HBF for all links
  }
  mLastIR++; // next call to fillHBIRvector should start from yet non-processed IR

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
  ru.cableHWID[chip.cableSW] = chip.cableHW; // register the cable HW ID
  auto& pixels = chipData.getData();
  std::sort(pixels.begin(), pixels.end(),
            [](auto lhs, auto rhs) {
              return (lhs.getRow() < rhs.getRow()) ? true : ((lhs.getRow() > rhs.getRow()) ? false : (lhs.getCol() < rhs.getCol()));
            });
  ru.cableData[chip.cableSW].ensureFreeCapacity(40 * (2 + pixels.size())); // make sure buffer has enough capacity
  mCoder.encodeChip(ru.cableData[chip.cableSW], chipData, chip.chipOnModuleHW, mCurrIR.bc);
}

//___________________________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::openPageLinkHBF(GBTLink& link)
{
  /// create 1st page of the new HBF
  if (mImposeROModeFlag) {
    mRDH.triggerType |= (mROMode == Continuous) ? o2::trigger::SOC : o2::trigger::SOT;
  }
  if (mRDH.triggerType & o2::trigger::TF) {
    if (mVerbose) {
      LOG(INFO) << "Starting new TF for link FEEId 0x" << std::hex << std::setfill('0') << std::setw(6) << link.feeID;
    }
    if (mStartTFOnNewSPage && link.lastPageSize > 0) { // new TF being started, need to start new superpage
      flushLinkSuperPage(link);
    }
  } else if (link.data.getSize() >= mSuperPageSize) {
    flushLinkSuperPage(link);
  }

  link.data.ensureFreeCapacity(MaxGBTPacketBytes + sizeof(RDH));
  RDH rdh = mRDH; // new RDH to open
  rdh.detectorField = mMAP.getRUDetectorField();
  rdh.feeId = link.feeID;
  rdh.linkID = link.id;
  rdh.cruID = link.cruID;
  rdh.pageCnt = 0;
  rdh.stop = 0;
  rdh.packetCounter = link.packetCounter++;
  link.lastRDH = reinterpret_cast<RDH*>(link.data.getEnd()); // pointer on last (to be written RDH for HBopen)
  link.data.addFast(reinterpret_cast<uint8_t*>(&rdh), sizeof(RDH)); // write RDH for new packet
}

//______________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::convertEmptyChips(int fromChip, int uptoChip, RUDecodeData& ru)
{
  // add empty chip words to respective cable's buffers for all chips of the current RU container
  for (int chipIDSW = fromChip; chipIDSW < uptoChip; chipIDSW++) { // flag chips w/o data
    const auto& chip = *mMAP.getChipOnRUInfo(ru.ruInfo->ruType, chipIDSW);
    ru.cableHWID[chip.cableSW] = chip.cableHW; // register the cable HW ID
    ru.cableData[chip.cableSW].ensureFreeCapacity(100);
    mCoder.addEmptyChip(ru.cableData[chip.cableSW], chip.chipOnModuleHW, mCurrIR.bc);
  }
}

//___________________________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::addPageLinkHBF(GBTLink& link, bool stop)
{
  /// Add new page (RDH) to existing one for the link (possibly stop page)
  if (!link.lastRDH) {
    return;
  }
  // check if the superpage reached the size where it hase to be flushed
  if (link.data.getSize() >= mSuperPageSize) {
    flushLinkSuperPage(link);
  }
  auto rdh = link.lastRDH;
  rdh->memorySize = link.data.getEnd() - reinterpret_cast<uint8_t*>(rdh); // set the size for the previous header RDH
  rdh->offsetToNext = rdh->memorySize;
  link.lastPageSize = link.data.getSize(); // register current size of the superpage

  // pointer on last (to be written) RDH for HBclose
  rdh = reinterpret_cast<RDH*>(link.data.getEnd()); // fetch pointer on to-be-written new RDH

  link.data.addFast(reinterpret_cast<uint8_t*>(link.lastRDH), sizeof(RDH)); // copy RDH for old packet as a prototope for the new one
  rdh->packetCounter = link.packetCounter++;
  if (mVerbose) {
    LOG(INFO) << "Prev HBF for link FEEId 0x" << std::hex << std::setfill('0') << std::setw(6) << link.feeID;
    if (mVerbose > 1) {
      mHBFUtils.printRDH(*link.lastRDH);
    }
  }
  rdh->offsetToNext = rdh->headerSize; // these settings will be correct only for the empty page, in case of payload the offset/size
  rdh->memorySize = rdh->headerSize;   // will be set at the next call of this method
  rdh->pageCnt++;
  rdh->stop = stop ? 0x1 : 0;
  if (mVerbose) {
    LOG(INFO) << (stop ? "Stop" : "Add") << " HBF for link FEEId 0x"
              << std::hex << std::setfill('0') << std::setw(6) << link.feeID << " Pages: " << link.nTriggers;
    if (mVerbose > 1 && stop) {
      mHBFUtils.printRDH(*rdh);
    }
  }
  link.lastRDH = rdh;                      // redirect to newly written RDH
  link.nTriggers++;                        // number of pages on this superpage
  if (stop) {
    link.lastRDH = nullptr; // after closing it is not valid anymore
  } else {
    size_t offs = reinterpret_cast<uint8_t*>(link.lastRDH) - link.data.getPtr();
    link.data.ensureFreeCapacity(MaxGBTPacketBytes + sizeof(RDH)); // make sure there is a room for next page
    link.lastRDH = reinterpret_cast<RDH*>(&link.data[offs]);       // fix link.lastRDH if the array was relocated
  }
  //
}

//___________________________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::openHBF()
{
  // open new HBF with mRDH
  for (int ruSW = mRUSWMin; ruSW <= mRUSWMax; ruSW++) {
    auto* ru = getRUDecode(ruSW);
    for (int il = 0; il < MaxLinksPerRU; il++) {
      auto link = ru->links[il].get();
      if (!link) {
        continue;
      }
      closePageLinkHBF(*link);
      openPageLinkHBF(*link);
    }
  }
  if (mImposeROModeFlag) {
    mRDH.triggerType &= ~(o2::trigger::SOT | o2::trigger::SOC);
    mImposeROModeFlag = false;
  }
}

//___________________________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::closeHBF()
{
  // close all open HBFs
  for (int ruSW = mRUSWMin; ruSW <= mRUSWMax; ruSW++) {
    auto* ru = getRUDecode(ruSW);
    RDH rdh = mRDH; // new RDH to open
    rdh.detectorField = mMAP.getRUDetectorField();
    for (int il = 0; il < MaxLinksPerRU; il++) {
      auto link = ru->links[il].get();
      if (link) {
        closePageLinkHBF(*link);
      }
    }
  }
}

//___________________________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::flushAllLinks()
{
  // write data remaining in the link
  for (int ruSW = mRUSWMin; ruSW <= mRUSWMax; ruSW++) {
    auto* ru = getRUDecode(ruSW);
    RDH rdh = mRDH; // new RDH to open
    rdh.detectorField = mMAP.getRUDetectorField();
    for (int il = 0; il < MaxLinksPerRU; il++) {
      auto link = ru->links[il].get();
      if (link) {
        closePageLinkHBF(*link);
        flushLinkSuperPage(*link);
      }
    }
  }
}

//___________________________________________________________________________________
template <class Mapping>
void MC2RawEncoder<Mapping>::flushLinkSuperPage(GBTLink& link, FILE* outFl)
{
  // write link superpage data to file
  if (!outFl) {
    outFl = mOutFile; // by default use global file
  }
  if (mVerbose) {
    LOG(INFO) << "Flushing super page for link FEEId 0x" << std::hex << std::setfill('0') << std::setw(6)
              << link.feeID << std::dec << std::setfill(' ') << " | size= " << link.lastPageSize << " bytes";
    LOG(INFO) << "ptr=0x" << (void*)link.data.getPtr() << " head=0x" << (void*)link.data.data() << " lastRDH = 0x" << (void*)link.lastRDH;
  }
  const auto ptr = link.data.getPtr();                     // beginning of the buffer
  fwrite(link.data.getPtr(), 1, link.lastPageSize, outFl); //write to file
  link.data.setPtr(ptr + link.lastPageSize);
  if (link.lastRDH) {
    link.lastRDH = reinterpret_cast<RDH*>(reinterpret_cast<uint8_t*>(link.lastRDH) - link.lastPageSize);
  }
  link.data.moveUnusedToHead(); // bring non-flushed data to top of the buffer
  link.lastPageSize = 0;        // will be updated on following closePageLinkHBF
  link.nTriggers = 0;
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

  for (int il = 0; il < MaxLinksPerRU; il++) {
    auto link = ru.links[il].get();
    if (!link) {
      continue;
    }
    // number of words written (RDH included) on current page
    int nGBTWords = int(link->data.getEnd() - reinterpret_cast<const uint8_t*>(link->lastRDH)) / GBTPaddedWordLength;
    // estimate real payload size in GBT words
    int nPayLoadWordsNeeded = 0;
    for (int icab = ru.nCables; icab--;) { // calculate number of GBT words per link
      if ((link->lanes & (0x1 << icab))) {
        int nb = ru.cableData[icab].getSize();
        nPayLoadWordsNeeded += nb ? 1 + (nb - 1) / 9 : 0; // single GBT word carries at most 9 payload bytes
      }
    }
    if (nGBTWords >= MaxGBTWordsPerPacket - 3 - (nPayLoadWordsNeeded > 0)) {
      // we write on same page if there is a room for trigger, header and trailer words + at least 1 payload word (if any)
      addPageLinkHBF(*link);                                                                                        // open new page for the same trigger, repeating RDH of the same HBF
      nGBTWords = int(link->data.getEnd() - reinterpret_cast<const uint8_t*>(link->lastRDH)) / GBTPaddedWordLength; // update counters
    }
    link->data.addFast(gbtTrigger.getW8(), GBTPaddedWordLength); // write GBT trigger
    nGBTWords++;

    GBTDataHeader gbtHeader;
    gbtHeader.packetIdx = 0;
    gbtHeader.activeLanes = link->lanes;
    link->data.addFast(gbtHeader.getW8(), GBTPaddedWordLength); // write GBT header
    nGBTWords++;
    //
    GBTDataTrailer gbtTrailer; // lanes will be set on closing the trigger

    // now loop over the lanes served by this link, writing each time at most 9 bytes, untill all lanes are copied
    do {
      for (int icab = 0; icab < ru.nCables; icab++) {
        if ((link->lanes & (0x1 << icab))) {
          auto& cableData = ru.cableData[icab];
          int nb = cableData.getUnusedSize();
          if (!nb) {
            continue; // write 80b word only if there is something to write
          }
          if (nb > 9) {
            nb = 9;
          }
          int gbtWordStart = link->data.getSize();                                                       // beginning of the current GBT word in the link
          link->data.addFast(cableData.getPtr(), nb);                                                    // fill payload of cable
          link->data.addFast(zero16, GBTPaddedWordLength - nb);                                          // fill the rest of the GBT word by 0
          link->data[gbtWordStart + 9] = mMAP.getGBTHeaderRUType(ru.ruInfo->ruType, ru.cableHWID[icab]); // set cable flag
          cableData.setPtr(cableData.getPtr() + nb);
          nPayLoadWordsNeeded--;                                           // number of payload GBT words left
          if (++nGBTWords == MaxGBTWordsPerPacket - 1) {                   // check if current GBT packet must be created (reserve 1 word for trailer)
            if (nPayLoadWordsNeeded) {                                     // we need to write the rest of the data on the new page
              link->data.add(gbtTrailer.getW8(), GBTPaddedWordLength);     // write empty GBT trailer for current packet
              addPageLinkHBF(*link);                                       // open new page for the same trigger, repeating RDH of the same HBF
              link->data.addFast(gbtTrigger.getW8(), GBTPaddedWordLength); // repeate GBT trigger word
              gbtHeader.packetIdx++;
              link->data.addFast(gbtHeader.getW8(), GBTPaddedWordLength); // repeate GBT header
              // update counters
              nGBTWords = int(link->data.getEnd() - reinterpret_cast<const uint8_t*>(link->lastRDH)) / GBTPaddedWordLength;
            }
          }
        } // storing data of single cable
      }   // loop over cables of this link
    } while (nPayLoadWordsNeeded);
    //
    // all payload was dumped, write final trailer

    gbtTrailer.lanesStops = link->lanes;
    gbtTrailer.packetDone = true;
    link->data.addFast(gbtTrailer.getW8(), GBTPaddedWordLength); // write GBT trailer for the last packet

  } // loop over links of RU
  ru.clearTrigger();
  ru.nChipsFired = 0;
}

///______________________________________________________________________
template <class Mapping>
RUDecodeData& MC2RawEncoder<Mapping>::getCreateRUDecode(int ruSW)
{
  assert(ruSW < mMAP.getNRUs());
  if (mRUEntry[ruSW] < 0) {
    mRUEntry[ruSW] = mNRUs++;
    mRUDecodeVec[mRUEntry[ruSW]].ruInfo = mMAP.getRUInfoSW(ruSW); // info on the stave/RU
    LOG(INFO) << "Defining container for RU " << ruSW << " at slot " << mRUEntry[ruSW];
  }
  return mRUDecodeVec[mRUEntry[ruSW]];
}

template class o2::itsmft::MC2RawEncoder<o2::itsmft::ChipMappingITS>;
template class o2::itsmft::MC2RawEncoder<o2::itsmft::ChipMappingMFT>;
