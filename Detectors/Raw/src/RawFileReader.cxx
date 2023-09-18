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

/// @file   RawFileReader.h
/// @author ruben.shahoyan@cern.ch
/// @brief  Reader for (multiple) raw data files

#include <algorithm>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <iostream>
#include "DetectorsRaw/RawFileReader.h"
#include "Headers/DAQID.h"
#include "CommonConstants/Triggers.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DetectorsRaw/HBFUtils.h"
#include "Framework/Logger.h"

#include <Common/Configuration.h>
#include <TStopwatch.h>
#include <fcntl.h>

using namespace o2::raw;
namespace o2h = o2::header;

//====================== methods of LinkBlock ========================
//____________________________________________
void RawFileReader::LinkBlock::print(const std::string& pref) const
{
  LOGF(info, "%sfile:%3d offs:%10zu size:%8d newSP:%d newTF:%d newHB:%d endHB:%d | Orbit %u TF %u",
       pref, fileID, offset, size, testFlag(StartSP), testFlag(StartTF), testFlag(StartHB),
       testFlag(EndHB), ir.orbit, tfID);
}

//====================== methods of LinkData ========================

//____________________________________________
std::string RawFileReader::LinkData::describe() const
{
  std::stringstream ss;
  ss << "Link " << origin.as<std::string>() << '/' << description.as<std::string>() << "/0x"
     << std::hex << std::setw(8) << std::setfill('0') << subspec
     << " RO: " << (continuousRO ? "Cont" : "Trig");
  return ss.str();
}

//____________________________________________
void RawFileReader::LinkData::print(bool verbose, const std::string& pref) const
{
  LOGF(info, "%s %s FEE:0x%04x CRU:%4d Lnk:%3d EP:%d RDHv%d Src:%s | SPages:%4d Pages:%6d TFs:%6d with %6d HBF in %4d blocks (%d err)",
       pref, describe(), int(RDHUtils::getFEEID(rdhl)), int(RDHUtils::getCRUID(rdhl)), int(RDHUtils::getLinkID(rdhl)),
       int(RDHUtils::getEndPointID(rdhl)), int(RDHUtils::getVersion(rdhl)),
       RDHUtils::getVersion(rdhl) > 5 ? o2h::DAQID::DAQtoO2(RDHUtils::getSourceID(rdhl)).str : "N/A",
       nSPages, nCRUPages, nTimeFrames, nHBFrames, int(blocks.size()), nErrors);
  if (verbose) {
    for (int i = 0; i < int(blocks.size()); i++) {
      std::stringstream counts;
      counts << '#' << std::setw(5) << i << " | ";
      blocks[i].print(counts.str());
    }
  }
}

//____________________________________________
size_t RawFileReader::LinkData::getNextTFSuperPagesStat(std::vector<RawFileReader::PartStat>& parts) const
{
  // get stat. of superpages for this link in this TF. We treat as a start of a superpage the discontinuity in the link data, new TF
  // or continuous data exceeding a threshold (e.g. 1MB)
  if (nextBlock2Read >= 0) { // negative nextBlock2Read signals absence of data
    int sz = 0, nSP = 0, ibl = nextBlock2Read, nbl = blocks.size(), nblPart = 0;
    parts.clear();
    while (ibl < nbl && (blocks[ibl].tfID == blocks[nextBlock2Read].tfID)) {
      if (ibl > nextBlock2Read && (blocks[ibl].testFlag(LinkBlock::StartSP) ||
                                   (sz + blocks[ibl].size) > reader->mNominalSPageSize ||
                                   (blocks[ibl - 1].offset + blocks[ibl - 1].size) < blocks[ibl].offset)) { // new superpage
        parts.emplace_back(RawFileReader::PartStat{sz, nblPart});
        sz = 0;
        nblPart = 0;
      }
      sz += blocks[ibl].size;
      nblPart++;
      ibl++;
    }
    if (sz) {
      parts.emplace_back(RawFileReader::PartStat{sz, nblPart});
    }
  }
  return parts.size();
}

//____________________________________________
size_t RawFileReader::LinkData::getNextHBFSize() const
{
  // estimate the memory size of the next HBF to read
  // The blocks are guaranteed to not cover more than 1 HB
  size_t sz = 0;
  if (nextBlock2Read >= 0) { // negative nextBlock2Read signals absence of data
    int ibl = nextBlock2Read, nbl = blocks.size();
    while (ibl < nbl && (blocks[ibl].ir == blocks[nextBlock2Read].ir)) {
      sz += blocks[ibl].size;
      ibl++;
    }
  }
  return sz;
}

//____________________________________________
size_t RawFileReader::LinkData::readNextHBF(char* buff)
{
  // read data of the next complete HB, buffer of getNextHBFSize() must be allocated in advance
  size_t sz = 0;
  if (nextBlock2Read < 0) { // negative nextBlock2Read signals absence of data
    return sz;
  }
  int ibl = nextBlock2Read, nbl = blocks.size();
  bool error = false;
  while (ibl < nbl) {
    auto& blc = blocks[ibl];
    if (blc.ir != blocks[nextBlock2Read].ir) {
      break;
    }
    ibl++;
    if (blc.dataCache) {
      memcpy(buff + sz, blc.dataCache.get(), blc.size);
    } else {
      auto fl = reader->mFiles[blc.fileID];
      if (fseek(fl, blc.offset, SEEK_SET) || fread(buff + sz, 1, blc.size, fl) != blc.size) {
        LOGF(error, "Failed to read for the %s a bloc:", describe());
        blc.print();
        error = true;
      } else if (reader->mCacheData) { // need to fill the cache at 1st reading
        blc.dataCache = std::make_unique<char[]>(blc.size);
        memcpy(blc.dataCache.get(), buff + sz, blc.size); // will be used at next reading
      }
    }
    sz += blc.size;
  }
  nextBlock2Read = ibl;
  return error ? 0 : sz; // in case of the error we ignore the data
}

//____________________________________________
size_t RawFileReader::LinkData::skipNextHBF()
{
  // skip next complete HB
  size_t sz = 0;
  if (nextBlock2Read < 0) { // negative nextBlock2Read signals absence of data
    return sz;
  }
  int ibl = nextBlock2Read, nbl = blocks.size();
  while (ibl < nbl) {
    const auto& blc = blocks[ibl];
    if (blc.ir.orbit != blocks[nextBlock2Read].ir.orbit) {
      break;
    }
    ibl++;
    sz += blc.size;
  }
  nextBlock2Read = ibl;
  return sz;
}

//____________________________________________
size_t RawFileReader::LinkData::getNextTFSize() const
{
  // estimate the memory size of the next TF to read
  // (assuming nextBlock2Read is at the start of the TF)
  size_t sz = 0;
  if (nextBlock2Read >= 0) { // negative nextBlock2Read signals absence of data
    int ibl = nextBlock2Read, nbl = blocks.size();
    while (ibl < nbl && (blocks[ibl].tfID == blocks[nextBlock2Read].tfID)) {
      sz += blocks[ibl].size;
      ibl++;
    }
  }
  return sz;
}

//_____________________________________________________________________
size_t RawFileReader::LinkData::readNextTF(char* buff)
{
  // read next complete TF, buffer of getNextTFSize() must be allocated in advance
  size_t sz = 0;
  if (nextBlock2Read < 0) { // negative nextBlock2Read signals absence of data
    return sz;
  }
  int ibl0 = nextBlock2Read, nbl = blocks.size();
  bool error = false;
  while (nextBlock2Read < nbl && (blocks[nextBlock2Read].tfID == blocks[ibl0].tfID)) { // nextBlock2Read is incremented by the readNextHBF!
    auto szb = readNextHBF(buff + sz);
    if (!szb) {
      error = true;
    }
    sz += szb;
  }
  return error ? 0 : sz; // in case of the error we ignore the data
}

//_____________________________________________________________________
size_t RawFileReader::LinkData::skipNextTF()
{
  // skip next complete TF
  size_t sz = 0;
  if (nextBlock2Read < 0) { // negative nextBlock2Read signals absence of data
    return sz;
  }
  int ibl0 = nextBlock2Read, nbl = blocks.size();
  bool error = false;
  while (nextBlock2Read < nbl && (blocks[nextBlock2Read].tfID == blocks[ibl0].tfID)) { // nextBlock2Read is incremented by the readNextHBF!
    auto szb = skipNextHBF();
    if (!szb) {
      error = true;
    }
    sz += szb;
  }
  return error ? 0 : sz; // in case of the error we ignore the data
}

//_____________________________________________________________________
bool RawFileReader::LinkData::rewindToTF(uint32_t tf)
{
  // go to given TF
  if (tf < tfStartBlock.size()) {
    nextBlock2Read = tfStartBlock[tf].first;
  } else {
    LOG(warning) << "No TF " << tf << " for " << describe();
    nextBlock2Read = -1;
    return false;
  }
  return true;
}

//____________________________________________
int RawFileReader::LinkData::getNHBFinTF() const
{
  // estimate number of HBFs left in the TF
  int ibl = nextBlock2Read, nbl = blocks.size(), nHB = 0;
  if (nextBlock2Read >= 0) { // negative nextBlock2Read signals absence of data
    while (ibl < nbl && (blocks[ibl].tfID == blocks[nextBlock2Read].tfID)) {
      if (blocks[ibl].testFlag(LinkBlock::StartHB)) {
        nHB++;
      }
      ibl++;
    }
  }
  return nHB;
}

//____________________________________________
size_t RawFileReader::LinkData::readNextSuperPage(char* buff, const RawFileReader::PartStat* pstat)
{
  // read data of the next complete HB, buffer of getNextHBFSize() must be allocated in advance
  size_t sz = 0;
  if (nextBlock2Read < 0) { // negative nextBlock2Read signals absence of data
    return sz;
  }
  int ibl = nextBlock2Read, nbl = blocks.size();
  auto tfID = blocks[nextBlock2Read].tfID;
  bool error = false;
  if (pstat) { // info is provided, use it derictly
    sz = pstat->size;
    ibl += pstat->nBlocks;
  } else { // need to calculate blocks to read
    while (ibl < nbl) {
      auto& blc = blocks[ibl];
      if (ibl > nextBlock2Read && (blc.tfID != blocks[nextBlock2Read].tfID ||
                                   blc.testFlag(LinkBlock::StartSP) ||
                                   (sz + blc.size) > reader->mNominalSPageSize ||
                                   blocks[ibl - 1].offset + blocks[ibl - 1].size < blc.offset)) { // new superpage or TF
        break;
      }
      ibl++;
      sz += blc.size;
    }
  }
  if (sz) {
    if (reader->mCacheData && blocks[nextBlock2Read].dataCache) {
      memcpy(buff, blocks[nextBlock2Read].dataCache.get(), sz);
    } else {
      auto fl = reader->mFiles[blocks[nextBlock2Read].fileID];
      if (fseek(fl, blocks[nextBlock2Read].offset, SEEK_SET) || fread(buff, 1, sz, fl) != sz) {
        LOGF(error, "Failed to read for the %s a bloc:", describe());
        blocks[nextBlock2Read].print();
        error = true;
      } else if (reader->mCacheData) { // cache after 1st reading
        blocks[nextBlock2Read].dataCache = std::make_unique<char[]>(sz);
        memcpy(blocks[nextBlock2Read].dataCache.get(), buff, sz);
      }
    }
  }
  nextBlock2Read = ibl;
  return error ? 0 : sz; // in case of the error we ignore the data
}

//____________________________________________
size_t RawFileReader::LinkData::getLargestSuperPage() const
{
  // estimate largest super page size
  size_t szMax = 0, szLast = 0;
  for (const auto& bl : blocks) {
    if (bl.testFlag(LinkBlock::StartSP)) { // account previous SPage and start accumulation of the next one
      if (szLast > szMax) {
        szMax = szLast;
      }
      szLast = 0;
    }
    szLast += bl.size;
  }
  return szLast > szMax ? szLast : szMax;
}

//____________________________________________
size_t RawFileReader::LinkData::getLargestTF() const
{
  // estimate largest TF
  size_t szMax = 0, szLast = 0;
  for (const auto& bl : blocks) {
    if (bl.testFlag(LinkBlock::StartTF)) { // account previous TF and start accumulation of the next one
      if (szLast > szMax) {
        szMax = szLast;
      }
      szLast = 0;
    }
    szLast += bl.size;
  }
  return szLast > szMax ? szLast : szMax;
}

//_____________________________________________________________________
bool RawFileReader::LinkData::preprocessCRUPage(const RDHAny& rdh, bool newSPage)
{
  // account RDH in statistics
  bool ok = true;
  bool newTF = false, newHB = false;
  const auto& HBU = HBFUtils::Instance();

  if (RDHUtils::getFEEID(rdh) != RDHUtils::getFEEID(rdhl)) { // make sure links with different FEEID were not assigned same subspec
    LOGF(error, "Same SubSpec is found for %s with different RDH.feeId", describe());
    LOGF(error, "old RDH assigned SubSpec=0x%-8d:", subspec);
    RDHUtils::dumpRDH(rdhl);
    LOGF(error, "new RDH assigned SubSpec=0x%-8d:", subspec);
    RDHUtils::dumpRDH(rdh);
    throw std::runtime_error("Conflicting SubSpecs are provided");
    ok = false;
    nErrors++;
  }
  auto ir = RDHUtils::getTriggerIR(rdh);
  auto pageCnt = RDHUtils::getPageCounter(rdh);

  if (pageCnt == 0) {
    auto triggerType = RDHUtils::getTriggerType(rdh);
    if (!nCRUPages) { // 1st page, expect SOX
      if (triggerType & o2::trigger::SOC) {
        continuousRO = true;
        irOfSOX = ir;
      } else if (triggerType & o2::trigger::SOT) {
        continuousRO = false;
        irOfSOX = ir;
      } else {
        if (reader->mCheckErrors & (0x1 << ErrNoSOX)) {
          LOG(error) << ErrNames[ErrNoSOX];
          ok = false;
          nErrors++;
        }
      }
      if (!irOfSOX.isDummy() && reader->getTFAutodetect() == FirstTFDetection::Pending) {
        reader->imposeFirstTF(irOfSOX.orbit);
      }
    }
    auto newTFCalc = reader->getTFAutodetect() != FirstTFDetection::Pending && (blocks.empty() || HBU.getTF(blocks.back().ir) != HBU.getTF(ir)); // TF change
    if (cruDetector) {
      newTF = (triggerType & o2::trigger::TF);
      newHB = (triggerType & o2::trigger::HB);
      if (newTFCalc != newTF && (reader->mCheckErrors & (0x1 << ErrMismatchTF))) {
        LOG(error) << ErrNames[ErrMismatchTF];
        ok = false;
        nErrors++;
      }
      if (reader->mPreferCalculatedTFStart) {
        newTF = newTFCalc;
        if (newTF) {
          newHB = true;
        }
      }
    } else {
      newHB = true; // in RORC detectors treat each trigger as a HBF
      if (newTFCalc) {
        newTF = true;
        // continuousRO = false;
      }
    }
  } else if (reader->mCheckErrors & (0x1 << ErrWrongPageCounterIncrement)) {
    // check increasing pageCnt
    if (nCRUPages && (pageCnt != (RDHUtils::getPageCounter(rdhl) + 1))) { // skip for very 1st page
      LOG(error) << ErrNames[ErrWrongPageCounterIncrement]
                 << " old=" << int(pageCnt) << " new=" << int(RDHUtils::getPageCounter(rdhl));
      ok = false;
      nErrors++;
    }
  }

  if (reader->mCheckErrors) {
    if (nCRUPages) {
      // check increasing (or wrapping) packetCounter
      auto packetCounter = RDHUtils::getPacketCounter(rdh);
      auto packetCounterL = RDHUtils::getPacketCounter(rdhl);
      if ((packetCounter != ((packetCounterL + 1) & 0xff)) &&
          (reader->mCheckErrors & (0x1 << ErrWrongPacketCounterIncrement))) { // skip for very 1st page
        LOG(error) << ErrNames[ErrWrongPacketCounterIncrement]
                   << " new=" << int(packetCounter) << " old=" << int(packetCounterL);
        ok = false;
        nErrors++;
      }
      // check if number of HBFs in the TF is as expected
      if (newTF) {
        if (nHBFinTF != HBFUtils::Instance().getNOrbitsPerTF() &&
            (reader->mCheckErrors & (0x1 << ErrWrongHBFsPerTF)) && cruDetector) {
          LOG(error) << ErrNames[ErrWrongHBFsPerTF] << ": "
                     << nHBFinTF << " instead of " << HBFUtils::Instance().getNOrbitsPerTF();
          ok = false;
          nErrors++;
        }
        nHBFinTF = 0; // reset
      }

    } else { // make sure data starts with TF and HBF
      if ((!newTF || !newHB || pageCnt != 0) &&
          (reader->mCheckErrors & (0x1 << ErrWrongFirstPage) && cruDetector)) {
        LOG(error) << ErrNames[ErrWrongFirstPage];
        ok = false;
        nErrors++;
      }
    }
  }
  auto stop = RDHUtils::getStop(rdh);
  auto hbIR = RDHUtils::getHeartBeatIR(rdh), hblIR = RDHUtils::getHeartBeatIR(rdhl);
  if (newHB) {
    if (reader->mCheckErrors) {
      nHBFinTF++;
      if (stop && (reader->mCheckErrors & (0x1 << ErrHBFStopOnFirstPage))) {
        LOG(error) << ErrNames[ErrHBFStopOnFirstPage] << " @ HBF#" << nHBFrames;
        ok = false;
        nErrors++;
      }
      if (openHB && (reader->mCheckErrors & (0x1 << ErrHBFNoStop)) && cruDetector) {
        LOG(error) << ErrNames[ErrHBFNoStop] << " @ HBF#" << nHBFrames;
        ok = false;
        nErrors++;
      }
      if ((reader->mCheckErrors & (0x1 << ErrHBFJump)) &&
          (nCRUPages && // skip this check for the very 1st RDH
           !(/*hbIR.bc == hblIR.bc &&*/ hbIR.orbit == hblIR.orbit + 1)) &&
          cruDetector) {
        LOG(error) << ErrNames[ErrHBFJump] << " @ HBF#" << nHBFrames << " New HB orbit/bc=" << hbIR.orbit << '/' << int(hbIR.bc)
                   << " is not incremented by 1 orbit wrt Old HB orbit/bc=" << hblIR.orbit << '/' << int(hblIR.bc);
        ok = false;
        nErrors++;
      }
    } // end of check errors
    openHB = true;
    nHBFrames++;
  }
  if (stop) {
    openHB = false;
  }

  if (cruDetector &&
      ((reader->getTFAutodetect() == FirstTFDetection::Pending && !newTF) ||
       (reader->getTFAutodetect() == FirstTFDetection::Done && ir.orbit < HBU.orbitFirst))) { // skip data until TF start is seen or orbit is less than determined 1st TF orbit
    LOG(error) << "skipping RDH w/o newTF flag until TF start is found";
    ok = false;
    newTF = newSPage = newHB = false;
  }

  if (newTF || newSPage || newHB) {
    if (newTF && reader->getTFAutodetect() == FirstTFDetection::Pending) {
      if (cruDetector) {
        reader->imposeFirstTF(hbIR.orbit);
      } else {
        throw std::runtime_error("HBFUtil first orbit/bc autodetection cannot be done with first link from CRORC detector");
      }
    }
    int nbl = blocks.size();
    auto& bl = blocks.emplace_back(reader->mCurrentFileID, reader->mPosInFile);
    bl.ir = hbIR;
    bl.tfID = HBU.getTF(hbIR); // nTimeFrames - 1;
    if (newTF) {
      tfStartBlock.emplace_back(nbl, bl.tfID);
      nTimeFrames++;
      bl.setFlag(LinkBlock::StartTF);
      if (reader->mCheckErrors & (0x1 << ErrNoSuperPageForTF) && cruDetector) {
        if (reader->mMultiLinkFile && !newSPage) {
          LOG(error) << ErrNames[ErrNoSuperPageForTF] << " @ TF#" << nTimeFrames;
          ok = false;
          nErrors++;
        }
      } // end of check errors
    }

    if (newSPage) {
      nSPages++;
      bl.setFlag(LinkBlock::StartSP);
    }
    if (newHB) {
      bl.setFlag(LinkBlock::StartHB);
    }
  }
  if (blocks.size()) {
    blocks.back().setFlag(LinkBlock::EndHB, stop); // last processed RDH defines this flag
    blocks.back().size += RDHUtils::getOffsetToNext(rdh);
    rdhl = rdh;
    nCRUPages++;
  }
  if (!ok) {
    LOG(error) << " ^^^Problem(s) was encountered at offset " << reader->mPosInFile << " of file " << reader->mCurrentFileID;
    RDHUtils::printRDH(rdh);
  } else if (reader->mVerbosity > 1) {
    if (reader->mVerbosity > 2) {
      RDHUtils::dumpRDH(rdh);
    } else {
      RDHUtils::printRDH(rdh);
    }
    LOG(info) << "--------------- reader tags: newTF: " << newTF << " newHBF/Trigger: " << newHB << " newSPage: " << newSPage;
  }
  return true;
}

//====================== methods of RawFileReader ========================

//_____________________________________________________________________
RawFileReader::RawFileReader(const std::string& config, int verbosity, size_t buffSize, const std::string& onlyDet) : mVerbosity(verbosity), mBufferSize(buffSize)
{
  if (!config.empty()) {
    auto inp = parseInput(config, onlyDet, true);
    loadFromInputsMap(inp);
  }
}

//_____________________________________________________________________
int RawFileReader::getLinkLocalID(const RDHAny& rdh, int fileID)
{
  // get id of the link subspec. in the parser (create entry if new)
  auto orig = std::get<0>(mDataSpecs[fileID]);
  LinkSubSpec_t subspec = RDHUtils::getSubSpec(rdh);
  LinkSpec_t spec = createSpec(orig, subspec);
  auto entryMap = mLinkEntries.find(spec);
  if (entryMap == mLinkEntries.end()) { // need to register a new link
    int n = mLinkEntries.size();
    mLinkEntries[spec] = n;
    auto& lnk = mLinksData.emplace_back(rdh, this);
    lnk.subspec = subspec;
    lnk.origin = orig;
    lnk.description = std::get<1>(mDataSpecs[fileID]);
    lnk.spec = spec;
    lnk.cruDetector = std::get<2>(mDataSpecs[fileID]) == CRU;
    return n;
  }
  return entryMap->second;
}

//_____________________________________________________________________
bool RawFileReader::preprocessFile(int ifl)
{
  // preprocess file, check RDH data, build statistics
  std::unique_ptr<char[]> buffer = std::make_unique<char[]>(mBufferSize);
  FILE* fl = mFiles[ifl];
  mCurrentFileID = ifl;
  LinkSpec_t specPrev = 0xffffffffffffffff;
  int lIDPrev = -1;
  mMultiLinkFile = false;
  fseek(fl, 0L, SEEK_END);
  const auto fileSize = ftell(fl);
  rewind(fl);
  long int nr = 0;
  mPosInFile = 0;
  size_t nRDHread = 0, boffs;
  bool readMore = true;
  while (readMore && (nr = fread(buffer.get(), 1, mBufferSize, fl))) {
    boffs = 0;
    while (1) {
      auto& rdh = *reinterpret_cast<RDHUtils::RDHAny*>(&buffer[boffs]);
      if ((mPosInFile + RDHUtils::getOffsetToNext(rdh)) > fileSize) {
        LOGP(warning, "File {} truncated current file pos {} + offsetToNext {} > fileSize {}", ifl, mPosInFile, RDHUtils::getOffsetToNext(rdh), fileSize);
        readMore = false;
        break;
      }
      nRDHread++;
      LinkSpec_t spec = createSpec(std::get<0>(mDataSpecs[mCurrentFileID]), RDHUtils::getSubSpec(rdh));
      int lID = lIDPrev;
      if (spec != specPrev) { // link has changed
        specPrev = spec;
        if (lIDPrev != -1) {
          mMultiLinkFile = true;
        }
        lID = getLinkLocalID(rdh, mCurrentFileID);
      }
      bool newSPage = lID != lIDPrev;
      try {
        mLinksData[lID].preprocessCRUPage(rdh, newSPage);
      } catch (...) {
        LOG(error) << "Corrupted data, abandoning processing";
        mStopProcessing = true;
        break;
      }

      if (mLinksData[lID].nTimeFrames && (mLinksData[lID].nTimeFrames - 1 > mMaxTFToRead)) { // limit reached, discard the last read
        mLinksData[lID].nTimeFrames--;
        mLinksData[lID].blocks.pop_back();
        if (mLinksData[lID].nHBFrames > 0) {
          mLinksData[lID].nHBFrames--;
        }
        if (mLinksData[lID].nCRUPages > 0) {
          mLinksData[lID].nCRUPages--;
        }
        lIDPrev = -1; // last block is closed
        readMore = false;
        break;
      }
      boffs += RDHUtils::getOffsetToNext(rdh);
      mPosInFile += RDHUtils::getOffsetToNext(rdh);
      lIDPrev = lID;
      if (boffs + sizeof(RDHUtils::RDHAny) >= nr) {
        if (fseek(fl, mPosInFile, SEEK_SET)) {
          readMore = false;
          break;
        }
        break;
      }
    }
  }
  LOGF(info, "File %3d : %9li bytes scanned, %6d RDH read for %4d links from %s",
       mCurrentFileID, mPosInFile, nRDHread, int(mLinkEntries.size()), mFileNames[mCurrentFileID]);
  return nRDHread > 0;
}

//_____________________________________________________________________
void RawFileReader::printStat(bool verbose) const
{
  int n = getNLinks();
  for (int i = 0; i < n; i++) {
    const auto& link = getLink(i);
    std::stringstream counts;
    counts << "Lnk" << std::left << std::setw(4) << i << "| ";
    link.print(verbose, counts.str());
  }
}

//_____________________________________________________________________
void RawFileReader::clear()
{
  mLinkEntries.clear();
  mOrderedIDs.clear();
  mLinksData.clear();
  for (auto fl : mFiles) {
    fclose(fl);
  }
  mFiles.clear();
  mFileNames.clear();

  mCurrentFileID = 0;
  mMultiLinkFile = false;
  mInitDone = false;
}

//_____________________________________________________________________
bool RawFileReader::addFile(const std::string& sname, o2::header::DataOrigin origin, o2::header::DataDescription desc, ReadoutCardType t)
{
  if (mInitDone) {
    LOG(error) << "Cannot add new files after initialization";
    return false;
  }
  bool ok = true;

  mFileBuffers.push_back(std::make_unique<char[]>(mBufferSize));
  auto inFile = fopen(sname.c_str(), "rb");
  if (!inFile) {
    LOG(error) << "Failed to open input file " << sname;
    return false;
  }
  setvbuf(inFile, mFileBuffers.back().get(), _IOFBF, mBufferSize);

  if (origin == o2h::gDataOriginInvalid) {
    LOG(error) << "Invalid data origin " << origin.as<std::string>() << " for file " << sname;
    ok = false;
  }
  if (desc == o2h::gDataDescriptionInvalid) {
    LOG(error) << "Invalid data description " << desc.as<std::string>() << " for file " << sname;
    ok = false;
  }
  if (!ok) {
    fclose(inFile);
    return false;
  }
  mFileNames.push_back(sname);
  mFiles.push_back(inFile);
  mDataSpecs.emplace_back(origin, desc, t);
  return true;
}

//_____________________________________________________________________
bool RawFileReader::init()
{
  // make initialization, preprocess files and chack for errors if asked

  for (int i = 0; i < NErrorsDefined; i++) {
    if (mCheckErrors & (0x1 << i)) {
      LOGF(info, "%s check for /%s/", (mCheckErrors & (0x1 << i)) ? "perform" : "ignore ", ErrNames[i].data());
    }
  }
  if (mMaxTFToRead < 0xffffffff) {
    LOGF(info, "at most %u TF will be processed", mMaxTFToRead);
  }

  int nf = mFiles.size();
  mEmpty = true;
  for (int i = 0; i < nf; i++) {
    if (preprocessFile(i)) {
      mEmpty = false;
    }
  }
  if (mStopProcessing) {
    LOG(error) << "Abandoning processing due to corrupted data";
    return false;
  }
  mOrderedIDs.resize(mLinksData.size());
  for (int i = mLinksData.size(); i--;) {
    mOrderedIDs[i] = i;
    if (mNTimeFrames < mLinksData[i].nTimeFrames) {
      mNTimeFrames = mLinksData[i].nTimeFrames;
    }
  }
  std::sort(mOrderedIDs.begin(), mOrderedIDs.end(),
            [&links = mLinksData](int a, int b) { return links[a].spec < links[b].spec; });

  size_t maxSP = 0, maxTF = 0;

  LOGF(info, "Summary of preprocessing:");
  for (int i = 0; i < int(mLinksData.size()); i++) {
    auto& link = getLink(i);
    auto msp = link.getLargestSuperPage();
    auto mtf = link.getLargestTF();
    if (maxSP < msp) {
      maxSP = msp;
    }
    if (maxTF < mtf) {
      maxTF = mtf;
    }
    std::stringstream counts;
    counts << "Lnk" << std::setw(4) << std::left << i << "| ";
    link.print(mVerbosity, counts.str());
    if (msp > mNominalSPageSize) {
      LOGF(debug, "       Attention: largest superpage %zu B exceeds expected %d B",
           msp, mNominalSPageSize);
    }
    // min max orbits
    if (link.blocks.front().ir.orbit < mOrbitMin) {
      mOrbitMin = link.blocks.front().ir.orbit;
    }
    if (link.blocks.back().ir.orbit > mOrbitMax) {
      mOrbitMax = link.blocks.back().ir.orbit;
    }
    if (link.tfStartBlock.empty() && !link.blocks.empty()) {
      link.tfStartBlock.emplace_back(0, 0);
    }
    if ((mCheckErrors & (0x1 << ErrWrongNumberOfTF)) && (mNTimeFrames != link.nTimeFrames)) {
      LOGF(error, "%s for %s: %u TFs while %u were seen for other links", ErrNames[ErrWrongNumberOfTF],
           link.describe(), link.nTimeFrames, mNTimeFrames);
    }
  }
  LOGF(info, "First orbit: %u, Last orbit: %u", mOrbitMin, mOrbitMax);
  LOGF(info, "Largest super-page: %zu B, largest TF: %zu B", maxSP, maxTF);
  if (!mCheckErrors) {
    LOGF(info, "Detailed data format check was disabled");
  }
  mInitDone = true;

  return !mEmpty;
}

//_____________________________________________________________________
o2h::DataOrigin RawFileReader::getDataOrigin(const std::string& ors)
{
  constexpr int NGoodOrigins = 21;
  constexpr std::array<o2h::DataOrigin, NGoodOrigins> goodOrigins{
    o2h::gDataOriginFLP, o2h::gDataOriginTST, o2h::gDataOriginCPV, o2h::gDataOriginCTP, o2h::gDataOriginEMC,
    o2h::gDataOriginFT0, o2h::gDataOriginFV0, o2h::gDataOriginFDD, o2h::gDataOriginHMP, o2h::gDataOriginITS,
    o2h::gDataOriginMCH, o2h::gDataOriginMFT, o2h::gDataOriginMID, o2h::gDataOriginPHS, o2h::gDataOriginTOF,
    o2h::gDataOriginTPC, o2h::gDataOriginTRD, o2h::gDataOriginZDC, o2h::gDataOriginFOC};

  for (auto orgood : goodOrigins) {
    if (ors == orgood.as<std::string>()) {
      return orgood;
    }
  }
  return o2h::gDataOriginInvalid;
}

//_____________________________________________________________________
o2h::DataDescription RawFileReader::getDataDescription(const std::string& ors)
{
  constexpr int NGoodDesc = 5;
  constexpr std::array<o2h::DataDescription, NGoodDesc> goodDesc{
    o2h::gDataDescriptionRawData, o2h::gDataDescriptionClusters, o2h::gDataDescriptionTracks,
    o2h::gDataDescriptionConfig, o2h::gDataDescriptionInfo};

  for (auto dgood : goodDesc) {
    if (ors == dgood.as<std::string>()) {
      return dgood;
    }
  }
  return o2h::gDataDescriptionInvalid;
}

//_____________________________________________________________________
void RawFileReader::loadFromInputsMap(const RawFileReader::InputsMap& inp)
{
  // load from already parsed input
  for (const auto& entry : inp) {
    const auto& ordesc = entry.first;
    const auto& files = entry.second;
    if (files.empty()) { // these are default origin and decription
      setDefaultDataOrigin(std::get<0>(ordesc));
      setDefaultDataDescription(std::get<1>(ordesc));
      setDefaultReadoutCardType(std::get<2>(ordesc));
      continue;
    }
    for (const auto& fnm : files) { // specific file names
      if (!addFile(fnm, std::get<0>(ordesc), std::get<1>(ordesc), std::get<2>(ordesc))) {
        throw std::runtime_error("wrong raw data file path or origin/description");
      }
    }
  }
}

//_____________________________________________________________________
RawFileReader::InputsMap RawFileReader::parseInput(const std::string& confUri, const std::string& onlyDet, bool verbose)
{
  // read input files from configuration
  std::map<OrigDescCard, std::vector<std::string>> entries;

  if (confUri.empty()) {
    throw std::runtime_error("Input configuration file is not provided");
  }
  std::string confFile = confUri;
  ConfigFile cfg;
  // ConfigFile file requires input as file:///absolute_path or file:local
  if (confFile.rfind("file:", 0) != 0) {
    confFile.insert(0, "file:");
  }
  try {
    cfg.load(confFile);
  } catch (std::string& e) { // unfortunately, the ConfigFile::load throws a string rather than the std::exception
    throw std::runtime_error(std::string("Failed to parse configuration ") + confFile + " : " + e);
  }
  //
  std::unordered_map<std::string, int> detFilter;
  auto msk = DetID::getMask(onlyDet);
  for (DetID::ID id = DetID::First; id <= DetID::Last; id++) {
    if (msk[id]) {
      detFilter[DetID::getName(id)] = 1;
    }
  }

  try {
    std::string origStr, descStr, cardStr, defstr = "defaults";
    cfg.getOptionalValue<std::string>(defstr + ".dataOrigin", origStr, DEFDataOrigin.as<std::string>());
    auto defDataOrigin = getDataOrigin(origStr);
    if (defDataOrigin == o2h::gDataOriginInvalid) {
      throw std::runtime_error(std::string("Invalid default data origin ") + origStr);
    }
    cfg.getOptionalValue<std::string>(defstr + ".dataDescription", descStr, DEFDataDescription.as<std::string>());
    auto defDataDescription = getDataDescription(descStr);
    if (defDataDescription == o2h::gDataDescriptionInvalid) {
      throw std::runtime_error(std::string("Invalid default data description ") + descStr);
    }
    auto defCardType = DEFCardType;
    cfg.getOptionalValue<std::string>(defstr + ".readoutCard", cardStr, std::string{CardNames[DEFCardType]});
    if (cardStr == CardNames[CRU]) {
      defCardType = CRU;
    } else if (cardStr == CardNames[RORC]) {
      defCardType = RORC;
    } else {
      throw std::runtime_error(std::string("Invalid default readout card ") + cardStr);
    }

    entries[{defDataOrigin, defDataDescription, defCardType}]; // insert
    LOG(debug) << "Setting default dataOrigin/Description/CardType " << defDataOrigin.as<std::string>() << '/' << defDataDescription.as<std::string>() << '/' << CardNames[defCardType];

    for (auto flsect : ConfigFileBrowser(&cfg, "input-")) {
      std::string flNameStr, defs{""};
      cfg.getOptionalValue<std::string>(flsect + ".dataOrigin", origStr, defDataOrigin.as<std::string>());
      cfg.getOptionalValue<std::string>(flsect + ".dataDescription", descStr, defDataDescription.as<std::string>());
      cfg.getOptionalValue<std::string>(flsect + ".filePath", flNameStr, defs);
      cfg.getOptionalValue<std::string>(flsect + ".readoutCard", cardStr, std::string{CardNames[CRU]});
      if (flNameStr.empty()) {
        LOG(debug) << "Skipping incomplete input " << flsect;
        continue;
      }
      auto dataOrigin = getDataOrigin(origStr);
      if (dataOrigin == o2h::gDataOriginInvalid) {
        throw std::runtime_error(std::string("Invalid data origin ") + origStr + " for " + flsect);
      }
      if (!detFilter.empty()) {
        int& sdet = detFilter[dataOrigin.as<std::string>()];
        if (sdet < 1) {
          if (sdet == 0 && verbose) { // print only once
            LOG(info) << "discarding data of detector " << dataOrigin.as<std::string>();
            sdet--;
          }
          continue;
        }
      }
      auto dataDescription = getDataDescription(descStr);
      if (dataDescription == o2h::gDataDescriptionInvalid) {
        throw std::runtime_error(std::string("Invalid data description ") + descStr + " for " + flsect);
      }

      auto cardType = defCardType;
      if (cardStr == CardNames[CRU]) {
        cardType = CRU;
      } else if (cardStr == CardNames[RORC]) {
        cardType = RORC;
      } else {
        throw std::runtime_error(std::string("Invalid default readout card ") + cardStr + " for " + flsect);
      }
      entries[{dataOrigin, dataDescription, cardType}].push_back(flNameStr);
      LOG(debug) << "adding file " << flNameStr << " to dataOrigin/Description " << dataOrigin.as<std::string>() << '/' << dataDescription.as<std::string>();
    }
  } catch (std::string& e) { // to catch exceptions from the parser
    throw std::runtime_error(std::string("Aborting due to the exception: ") + e);
  }

  return entries;
}

void RawFileReader::imposeFirstTF(uint32_t orbit)
{
  if (mFirstTFAutodetect != FirstTFDetection::Pending) {
    throw std::runtime_error("reader was not expecting imposing first TF");
  }
  auto& hbu = o2::raw::HBFUtils::Instance();
  o2::raw::HBFUtils::setValue("HBFUtils", "orbitFirst", orbit);
  LOG(info) << "Imposed data-driven TF start";
  mFirstTFAutodetect = FirstTFDetection::Done;
  hbu.printKeyValues();
}

std::string RawFileReader::nochk_opt(RawFileReader::ErrTypes e)
{
  std::string opt = ErrCheckDefaults[e] ? "nocheck-" : "check-";
  return opt + RawFileReader::ErrNamesShort[e].data();
}

std::string RawFileReader::nochk_expl(RawFileReader::ErrTypes e)
{
  std::string opt = ErrCheckDefaults[e] ? "ignore /" : "check  /";
  return opt + RawFileReader::ErrNames[e].data() + '/';
}
