// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <sstream>
#include <pair>
#include "DetectorsRaw/RawFileReader.h"
#include "CommonConstants/Triggers.h"
#include "DetectorsRaw/HBFUtils.h"
#include "Framework/Logger.h"
#include "Headers/DataHeader.h"

using namespace o2::raw;

//====================== methods of LinkBlock ========================
//____________________________________________
void RawFileReader::LinkBlock::print(const std::string pref) const
{
  LOGF(INFO, "%sfile:%3d offs:%10zu size:%8d newSP:%d newTF:%d newHB:%d endHB:%d | Orbit %u TF %u",
       pref, fileID, offset, size, testFlag(StartSP), testFlag(StartTF), testFlag(StartHB),
       testFlag(EndHB), orbit, tfID);
}

//====================== methods of LinkData ========================

//____________________________________________
RawFileReader::LinkData::LinkData(const o2::header::RAWDataHeaderV4& rdh, const RawFileReader* r) : reader(r)
{
  std::memcpy(&rdhl, &rdh, sizeof(rdh));
}

//____________________________________________
RawFileReader::LinkData::LinkData(const o2::header::RAWDataHeaderV5& rdh, const RawFileReader* r) : reader(r)
{
  std::memcpy(&rdhl, &rdh, sizeof(rdh));
}

//____________________________________________
void RawFileReader::LinkData::print(bool verbose, const std::string pref) const
{
  LOGF(INFO, "%sSSpec:0x%-8d FEE:0x%4x CRU:%4d Lnk:%3d EP:%d | SPages:%4d Pages:%6d TFs:%6d with %6d HBF in %4d blocks (%d err)",
       pref, subspec, int(rdhl.feeId), int(rdhl.cruID), int(rdhl.linkID), int(rdhl.endPointID), nSPages, nCRUPages,
       nTimeFrames, nHBFrames, int(blocks.size()), nErrors);
  if (verbose) {
    for (int i = 0; i < int(blocks.size()); i++) {
      std::stringstream counts;
      counts << '#' << std::setw(5) << i << " | ";
      blocks[i].print(counts.str());
    }
  }
}

//____________________________________________
size_t RawFileReader::LinkData::getNextHBFSize() const
{
  // estimate the memory size of the next HBF to read
  // The blocks are guaranteed to not cover more than 1 HB
  size_t sz = 0;
  int ibl = nextBlock2Read, nbl = blocks.size();
  while(ibl<nbl && (blocks[ibl].orbit == blocks[nextBlock2Read].orbit)) {
    sz += blocks[ibl].size;
    ibl++;
  }
  return sz;
}

//____________________________________________
size_t RawFileReader::LinkData::readNextHBF(char* buff)
{
  // read data of the next complete HB
  size_t sz = 0;
  int ibl = nextBlock2Read, nbl = blocks.size();
  bool error = false;
  while(ibl<nbl) {
    const auto& blc = blocks[ibl++];
    if (blc.orbit != blocks[nextBlock2Read].orbit) {
      break;
    }
    auto fl = reader->mFiles[blc.fileID];
    if (fseek(fl, blc.offset, SEED_SET) || fread(fuff, 1, blc.size, fl)!=blc.size) {
      LOGF(ERROR,"Failed to read for the link with subspec %x a bloc:", subspec);
      blc.print();
      error = true;
    }
    sz += blc.size;
  }
  nextBlock2Read = ibl;
  return error ? 0 : sz; // in case of the error we ignore the data
}

//____________________________________________
size_t RawFileReader::LinkData::getNextTFSize() const
{
  // estimate the memory size of the next TF to read
  // (assuming nextBlock2Read is at the start of the TF)
  size_t sz = 0;
  int ibl = nextBlock2Read, nbl = blocks.size();  
  while(ibl<nbl && (blocks[ibl].tfID==blocks[nextBlock2Read].tfID)) {
    sz += blocks[ibl].size;
    ibl++;
  }
  return sz;
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
bool RawFileReader::LinkData::preprocessCRUPage(const RDH& rdh, bool newSPage)
{
  // account RDH in statistics
  bool ok = true;
  bool newTF = false, newHB = false;

  if (rdh.feeId != rdhl.feeId) { // make sure links with different FEEID were not assigned same subspec
    LOGF(ERROR, "Same SubSpec is found for Links with different RDH.feeId");
    LOGF(ERROR, "old RDH assigned SubSpec=0x%-8d:", subspec);
    o2::raw::HBFUtils::dumpRDH(rdhl);
    LOGF(ERROR, "new RDH assigned SubSpec=0x%-8d:", subspec);
    o2::raw::HBFUtils::dumpRDH(rdh);
    LOG(FATAL) << "Critical error, aborting";
    ok = false;
    nErrors++;
  }

  if (rdh.pageCnt == 0) {
    newTF = (rdh.triggerType & o2::trigger::TF);
    newHB = (rdh.triggerType & (o2::trigger::ORBIT | o2::trigger::HB)) == (o2::trigger::ORBIT | o2::trigger::HB);
  } else if (reader->mCheckErrors & (0x1 << ErrWrongPageCounterIncrement)) {
    // check increasing pageCnt
    if (nCRUPages && (rdh.pageCnt != ((rdhl.pageCnt + 1) & 0xffff))) { // skip for very 1st page
      LOG(ERROR) << ErrNames[ErrWrongPageCounterIncrement]
                 << " old=" << int(rdh.pageCnt) << " new=" << int(rdhl.pageCnt);
      ok = false;
      nErrors++;
    }
  }

  if (reader->mCheckErrors) {
    if (nCRUPages) {
      // check increasing (or wrapping) packetCounter
      if ((rdh.packetCounter != ((rdhl.packetCounter + 1) & 0xff)) &&
          (reader->mCheckErrors & (0x1 << ErrWrongPacketCounterIncrement))) { // skip for very 1st page
        LOG(ERROR) << ErrNames[ErrWrongPacketCounterIncrement]
                   << " new=" << int(rdh.packetCounter) << " old=" << int(rdhl.packetCounter);
        ok = false;
        nErrors++;
      }
      // check if number of HBFs in the TF is as expected
      if (newTF) {
        if (nHBFinTF != reader->mNominalHBFperTF &&
            (reader->mCheckErrors & (0x1 << ErrWrongHBFsPerTF))) {
          LOG(ERROR) << ErrNames[ErrWrongHBFsPerTF] << ": "
                     << nHBFinTF << " instead of " << reader->mNominalHBFperTF;
          ok = false;
          nErrors++;
        }
        nHBFinTF = 0; // reset
      }

    } else { // make sure data starts with TF and HBF
      if ((!newTF || !newHB || rdh.pageCnt != 0) &&
          (reader->mCheckErrors & (0x1 << ErrWrongFirstPage))) {
        LOG(ERROR) << ErrNames[ErrWrongFirstPage];
        ok = false;
        nErrors++;
      }
    }
  }

  if (newHB) {
    if (reader->mCheckErrors) {
      nHBFinTF++;
      if (rdh.stop && (reader->mCheckErrors & (0x1 << ErrHBFStopOnFirstPage))) {
        LOG(ERROR) << ErrNames[ErrHBFStopOnFirstPage] << " @ HBF#" << nHBFrames;
        ok = false;
        nErrors++;
      }
      if (openHB && (reader->mCheckErrors & (0x1 << ErrHBFNoStop))) {
        LOG(ERROR) << ErrNames[ErrHBFNoStop] << " @ HBF#" << nHBFrames;
        ok = false;
        nErrors++;
      }
      if ((reader->mCheckErrors & (0x1 << ErrHBFJump)) &&
          (nCRUPages && // skip this check for the very 1st RDH
           !(o2::raw::HBFUtils::getHBBC(rdh) == o2::raw::HBFUtils::getHBBC(rdhl) &&
             o2::raw::HBFUtils::getHBOrbit(rdh) == o2::raw::HBFUtils::getHBOrbit(rdhl) + 1))) {
        LOG(ERROR) << ErrNames[ErrHBFJump] << " @ HBF#" << nHBFrames << " New HB orbit/bc="
                   << o2::raw::HBFUtils::getHBOrbit(rdh) << '/' << int(o2::raw::HBFUtils::getHBBC(rdh))
                   << " is not incremented by 1 orbit wrt Old HB orbit/bc="
                   << o2::raw::HBFUtils::getHBOrbit(rdhl) << '/' << int(o2::raw::HBFUtils::getHBBC(rdhl));
        ok = false;
        nErrors++;
      }
    } // end of check errors
    openHB = true;
    nHBFrames++;
  }
  if (rdh.stop) {
    openHB = false;
  }

  if (newTF || newSPage || newHB) {
    auto& bl = blocks.emplace_back(reader->mCurrentFileID, reader->mPosInFile);
    if (newTF) {      
      nTimeFrames++;
      bl.setFlag(LinkBlock::StartTF);
      if (reader->mCheckErrors & ErrNoSuperPageForTF) {
        if (reader->mMultiLinkFile && !newSPage) {
          LOG(ERROR) << ErrNames[ErrNoSuperPageForTF] << " @ TF#" << nTimeFrames;
          ok = false;
          nErrors++;
        }
      } // end of check errors
    }
    bl.orbit = o2::raw::HBFUtils::getHBOrbit(rdh);
    bl.tfID = nTimeFrames-1;
    
    if (newSPage) {
      nSPages++;
      bl.setFlag(LinkBlock::StartSP);
    }
    if (newHB) {
      bl.setFlag(LinkBlock::StartHB);
    }
  }
  blocks.back().setFlag(LinkBlock::EndHB, rdh.stop);  // last processed RDH defines this flag
  blocks.back().size += rdh.offsetToNext; 
  rdhl = rdh;
  nCRUPages++;
  if (!ok) {
    LOG(ERROR) << " ^^^Problem(s) was encountered at offset " << reader->mPosInFile << " of file " << reader->mCurrentFileID;
    o2::raw::HBFUtils::printRDH(rdh);
  } else if (reader->mVerbosity > 1) {
    if (reader->mVerbosity > 2) {
      o2::raw::HBFUtils::dumpRDH(rdh);
    } else {
      o2::raw::HBFUtils::printRDH(rdh);
    }
  }
  return true;
}

//====================== methods of RawFileReader ========================

//_____________________________________________________________________

//_____________________________________________________________________
bool RawFileReader::checkRDH(const o2::header::RAWDataHeaderV4& rdh) const
{
  // check if rdh conforms with RDH4 fields
  bool ok = true;
  if (rdh.version != 4) {
    LOG(ERROR) << "RDH version 4 is expected instead of " << int(rdh.version);
    ok = false;
  }
  if (rdh.headerSize != 64) {
    LOG(ERROR) << "RDH with header size of 64 B is expected instead of " << int(rdh.headerSize);
    ok = false;
  }
  if (rdh.memorySize < 64 || rdh.offsetToNext < 64) {
    LOG(ERROR) << "RDH expected to have memory size/offset to next >= 64 B instead of "
               << int(rdh.memorySize) << '/' << int(rdh.offsetToNext);
    ok = false;
  }
  if (rdh.zero0 || rdh.word3 || rdh.zero41 || rdh.zero42 || rdh.word5 || rdh.zero6 || rdh.word7) {
    LOG(ERROR) << "Some reserved fields of RDH v4 are not empty";
    ok = false;
  }
  if (!ok) {
    o2::raw::HBFUtils::dumpRDH(rdh);
  }
  return ok;
}

//_____________________________________________________________________
bool RawFileReader::checkRDH(const o2::header::RAWDataHeaderV5& rdh) const
{
  // check if rdh conforms with RDH5 fields
  bool ok = true;
  if (rdh.version != 5) {
    LOG(ERROR) << "RDH version 5 is expected instead of " << int(rdh.version);
    ok = false;
  }
  if (rdh.headerSize != 64) {
    LOG(ERROR) << "RDH with header size of 64 B is expected instead of " << int(rdh.headerSize);
    ok = false;
  }
  if (rdh.memorySize < 64 || rdh.offsetToNext < 64) {
    LOG(ERROR) << "RDH expected to have memory size and offset to next >= 64 B instead of "
               << int(rdh.memorySize) << '/' << int(rdh.offsetToNext);
    ok = false;
  }
  if (rdh.zero0 || rdh.word3 || rdh.zero4 || rdh.word5 || rdh.zero6 || rdh.word7) {
    LOG(ERROR) << "Some reserved fields of RDH v5 are not empty";
    ok = false;
  }
  if (!ok) {
    o2::raw::HBFUtils::dumpRDH(rdh);
  }
  return ok;
}

//_____________________________________________________________________
int RawFileReader::getLinkLocalID(const RDH& rdh)
{
  // get id of the link subspec. in the parser (create entry if new)
  LinkSpec_t subspec = getSubSpec(rdh.cruID, rdh.linkID, rdh.endPointID);
  auto entryMap = mLinkEntries.find(subspec);
  if (entryMap == mLinkEntries.end()) { // need to register a new link
    int n = mLinkEntries.size();
    mLinkEntries[subspec] = n;
    auto& lnk = mLinksData.emplace_back(rdh, this);
    lnk.subspec = subspec;
    return n;
  }
  return entryMap->second;
}

//_____________________________________________________________________
bool RawFileReader::preprocessFile(int ifl)
{
  // preprocess file, check RDH data, build statistics
  FILE* fl = mFiles[ifl];
  mCurrentFileID = ifl;
  RDH rdh;

  LinkSpec_t subspecPrev = 0xffffffff;
  int lIDPrev = -1;
  mMultiLinkFile = false;
  rewind(fl);
  long int nr = 0;
  mPosInFile = 0;
  int nRDHread = 0;
  bool ok = true;
  int readBytes = sizeof(RDH);
  while ((nr = fread(&rdh, 1, readBytes, fl))) {
    if (nr < readBytes) {
      LOG(ERROR) << "EOF was unexpected, only " << nr << " bytes were read for RDH";
      ok = false;
      break;
    }
    if (!(ok = checkRDH(rdh))) {
      break;
    }
    nRDHread++;
    LinkSpec_t subspec = getSubSpec(rdh.cruID, rdh.linkID, rdh.endPointID);
    int lID = lIDPrev;
    if (subspec != subspecPrev) { // link has changed
      subspecPrev = subspec;
      if (lIDPrev != -1) {
        mMultiLinkFile = true;
      }
      lID = getLinkLocalID(rdh);
    }
    bool newSPage = lID != lIDPrev;
    mLinksData[lID].preprocessCRUPage(rdh, newSPage);
    //
    mPosInFile += rdh.offsetToNext;
    if (fseek(fl, mPosInFile, SEEK_SET)) {
      break;
    }
    lIDPrev = lID;
  }
  mPosInFile = ftell(fl);
  if (lIDPrev != -1) { // close last block
    auto& lastBlock = mLinksData[lIDPrev].blocks.back();
    lastBlock.size = mPosInFile - lastBlock.offset;
  }

  LOGF(INFO, "File %3d : %9li bytes scanned, %6d RDH read for %4d links from %s",
       mCurrentFileID, mPosInFile, nRDHread, int(mLinkEntries.size()), mFileNames[mCurrentFileID]);
  return ok;
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
bool RawFileReader::addFile(const std::string& sname)
{
  if (mInitDone) {
    LOG(ERROR) << "Cannot add new files after initialization";
    return false;
  }
  auto inFile = fopen(sname.c_str(), "rb");
  if (!inFile) {
    LOG(ERROR) << "Failed to open input file " << sname;
    return false;
  }
  mFileNames.push_back(sname);
  mFiles.push_back(inFile);
  return true;
}

//_____________________________________________________________________
bool RawFileReader::init()
{
  // make initialization, preprocess files and chack for errors if asked
  
  for (int i = 0; i < NErrorsDefined; i++) {
    if (mCheckErrors & (0x1 << i))
      LOGF(INFO, "perform check for /%s/", ErrNames[i].data());
  }

  int nf = mFiles.size();
  bool ok = true;
  for (int i = 0; i < nf; i++) {
    ok &= preprocessFile(i);
  }
  mOrderedIDs.resize(mLinksData.size());
  for (int i = mLinksData.size(); i--;) {
    mOrderedIDs[i] = i;
  }
  std::sort(mOrderedIDs.begin(), mOrderedIDs.end(),
            [& links = mLinksData](int a, int b) { return links[a].subspec < links[b].subspec; });

  size_t maxSP = 0, maxTF = 0;
  
  LOGF(INFO, "Summary of preprocessing:");
  for (int i = 0; i < int(mLinksData.size()); i++) {
    const auto& link = getLink(i);
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
      LOGF(WARNING, "       Attention: largest superpage %zu B exceeds expected %d B",
           msp, mNominalSPageSize);
    }
    // min max orbits
    if (link.blocks.front().orbit < mOrbitMin) {
      mOrbitMin = link.blocks.front().orbit;      
    }
    if (link.blocks.back().orbit > mOrbitMax) {
      mOrbitMax = link.blocks.back().orbit;      
    }    
  }
  LOGF(INFO, "First orbit: %d, Last orbit: %d",mOrbitMin, mOrbitMax);
  LOGF(INFO, "Largest super-page: %zu B, largest TF: %zu B", maxSP, maxTF);
  if (!mCheckErrors) {
    LOGF(INFO, "Detailed data format check was disabled");
  }
  mInitDone = true;

  return ok;
}

//_____________________________________________________________________
bool RawFileReader::readNextTF()
{
  size_t sz = 0;
  for (int i = 0; i < int(mLinksData.size()); i++) {
    const auto& link = getLink(i);
    sz += link.getNextTFSize(); // size of TF and number of HBFs (don't trust to expected value?)
  }
  // create a buffer for new TF
  std::unique<char[]> buff = std::make_unique<char[]>(sz);
  char* buffPtr = buff.get();
  
  for (int i = 0; i < int(mLinksData.size()); i++) {
    auto& link = getLink(i);
    if (link.nextBlock2Read<int(link.blocks.size()) && link.blocks[link.nextBlock2Read].tfID == mNextTF2Read) {
      auto szHBF = link->readNextHBF(buffPtr);
      //
      // RSTODO: create a DataHeader, add to multipart
      //
    }
  }
  mNextTF2Read++;
  return true;
}
