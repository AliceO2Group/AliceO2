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
#include "DetectorsRaw/RawFileReader.h"
#include "CommonConstants/Triggers.h"
#include "DetectorsRaw/HBFUtils.h"

using namespace o2::raw;

//====================== methods of LinkBlock ========================
//____________________________________________
void RawFileReader::LinkBlock::print() const
{
  printf("file:%3d offs:%10zu size:%8d newSP:%d newTF:%d newHB:%d\n", fileID, offset, size, startSP, startTF, startHB);
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
void RawFileReader::LinkData::print(bool verbose) const
{
  printf("SSpec:0x%-8d FEE:0x%4x CRU:%4d Lnk:%3d EP:%d | SPages:%4d Pages:%6d TFs:%6d with %6d HBF in %4d blocks",
         subspec, int(rdhl.feeId), int(rdhl.cruID), int(rdhl.linkID), int(rdhl.endPointID), nSPages, nCRUPages,
         nTimeFrames, nHBFrames, int(blocks.size()));
  if (reader && reader->mCheckErrors) {
    printf(" (%d err)", nErrors);
  }
  printf("\n");
  if (verbose) {
    for (int i = 0; i < int(blocks.size()); i++) {
      printf("#%5d | ", i);
      blocks[i].print();
    }
  }
}

//____________________________________________
size_t RawFileReader::LinkData::getLargestSuperPage() const
{
  // estimate largest super page size
  size_t szMax = 0, szLast = 0;
  for (const auto& bl : blocks) {
    if (bl.startSP) { // account previous SPage and start accumulation of the next one
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
    if (bl.startTF) { // account previous TF and start accumulation of the next one
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
    LOG(ERROR) << "Same SubSpec is found for Links with different RDH.feeId";
    printf("old RDH assigned SubSpec=0x%-8d:\n", subspec);
    o2::raw::HBFUtils::dumpRDH(rdhl);
    printf("new RDH assigned SubSpec=0x%-8d:\n", subspec);
    o2::raw::HBFUtils::dumpRDH(rdh);
    LOG(FATAL) << "Critical error, aborting";
    ok = false;
    nErrors++;
  }

  if (rdh.pageCnt == 0) {
    newTF = (rdh.triggerType & o2::trigger::TF);
    newHB = (rdh.triggerType & (o2::trigger::ORBIT | o2::trigger::HB)) == (o2::trigger::ORBIT | o2::trigger::HB);
  } else if (reader->mCheckErrors) {
    // check increasing pageCnt
    if (nCRUPages && (rdh.pageCnt != ((rdhl.pageCnt + 1) & 0xffff))) { // skip for very 1st page
      LOG(ERROR) << "RDH.pageCnt=" << int(rdh.pageCnt) << " is not +1 wrt previous RDH.pageCnt=" << int(rdhl.pageCnt);
      ok = false;
      nErrors++;
    }
  }

  if (reader->mCheckErrors) {
    if (nCRUPages) {
      // check increasing (or wrapping) packetCounter
      if ((rdh.packetCounter != ((rdhl.packetCounter + 1) & 0xff))) { // skip for very 1st page
        LOG(ERROR) << "RDH.packetCounter=" << int(rdh.packetCounter)
                   << " is not + 1 wrt previous RDH.packetCounter=" << int(rdhl.packetCounter);
        ok = false;
        nErrors++;
      }
      // check if number of HBFs in the TF is as expected
      if (newTF) {
        if (nHBFinTF != reader->mNominalHBFperTF) {
          LOG(ERROR) << "Number of HBFs in TF " << nHBFinTF << " differs from expected " << reader->mNominalHBFperTF;
          ok = false;
          nErrors++;
        }
        nHBFinTF = 0; // reset
      }

    } else { // make sure data starts with TF and HBF
      if (!newTF || !newHB || rdh.pageCnt != 0) {
        LOG(ERROR) << "Very 1st RDH of the data does not start with TF or new HBF";
        ok = false;
        nErrors++;
      }
    }
  }

  if (newHB) {
    if (reader->mCheckErrors) {
      nHBFinTF++;
      if (rdh.stop) {
        LOG(ERROR) << "Stop of HBF #" << nHBFrames << " is found at 1st page";
        ok = false;
        nErrors++;
      }
      if (openHB) {
        LOG(ERROR) << "New HBF#" << nHBFrames << " starts while previous HBF was not closed";
        ok = false;
        nErrors++;
      }
      if (nCRUPages && !checkIRIncrement(rdh, rdhl)) { // skip check for the very 1st RDH
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

  if (newTF || newSPage) {
    auto& bl = blocks.emplace_back(reader->mCurrentFileID, reader->mPosInFile);
    if (newTF) {
      nTimeFrames++;
      bl.startTF = true;
      if (reader->mCheckErrors) {
        if (reader->mMultiLinkFile && !newSPage) {
          LOG(ERROR) << "New TF#" << nTimeFrames << " does not start new from new superpage";
          ok = false;
          nErrors++;
        }
        /* // this test seems to be meaningless
        if (!newHB) {
          LOG(ERROR) << "New TF#" << nTimeFrames << " does not start with new HBF";
          ok = false;
          nErrors++;
        }
	*/
      } // end of check errors
    }
    if (newSPage) {
      nSPages++;
      bl.startSP = true;
      bl.startHB = true;
    }
  }
  rdhl = rdh;
  nCRUPages++;
  if (!ok) {
    LOG(ERROR) << " ^^^Problem(s) was encountered at offset " << reader->mPosInFile << " of file " << reader->mCurrentFileID;
    if (reader->mVerbosity > 0) {
      o2::raw::HBFUtils::printRDH(rdh);
    }
  } else {
    if (reader->mVerbosity > 1) {
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
    if (newSPage && lIDPrev != -1) { // flag the end of the block of previosly processed link
      auto& prevBlock = mLinksData[lIDPrev].blocks.back();
      prevBlock.size = mPosInFile - prevBlock.offset;
    }
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

  LOG(INFO) << "#" << mCurrentFileID << ": " << mPosInFile << " bytes scanned, " << nRDHread << " RDH read for "
            << mLinkEntries.size() << " links from " << mFileNames[mCurrentFileID];
  return ok;
}

//_____________________________________________________________________
void RawFileReader::printStat(bool verbose) const
{
  int n = getNLinks();
  for (int i = 0; i < n; i++) {
    const auto& link = getLink(i);
    printf("#%-4d| ", i);
    link.print(verbose);
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
  printf("\nSummary of preprocessing:\n");
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
    printf("L%-4d| ", i);
    link.print(mVerbosity);
    if (msp > mNominalSPageSize) {
      printf("       Attention: largest superpage %zu B exceeds expected %d B\n",
             msp, mNominalSPageSize);
    }
  }
  printf("Largest super-page: %zu B, largest TF: %zu B\n", maxSP, maxTF);
  if (!mCheckErrors) {
    printf("Detailed data format check was disabled\n");
  }
  mInitDone = true;

  return ok;
}
