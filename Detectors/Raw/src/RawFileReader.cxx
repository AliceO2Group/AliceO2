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
  LOGF(INFO, "%sfile:%3d offs:%10zu size:%8d newSP:%d newTF:%d newHB:%d endHB:%d | Orbit %u TF %u",
       pref, fileID, offset, size, testFlag(StartSP), testFlag(StartTF), testFlag(StartHB),
       testFlag(EndHB), orbit, tfID);
}

//====================== methods of LinkData ========================

//____________________________________________
std::string RawFileReader::LinkData::describe() const
{
  std::stringstream ss;
  ss << "Link " << origin.as<std::string>() << '/' << description.as<std::string>() << "/0x"
     << std::hex << std::setw(8) << std::setfill('0') << subspec;
  return ss.str();
}

//____________________________________________
void RawFileReader::LinkData::print(bool verbose, const std::string& pref) const
{
  LOGF(INFO, "%s %s FEE:0x%04x CRU:%4d Lnk:%3d EP:%d RDHv%d Src:%s | SPages:%4d Pages:%6d TFs:%6d with %6d HBF in %4d blocks (%d err)",
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
size_t RawFileReader::LinkData::getNextHBFSize() const
{
  // estimate the memory size of the next HBF to read
  // The blocks are guaranteed to not cover more than 1 HB
  size_t sz = 0;
  int ibl = nextBlock2Read, nbl = blocks.size();
  while (ibl < nbl && (blocks[ibl].orbit == blocks[nextBlock2Read].orbit)) {
    sz += blocks[ibl].size;
    ibl++;
  }
  return sz;
}

//____________________________________________
size_t RawFileReader::LinkData::readNextHBF(char* buff)
{
  // read data of the next complete HB, buffer of getNextHBFSize() must be allocated in advance
  size_t sz = 0;
  int ibl = nextBlock2Read, nbl = blocks.size();
  bool error = false;
  while (ibl < nbl) {
    const auto& blc = blocks[ibl];
    if (blc.orbit != blocks[nextBlock2Read].orbit) {
      break;
    }
    ibl++;
    auto fl = reader->mFiles[blc.fileID];
    if (fseek(fl, blc.offset, SEEK_SET) || fread(buff + sz, 1, blc.size, fl) != blc.size) {
      LOGF(ERROR, "Failed to read for the %s a bloc:", describe());
      blc.print();
      error = true;
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
  int ibl = nextBlock2Read, nbl = blocks.size();
  while (ibl < nbl) {
    const auto& blc = blocks[ibl];
    if (blc.orbit != blocks[nextBlock2Read].orbit) {
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
  int ibl = nextBlock2Read, nbl = blocks.size();
  while (ibl < nbl && (blocks[ibl].tfID == blocks[nextBlock2Read].tfID)) {
    sz += blocks[ibl].size;
    ibl++;
  }
  return sz;
}

//_____________________________________________________________________
size_t RawFileReader::LinkData::readNextTF(char* buff)
{
  // read next complete TF, buffer of getNextTFSize() must be allocated in advance
  size_t sz = 0;
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
void RawFileReader::LinkData::rewindToTF(uint32_t tf)
{
  // go to given TF
  nextBlock2Read = 0;
  for (uint32_t i = 0; i < tf; i++) {
    skipNextTF();
  }
}

//____________________________________________
int RawFileReader::LinkData::getNHBFinTF() const
{
  // estimate number of HBFs left in the TF
  int ibl = nextBlock2Read, nbl = blocks.size(), nHB = 0;
  while (ibl < nbl && (blocks[ibl].tfID == blocks[nextBlock2Read].tfID)) {
    if (blocks[ibl].testFlag(LinkBlock::StartHB)) {
      nHB++;
    }
    ibl++;
  }
  return nHB;
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

  if (RDHUtils::getFEEID(rdh) != RDHUtils::getFEEID(rdhl)) { // make sure links with different FEEID were not assigned same subspec
    LOGF(ERROR, "Same SubSpec is found for %s with different RDH.feeId", describe());
    LOGF(ERROR, "old RDH assigned SubSpec=0x%-8d:", subspec);
    RDHUtils::dumpRDH(rdhl);
    LOGF(ERROR, "new RDH assigned SubSpec=0x%-8d:", subspec);
    RDHUtils::dumpRDH(rdh);
    throw std::runtime_error("Conflicting SubSpecs are provided");
    ok = false;
    nErrors++;
  }

  auto pageCnt = RDHUtils::getPageCounter(rdh);
  if (pageCnt == 0) {
    auto triggerType = RDHUtils::getTriggerType(rdh);
    newTF = (triggerType & o2::trigger::TF);
    newHB = (triggerType & (o2::trigger::ORBIT | o2::trigger::HB)) == (o2::trigger::ORBIT | o2::trigger::HB);
  } else if (reader->mCheckErrors & (0x1 << ErrWrongPageCounterIncrement)) {
    // check increasing pageCnt
    if (nCRUPages && (pageCnt != (RDHUtils::getPageCounter(rdhl) + 1))) { // skip for very 1st page
      LOG(ERROR) << ErrNames[ErrWrongPageCounterIncrement]
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
        LOG(ERROR) << ErrNames[ErrWrongPacketCounterIncrement]
                   << " new=" << int(packetCounter) << " old=" << int(packetCounterL);
        ok = false;
        nErrors++;
      }
      // check if number of HBFs in the TF is as expected
      if (newTF) {
        if (nHBFinTF != HBFUtils::Instance().getNOrbitsPerTF() &&
            (reader->mCheckErrors & (0x1 << ErrWrongHBFsPerTF))) {
          LOG(ERROR) << ErrNames[ErrWrongHBFsPerTF] << ": "
                     << nHBFinTF << " instead of " << HBFUtils::Instance().getNOrbitsPerTF();
          ok = false;
          nErrors++;
        }
        nHBFinTF = 0; // reset
      }

    } else { // make sure data starts with TF and HBF
      if ((!newTF || !newHB || pageCnt != 0) &&
          (reader->mCheckErrors & (0x1 << ErrWrongFirstPage))) {
        LOG(ERROR) << ErrNames[ErrWrongFirstPage];
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
           !(hbIR.bc == hblIR.bc && hbIR.orbit == hblIR.orbit + 1))) {
        LOG(ERROR) << ErrNames[ErrHBFJump] << " @ HBF#" << nHBFrames << " New HB orbit/bc=" << hbIR.orbit << '/' << int(hbIR.bc)
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

  if (newTF || newSPage || newHB) {
    auto& bl = blocks.emplace_back(reader->mCurrentFileID, reader->mPosInFile);
    if (newTF) {
      nTimeFrames++;
      bl.setFlag(LinkBlock::StartTF);
      if (reader->mCheckErrors & (0x1 << ErrNoSuperPageForTF)) {
        if (reader->mMultiLinkFile && !newSPage) {
          LOG(ERROR) << ErrNames[ErrNoSuperPageForTF] << " @ TF#" << nTimeFrames;
          ok = false;
          nErrors++;
        }
      } // end of check errors
    }
    bl.orbit = hbIR.orbit;
    bl.tfID = nTimeFrames - 1;

    if (newSPage) {
      nSPages++;
      bl.setFlag(LinkBlock::StartSP);
    }
    if (newHB) {
      bl.setFlag(LinkBlock::StartHB);
    }
  }
  blocks.back().setFlag(LinkBlock::EndHB, stop); // last processed RDH defines this flag
  blocks.back().size += RDHUtils::getOffsetToNext(rdh);
  rdhl = rdh;
  nCRUPages++;
  if (!ok) {
    LOG(ERROR) << " ^^^Problem(s) was encountered at offset " << reader->mPosInFile << " of file " << reader->mCurrentFileID;
    RDHUtils::printRDH(rdh);
  } else if (reader->mVerbosity > 1) {
    if (reader->mVerbosity > 2) {
      RDHUtils::dumpRDH(rdh);
    } else {
      RDHUtils::printRDH(rdh);
    }
  }
  return true;
}

//====================== methods of RawFileReader ========================

//_____________________________________________________________________
RawFileReader::RawFileReader(const std::string& config, int verbosity) : mVerbosity(verbosity)
{
  if (!config.empty()) {
    auto inp = parseInput(config);
    loadFromInputsMap(inp);
  }
}

//_____________________________________________________________________
int RawFileReader::getLinkLocalID(const RDHAny& rdh, const o2::header::DataOrigin orig)
{
  // get id of the link subspec. in the parser (create entry if new)
  LinkSubSpec_t subspec = RDHUtils::getSubSpec(rdh);
  LinkSpec_t spec = createSpec(orig, subspec);
  auto entryMap = mLinkEntries.find(spec);
  if (entryMap == mLinkEntries.end()) { // need to register a new link
    int n = mLinkEntries.size();
    mLinkEntries[spec] = n;
    auto& lnk = mLinksData.emplace_back(rdh, this);
    lnk.subspec = subspec;
    lnk.origin = mDataSpecs[mCurrentFileID].first;
    lnk.description = mDataSpecs[mCurrentFileID].second;
    lnk.spec = spec;
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
  rewind(fl);
  long int nr = 0;
  mPosInFile = 0;
  size_t nRDHread = 0, boffs;
  bool ok = true, readMore = true;
  while (readMore && (nr = fread(buffer.get(), 1, mBufferSize, fl))) {
    boffs = 0;
    while (1) {
      auto& rdh = *reinterpret_cast<RDHUtils::RDHAny*>(&buffer[boffs]);
      nRDHread++;
      LinkSpec_t spec = createSpec(mDataSpecs[mCurrentFileID].first, RDHUtils::getSubSpec(rdh));
      int lID = lIDPrev;
      if (spec != specPrev) { // link has changed
        specPrev = spec;
        if (lIDPrev != -1) {
          mMultiLinkFile = true;
        }
        lID = getLinkLocalID(rdh, mDataSpecs[mCurrentFileID].first);
      }
      bool newSPage = lID != lIDPrev;
      mLinksData[lID].preprocessCRUPage(rdh, newSPage);
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
bool RawFileReader::addFile(const std::string& sname, o2::header::DataOrigin origin, o2::header::DataDescription desc)
{
  if (mInitDone) {
    LOG(ERROR) << "Cannot add new files after initialization";
    return false;
  }
  auto inFile = fopen(sname.c_str(), "rb");
  bool ok = true;
  if (!inFile) {
    LOG(ERROR) << "Failed to open input file " << sname;
    ok = false;
  }
  if (origin == o2h::gDataOriginInvalid) {
    LOG(ERROR) << "Invalid data origin " << origin.as<std::string>() << " for file " << sname;
    ok = false;
  }
  if (desc == o2h::gDataDescriptionInvalid) {
    LOG(ERROR) << "Invalid data description " << desc.as<std::string>() << " for file " << sname;
    ok = false;
  }
  if (!ok) {
    fclose(inFile);
  }

  mFileNames.push_back(sname);
  mFiles.push_back(inFile);
  mDataSpecs.emplace_back(origin, desc);
  return true;
}

//_____________________________________________________________________
bool RawFileReader::init()
{
  // make initialization, preprocess files and chack for errors if asked

  for (int i = 0; i < NErrorsDefined; i++) {
    if (mCheckErrors & (0x1 << i))
      LOGF(INFO, "%s check for /%s/", (mCheckErrors & (0x1 << i)) ? "perform" : "ignore ", ErrNames[i].data());
  }
  if (mMaxTFToRead < 0xffffffff) {
    LOGF(INFO, "at most %u TF will be processed", mMaxTFToRead);
  }

  int nf = mFiles.size();
  bool ok = true;
  for (int i = 0; i < nf; i++) {
    ok &= preprocessFile(i);
  }
  mOrderedIDs.resize(mLinksData.size());
  for (int i = mLinksData.size(); i--;) {
    mOrderedIDs[i] = i;
    if (mNTimeFrames < mLinksData[i].nTimeFrames) {
      mNTimeFrames = mLinksData[i].nTimeFrames;
    }
  }
  std::sort(mOrderedIDs.begin(), mOrderedIDs.end(),
            [& links = mLinksData](int a, int b) { return links[a].spec < links[b].spec; });

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
      LOGF(DEBUG, "       Attention: largest superpage %zu B exceeds expected %d B",
           msp, mNominalSPageSize);
    }
    // min max orbits
    if (link.blocks.front().orbit < mOrbitMin) {
      mOrbitMin = link.blocks.front().orbit;
    }
    if (link.blocks.back().orbit > mOrbitMax) {
      mOrbitMax = link.blocks.back().orbit;
    }
    if ((mCheckErrors & (0x1 << ErrWrongNumberOfTF)) && (mNTimeFrames != link.nTimeFrames)) {
      LOGF(ERROR, "%s for %s: %u TFs while %u were seen for other links", ErrNames[ErrWrongNumberOfTF],
           link.describe(), link.nTimeFrames, mNTimeFrames);
    }
  }
  LOGF(INFO, "First orbit: %d, Last orbit: %d", mOrbitMin, mOrbitMax);
  LOGF(INFO, "Largest super-page: %zu B, largest TF: %zu B", maxSP, maxTF);
  if (!mCheckErrors) {
    LOGF(INFO, "Detailed data format check was disabled");
  }
  mInitDone = true;

  return ok;
}

//_____________________________________________________________________
o2h::DataOrigin RawFileReader::getDataOrigin(const std::string& ors)
{
  constexpr int NGoodOrigins = 18;
  constexpr std::array<o2h::DataOrigin, NGoodOrigins> goodOrigins{
    o2h::gDataOriginFLP, o2h::gDataOriginACO, o2h::gDataOriginCPV, o2h::gDataOriginCTP, o2h::gDataOriginEMC,
    o2h::gDataOriginFT0, o2h::gDataOriginFV0, o2h::gDataOriginFDD, o2h::gDataOriginHMP, o2h::gDataOriginITS,
    o2h::gDataOriginMCH, o2h::gDataOriginMFT, o2h::gDataOriginMID, o2h::gDataOriginPHS, o2h::gDataOriginTOF,
    o2h::gDataOriginTPC, o2h::gDataOriginTRD, o2h::gDataOriginZDC};
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
      setDefaultDataOrigin(ordesc.first);
      setDefaultDataDescription(ordesc.second);
      continue;
    }
    for (const auto& fnm : files) { // specific file names
      addFile(fnm, ordesc.first, ordesc.second);
    }
  }
}

//_____________________________________________________________________
RawFileReader::InputsMap RawFileReader::parseInput(const std::string& confUri)
{
  // read input files from configuration
  std::map<std::pair<o2h::DataOrigin, o2h::DataDescription>, std::vector<std::string>> entries;

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
  try {
    std::string origStr, descStr, defstr = "defaults";
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
    entries[{defDataOrigin, defDataDescription}]; // insert
    LOG(DEBUG) << "Setting default dataOrigin/Description " << defDataOrigin.as<std::string>() << '/' << defDataDescription.as<std::string>();

    for (auto flsect : ConfigFileBrowser(&cfg, "input-")) {
      std::string flNameStr, defs{""};
      cfg.getOptionalValue<std::string>(flsect + ".dataOrigin", origStr, defDataOrigin.as<std::string>());
      cfg.getOptionalValue<std::string>(flsect + ".dataDescription", descStr, defDataDescription.as<std::string>());
      cfg.getOptionalValue<std::string>(flsect + ".filePath", flNameStr, defs);
      if (flNameStr.empty()) {
        LOG(DEBUG) << "Skipping incomplete input " << flsect;
        continue;
      }
      auto dataOrigin = getDataOrigin(origStr);
      if (dataOrigin == o2h::gDataOriginInvalid) {
        throw std::runtime_error(std::string("Invalid data origin ") + origStr + " for " + flsect);
      }
      auto dataDescription = getDataDescription(descStr);
      if (dataDescription == o2h::gDataDescriptionInvalid) {
        throw std::runtime_error(std::string("Invalid data description ") + descStr + " for " + flsect);
      }
      entries[{dataOrigin, dataDescription}].push_back(flNameStr);
      LOG(DEBUG) << "adding file " << flNameStr << " to dataOrigin/Description " << dataOrigin.as<std::string>() << '/' << dataDescription.as<std::string>();
    }
  } catch (std::string& e) { // to catch exceptions from the parser
    throw std::runtime_error(std::string("Aborting due to the exception: ") + e);
  }

  return entries;
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
