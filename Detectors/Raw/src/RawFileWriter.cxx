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

/// @file   RawFileWriter.h
/// @author ruben.shahoyan@cern.ch
/// @brief  Utility class to write detectors data to (multiple) raw data file(s) respecting CRU format

#include <iomanip>
#include <iostream>
#include <sstream>
#include <functional>
#include <cassert>
#include "CommonUtils/NameConf.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "DetectorsRaw/HBFUtils.h"
#include "CommonConstants/Triggers.h"
#include "Framework/Logger.h"
#include <filesystem>

using namespace o2::raw;
using IR = o2::InteractionRecord;

//_____________________________________________________________________
RawFileWriter::~RawFileWriter()
{
  close();
}

//_____________________________________________________________________
void RawFileWriter::close()
{
  // finalize all links
  if (mFName2File.empty()) {
    return;
  }
  if (mCachingStage) {
    fillFromCache();
  }
  if (mDoLazinessCheck) {
    IR newIR = mDetLazyCheck.ir;
    mDetLazyCheck.completeLinks(this, ++newIR); // make sure that all links for previously called IR got their addData call
    mDoLazinessCheck = false;
  }

  if (!mFirstIRAdded.isDummy()) { // flushing and completing the last HBF makes sense only if data was added.
    auto irmax = getIRMax();
    // for CRU detectors link.updateIR and hence the irmax points on the last IR with data + 1 orbit
    if (isCRUDetector()) {
      irmax.orbit -= 1;
    }
    for (auto& lnk : mSSpec2Link) {
      lnk.second.close(irmax);
      lnk.second.print();
    }
  }
  //
  // close all files
  for (auto& flh : mFName2File) {
    LOG(info) << "Closing output file " << flh.first;
    fclose(flh.second.handler);
    flh.second.handler = nullptr;
  }
  mFName2File.clear();
  if (mDetLazyCheck.completeCount) {
    LOG(warning) << "RawFileWriter forced " << mDetLazyCheck.completeCount << " dummy addData calls in "
                 << mDetLazyCheck.irSeen << " IRs for links which did not receive data";
  }
  mTimer.Stop();
  mTimer.Print();
}

//_____________________________________________________________________
void RawFileWriter::fillFromCache()
{
  LOG(info) << "Filling links from cached trees";
  mCachingStage = false;
  for (const auto& cache : mCacheMap) {
    for (const auto& entry : cache.second) {
      auto& link = getLinkWithSubSpec(entry.first);
      link.cacheTree->GetEntry(entry.second);
      if (mDoLazinessCheck) {
        mDetLazyCheck.completeLinks(this, cache.first); // make sure that all links for previously called IR got their addData call
        mDetLazyCheck.acknowledge(link.subspec, cache.first, link.cacheBuffer.preformatted, link.cacheBuffer.trigger, link.cacheBuffer.detField);
      }
      link.addData(cache.first, link.cacheBuffer.payload, link.cacheBuffer.preformatted, link.cacheBuffer.trigger, link.cacheBuffer.detField);
    }
  }
  mCacheFile->cd();
  for (auto& linkEntry : mSSpec2Link) {
    if (linkEntry.second.cacheTree) {
      linkEntry.second.cacheTree->Write();
      linkEntry.second.cacheTree.reset(nullptr);
    }
  }
  std::string cacheName{mCacheFile->GetName()};
  mCacheFile->Close();
  mCacheFile.reset(nullptr);
  unlink(cacheName.c_str());
}

//_____________________________________________________________________
RawFileWriter::LinkData::LinkData(const LinkData& src) : rdhCopy(src.rdhCopy), updateIR(src.updateIR), lastRDHoffset(src.lastRDHoffset), startOfRun(src.startOfRun), packetCounter(src.packetCounter), pageCnt(src.pageCnt), subspec(src.subspec), nTFWritten(src.nTFWritten), nRDHWritten(src.nRDHWritten), nBytesWritten(src.nBytesWritten), fileName(src.fileName), buffer(src.buffer), writer(src.writer)
{
}

//_____________________________________________________________________
RawFileWriter::LinkData& RawFileWriter::LinkData::operator=(const LinkData& src)
{
  if (this != &src) {
    rdhCopy = src.rdhCopy;
    updateIR = src.updateIR;
    lastRDHoffset = src.lastRDHoffset;
    startOfRun = src.startOfRun;
    packetCounter = src.packetCounter;
    pageCnt = src.pageCnt;
    subspec = src.subspec;
    nTFWritten = src.nTFWritten;
    nRDHWritten = src.nRDHWritten;
    nBytesWritten = src.nBytesWritten;
    fileName = src.fileName;
    buffer = src.buffer;
    writer = src.writer;
  }
  return *this;
}

//_____________________________________________________________________
RawFileWriter::LinkData& RawFileWriter::registerLink(uint16_t fee, uint16_t cru, uint8_t link, uint8_t endpoint, std::string_view outFileNameV)
{
  // register the GBT link and its output file
  std::string outFileName{outFileNameV};
  auto sspec = RDHUtils::getSubSpec(cru, link, endpoint, fee);
  auto& linkData = mSSpec2Link[sspec];
  auto& file = mFName2File[std::string(outFileName)];
  if (!file.handler) {
    if (!(file.handler = fopen(outFileName.c_str(), "wb"))) { // if file does not exist, create it
      LOG(error) << "Failed to open output file " << outFileName;
      throw std::runtime_error(std::string("cannot open link output file ") + outFileName);
    }
  }
  if (!linkData.fileName.empty()) { // this link was already declared and associated with a file
    if (linkData.fileName == outFileName) {
      LOGF(info, "Link 0x%ux was already declared with same output, do nothing", sspec);
      return linkData;
    } else {
      LOGF(error, "Link 0x%ux was already connected to different output file %s", sspec, linkData.fileName);
      throw std::runtime_error("redifinition of the link output file");
    }
  }
  linkData.fileName = outFileName;
  linkData.subspec = sspec;
  RDHUtils::setVersion(linkData.rdhCopy, mUseRDHVersion);
  if (mUseRDHVersion > 6) {
    RDHUtils::setDataFormat(linkData.rdhCopy, mUseRDHDataFormat);
  }
  RDHUtils::setFEEID(linkData.rdhCopy, fee);
  RDHUtils::setCRUID(linkData.rdhCopy, cru);
  RDHUtils::setLinkID(linkData.rdhCopy, link);
  RDHUtils::setEndPointID(linkData.rdhCopy, endpoint);
  if (mUseRDHVersion >= 6) {
    RDHUtils::setSourceID(linkData.rdhCopy, o2::header::DAQID::O2toDAQ(mOrigin));
  }
  linkData.writer = this;
  linkData.updateIR = mHBFUtils.obligatorySOR ? mHBFUtils.getFirstIR() : mHBFUtils.getFirstSampledTFIR();
  linkData.buffer.reserve(mSuperPageSize);
  RDHUtils::printRDH(linkData.rdhCopy);
  LOGF(info, "Registered %s with output to %s", linkData.describe(), outFileName);
  return linkData;
}

//_____________________________________________________________________
void RawFileWriter::addData(uint16_t feeid, uint16_t cru, uint8_t lnk, uint8_t endpoint, const IR& ir, const gsl::span<char> data, bool preformatted, uint32_t trigger, uint32_t detField)
{
  // add payload to relevant links
  auto sspec = RDHUtils::getSubSpec(cru, lnk, endpoint, feeid);
  auto& link = getLinkWithSubSpec(sspec);
  if (mVerbosity > 10) {
    LOGP(info, "addData for {}  on IR BCid:{} Orbit: {}, payload: {}, preformatted: {}, trigger: {}, detField: {}", link.describe(), ir.bc, ir.orbit, data.size(), preformatted, trigger, detField);
  }
  if (isCRUDetector() && mUseRDHDataFormat == 0 && (data.size() % RDHUtils::GBTWord128)) {
    LOG(error) << "provided payload size " << data.size() << " is not multiple of GBT word size";
    throw std::runtime_error("payload size is not mutiple of GBT word size");
  }
  if (ir < mHBFUtils.getFirstSampledTFIR()) {
    LOG(warning) << "provided " << ir << " precedes first sampled TF " << mHBFUtils.getFirstSampledTFIR() << " | discarding data for " << link.describe();
    return;
  }
  if (link.discardData || ir.orbit - mHBFUtils.orbitFirst >= mHBFUtils.maxNOrbits) {
    if (!link.discardData) {
      link.discardData = true;
      LOG(info) << "Orbit " << ir.orbit << ": max. allowed orbit " << mHBFUtils.orbitFirst + mHBFUtils.maxNOrbits - 1 << " exceeded, " << link.describe() << " will discard further data";
    }
    return;
  }
  if (ir < mFirstIRAdded) {
    mHBFUtils.checkConsistency(); // done only once
    mFirstIRAdded = ir;
  }
  if (mDoLazinessCheck && !mCachingStage) {
    mDetLazyCheck.completeLinks(this, ir); // make sure that all links for previously called IR got their addData call
    mDetLazyCheck.acknowledge(sspec, ir, preformatted, trigger, detField);
  }
  link.addData(ir, data, preformatted, trigger, detField);
}

//_____________________________________________________________________
void RawFileWriter::setSuperPageSize(int nbytes)
{
  mSuperPageSize = nbytes < 16 * RDHUtils::MAXCRUPage ? RDHUtils::MAXCRUPage : nbytes;
  assert((mSuperPageSize % RDHUtils::MAXCRUPage) == 0); // make sure it is multiple of 8KB
}

//_____________________________________________________________________
IR RawFileWriter::getIRMax() const
{
  // highest IR seen so far
  IR irmax{0, 0};
  for (auto& lnk : mSSpec2Link) {
    if (irmax < lnk.second.updateIR) {
      irmax = lnk.second.updateIR;
    }
  }
  return irmax;
}

//_____________________________________________________________________
RawFileWriter::LinkData& RawFileWriter::getLinkWithSubSpec(LinkSubSpec_t ss)
{
  auto lnkIt = mSSpec2Link.find(ss);
  if (lnkIt == mSSpec2Link.end()) {
    LOGF(error, "The link for SubSpec=0x%u was not registered", ss);
    throw std::runtime_error("data for non-registered GBT link supplied");
  }
  return lnkIt->second;
}

//_____________________________________________________________________
void RawFileWriter::writeConfFile(std::string_view origin, std::string_view description, std::string_view cfgname, bool fullPath) const
{
  // write configuration file for generated data
  std::ofstream cfgfile;
  cfgfile.open(cfgname.data());
  // this is good for the global settings only, problematic for concatenation
  cfgfile << "#[defaults]" << std::endl;
  cfgfile << "#dataOrigin = " << origin << std::endl;
  cfgfile << "#dataDescription = " << description << std::endl;
  cfgfile << "#readoutCard = " << (isCRUDetector() ? "CRU" : "RORC") << std::endl;
  for (int i = 0; i < getNOutputFiles(); i++) {
    cfgfile << std::endl
            << "[input-" << mOrigin.str << '-' << i << "]" << std::endl;
    cfgfile << "dataOrigin = " << origin << std::endl;
    cfgfile << "dataDescription = " << description << std::endl;
    cfgfile << "readoutCard = " << (isCRUDetector() ? "CRU" : "RORC") << std::endl;
    cfgfile << "filePath = " << (fullPath ? o2::utils::Str::getFullPath(getOutputFileName(i)) : getOutputFileName(i)) << std::endl;
  }
  cfgfile.close();
}

//___________________________________________________________________________________
void RawFileWriter::useCaching()
{
  // impose preliminary caching of data to the tree, used in case of async. data input
  if (!mFirstIRAdded.isDummy()) {
    throw std::runtime_error("caching must be requested before feeding the data");
  }
  mCachingStage = true;
  if (mCacheFile) {
    return; // already done
  }
  auto cachename = o2::utils::Str::concat_string("_rawWriter_cache_", mOrigin.str, ::getpid(), ".root");
  mCacheFile.reset(TFile::Open(cachename.c_str(), "recreate"));
  LOG(info) << "Switched caching ON";
}

//===================================================================================

//___________________________________________________________________________________
void RawFileWriter::LinkData::cacheData(const IR& ir, const gsl::span<char> data, bool preformatted, uint32_t trigger, uint32_t detField)
{
  // cache data to temporary tree
  std::lock_guard<std::mutex> lock(writer->mCacheFileMtx);
  if (!cacheTree) {
    writer->mCacheFile->cd();
    cacheTree = std::make_unique<TTree>(o2::utils::Str::concat_string("lnk", std::to_string(subspec)).c_str(), "cache");
    cacheTree->Branch("cache", &cacheBuffer);
  }
  cacheBuffer.preformatted = preformatted;
  cacheBuffer.trigger = trigger;
  cacheBuffer.detField = detField;
  cacheBuffer.payload.resize(data.size());
  if (!data.empty()) {
    memcpy(cacheBuffer.payload.data(), data.data(), data.size());
  }
  writer->mCacheMap[ir].emplace_back(subspec, cacheTree->GetEntries());
  cacheTree->Fill();
  return;
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::addData(const IR& ir, const gsl::span<char> data, bool preformatted, uint32_t trigger, uint32_t detField)
{
  // add payload corresponding to IR, locking access to this method
  std::lock_guard<std::mutex> lock(mtx);
  addDataInternal(ir, data, preformatted, trigger, detField);
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::addDataInternal(const IR& ir, const gsl::span<char> data, bool preformatted, uint32_t trigger, uint32_t detField, bool checkEmpty)
{
  // add payload corresponding to IR
  LOG(debug) << "Adding " << data.size() << " bytes in IR " << ir << " to " << describe() << " checkEmpty=" << checkEmpty;
  if (writer->mCachingStage) {
    cacheData(ir, data, preformatted, trigger, detField);
    return;
  }
  if (startOfRun && ((writer->mHBFUtils.getFirstIRofTF(ir) > writer->mHBFUtils.getFirstIR()) && !writer->mHBFUtils.obligatorySOR)) {
    startOfRun = false;
  }

  if (startOfRun && writer->isRORCDetector()) { // in RORC mode we write separate RDH with SOX in the very beginning of the run
    writer->mHBFUtils.updateRDH<RDHAny>(rdhCopy, writer->mHBFUtils.getFirstIR(), false);
    RDHUtils::setTriggerType(rdhCopy, 0);
    openHBFPage(rdhCopy); // open new HBF just to report the SOX
    //    closeHBFPage();
  }

  int dataSize = data.size();
  if (ir >= updateIR && checkEmpty) { // new IR exceeds or equal IR of next HBF to open, insert missed HBFs if needed
    fillEmptyHBHs(ir, true);
  }
  // we are guaranteed to be under the valid RDH + possibly some data

  if (trigger) {
    auto& rdh = *getLastRDH();
    RDHUtils::setTriggerType(rdh, RDHUtils::getTriggerType(rdh) | trigger);
  }
  if (detField) {
    auto& rdh = *getLastRDH();
    RDHUtils::setDetectorField(rdh, detField);
  }

  if (!dataSize) {
    return;
  }
  if (preformatted) { // in case detectors wants to add new CRU page of predefined size
    addPreformattedCRUPage(data);
    return;
  }

  // if we are at the beginning of the page, detector may want to add some header
  if (isNewPage() && writer->newRDHFunc) {
    std::vector<char> newPageHeader;
    writer->newRDHFunc(getLastRDH(), false, newPageHeader);
    pushBack(newPageHeader.data(), newPageHeader.size());
  }

  const char* ptr = &data[0];
  // in case particular detector CRU pages need to be self-consistent, when carrying-over
  // large payload to new CRU page we may need to write optional trailer and header before
  // and after the new RDH.
  bool carryOver = false, wasSplit = false, lastSplitPart = false;
  int splitID = 0;
  std::vector<char> carryOverHeader;
  while (dataSize > 0) {

    if (carryOver) { // check if there is carry-over header to write in the buffer
      addHBFPage();  // start new CRU page, if needed, the completed superpage is flushed
      if (writer->newRDHFunc) {
        std::vector<char> newPageHeader;
        writer->newRDHFunc(getLastRDH(), false, newPageHeader);
        pushBack(newPageHeader.data(), newPageHeader.size());
      }

      // for sure after the carryOver we have space on the CRU page, no need to check
      LOG(debug) << "Adding carryOverHeader " << carryOverHeader.size()
                 << " bytes in IR " << ir << " to " << describe();
      pushBack(carryOverHeader.data(), carryOverHeader.size());
      carryOverHeader.clear();
      carryOver = false;
    }
    int sizeLeftSupPage = writer->mSuperPageSize - buffer.size();
    int sizeLeftCRUPage = RDHUtils::MAXCRUPage - (int(buffer.size()) - lastRDHoffset);
    int sizeLeft = sizeLeftCRUPage < sizeLeftSupPage ? sizeLeftCRUPage : sizeLeftSupPage;
    if (!sizeLeft || (sizeLeft < writer->mAlignmentSize)) { // this page is just over, open a new one
      addHBFPage();  // start new CRU page, if needed, the completed superpage is flushed
      if (writer->newRDHFunc) {
        std::vector<char> newPageHeader;
        writer->newRDHFunc(getLastRDH(), false, newPageHeader);
        pushBack(newPageHeader.data(), newPageHeader.size());
      }
      continue;
    }

    if (dataSize <= sizeLeft) {
      if (wasSplit && writer->mApplyCarryOverToLastPage) {
        lastSplitPart = true;
        carryOver = true;
      }
    } else {
      carryOver = true;
      wasSplit = true;
    }

    if (!carryOver) { // add all remaining data
      LOG(debug) << "Adding payload " << dataSize << " bytes in IR " << ir << " (carryover=" << carryOver << " ) to " << describe();
      pushBack(ptr, dataSize);
      dataSize = 0;
    } else { // need to carryOver payload, determine 1st wsize bytes to write starting from ptr
      if (sizeLeft > dataSize) {
        sizeLeft = dataSize;
      }
      int sizeActual = sizeLeft;
      std::vector<char> carryOverTrailer;
      if (writer->carryOverFunc) {
        sizeActual = writer->carryOverFunc(&rdhCopy, data, ptr, sizeLeft, splitID++, carryOverTrailer, carryOverHeader);
      }
      LOG(debug) << "Adding carry-over " << splitID - 1 << " fitted payload " << sizeActual << " bytes in IR " << ir << " to " << describe();
      if (sizeActual < 0 || (!lastSplitPart && (sizeActual + carryOverTrailer.size() > sizeLeft))) {
        throw std::runtime_error(std::string("wrong carry-over data size provided by carryOverMethod") + std::to_string(sizeActual));
      }
      // if there is carry-over trailer at the very last chunk, it must overwrite existing trailer
      int trailerOffset = 0;
      if (lastSplitPart) {
        trailerOffset = carryOverTrailer.size();
        if (sizeActual - trailerOffset < 0) {
          throw std::runtime_error("trailer size of last split chunk cannot exceed actual size as it overwrites the existing trailer");
        }
      }
      pushBack(ptr, sizeActual - trailerOffset); // write payload fitting to this page
      dataSize -= sizeActual;
      ptr += sizeActual;
      LOG(debug) << "Adding carryOverTrailer " << carryOverTrailer.size() << " bytes in IR " << ir << " to " << describe();
      pushBack(carryOverTrailer.data(), carryOverTrailer.size());
    }
  }
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::addPreformattedCRUPage(const gsl::span<char> data)
{
  // add preformatted CRU page w/o any attempt of splitting

  // we are guaranteed to have a page with RDH open
  int sizeLeftSupPage = writer->mSuperPageSize - buffer.size();
  if (sizeLeftSupPage < data.size()) { // we are not allowed to split this payload
    flushSuperPage(true);              // flush all but the last added RDH
  }
  if (data.size() > RDHUtils::MAXCRUPage - sizeof(RDHAny)) {
    LOG(error) << "Preformatted payload size of " << data.size() << " bytes for " << describe()
               << " exceeds max. size " << RDHUtils::MAXCRUPage - sizeof(RDHAny);
    throw std::runtime_error("preformatted payload exceeds max size");
  }
  if (int(buffer.size()) - lastRDHoffset > sizeof(RDHAny)) { // we must start from empty page
    addHBFPage();                                            // start new CRU page
  }
  pushBack(&data[0], data.size());
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::addHBFPage(bool stop)
{
  /// Add new page (RDH) to existing one for the link (possibly stop page)

  if (lastRDHoffset < 0) {
    return; // no page was open
  }
  // finalize last RDH
  auto& lastRDH = *getLastRDH();
  int psize = getCurrentPageSize(); // set the size for the previous header RDH

  if (writer->mAlignmentSize && psize % writer->mAlignmentSize != 0) { // need to pad to align to needed size
    std::vector<char> padding(writer->mAlignmentSize - psize % writer->mAlignmentSize, writer->mAlignmentPaddingFiller);
    pushBack(padding.data(), padding.size());
    psize += padding.size();
  }
  RDHUtils::setOffsetToNext(lastRDH, psize);
  RDHUtils::setMemorySize(lastRDH, psize);

  rdhCopy = lastRDH;
  bool add = true;
  if (stop && !writer->mAddSeparateHBFStopPage) {
    if (writer->isRDHStopUsed()) {
      RDHUtils::setStop(lastRDH, stop);
    }
    add = false;
  }
  if (writer->mVerbosity > 2) {
    RDHUtils::printRDH(lastRDH);
  }
  if (add) { // if we are in stopping HBF and new page is needed, add it
    // check if the superpage reached the size where it has to be flushed
    int left = writer->mSuperPageSize - buffer.size();
    if (left <= MarginToFlush) {
      flushSuperPage();
    }
    RDHUtils::setPacketCounter(rdhCopy, packetCounter++);
    RDHUtils::setPageCounter(rdhCopy, pageCnt++);
    RDHUtils::setStop(rdhCopy, stop);
    std::vector<char> userData;
    int sz = sizeof(RDHAny);
    if (stop) {
      if (writer->newRDHFunc) { // detector may want to write something in closing page
        writer->newRDHFunc(&rdhCopy, psize == sizeof(RDHAny), userData);
        sz += userData.size();
      }
      if (writer->mAlignmentSize && sz % writer->mAlignmentSize != 0) { // need to pad to align to needed size
        sz += writer->mAlignmentSize - sz % writer->mAlignmentSize;
        userData.resize(sz - sizeof(RDHAny), writer->mAlignmentPaddingFiller);
      }
    }
    RDHUtils::setOffsetToNext(rdhCopy, sz);
    RDHUtils::setMemorySize(rdhCopy, sz);
    lastRDHoffset = pushBack(rdhCopy); // entry of the new RDH
    if (!userData.empty()) {
      pushBack(userData.data(), userData.size());
    }
  }
  if (stop) {
    if (RDHUtils::getTriggerType(rdhCopy) & o2::trigger::TF) {
      nTFWritten++;
    }
    if (writer->mVerbosity > 2 && add) {
      RDHUtils::printRDH(rdhCopy);
    }
    lastRDHoffset = -1; // after closing, the previous RDH is not valid anymore
    startOfRun = false; // signal that we are definitely not in the beginning of the run
  }
  //
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::closeHBFPage()
{
  // close the HBF page, if it is empty and detector has a special treatment of empty pages
  // invoke detector callback method
  if (lastRDHoffset < 0) {
    return; // no page was open
  }
  bool emptyPage = getCurrentPageSize() == sizeof(RDHAny);
  if (emptyPage && writer->emptyHBFFunc) { // we are closing an empty page, does detector want to add something?
    std::vector<char> emtyHBFFiller;       // working space for optional empty HBF filler
    const auto rdh = getLastRDH();
    writer->emptyHBFFunc(rdh, emtyHBFFiller);
    if (!emtyHBFFiller.empty()) {
      auto ir = RDHUtils::getTriggerIR(rdh);
      LOG(debug) << "Adding empty HBF filler of size " << emtyHBFFiller.size() << " for " << describe();
      addDataInternal(ir, emtyHBFFiller, false, 0, 0, false); // add filler w/o new check for empty HBF
    }
  }
  addHBFPage(true);
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::openHBFPage(const RDHAny& rdhn, uint32_t trigger)
{
  /// create 1st page of the new HBF
  bool forceNewPage = false;
  // for RORC detectors the TF flag is absent, instead the 1st trigger after the start of TF will define the 1st be interpreted as 1st TF
  if ((RDHUtils::getTriggerType(rdhn) & o2::trigger::TF) ||
      (writer->isRORCDetector() &&
       (updateIR == writer->mHBFUtils.getFirstIR() || writer->mHBFUtils.getTF(updateIR - 1) < writer->mHBFUtils.getTF(RDHUtils::getTriggerIR(rdhn))))) {
    if (writer->mVerbosity > -10) {
      LOGF(info, "Starting new TF for link FEEId 0x%04x", RDHUtils::getFEEID(rdhn));
    }
    if (writer->mStartTFOnNewSPage && nTFWritten) { // don't flush if 1st TF
      forceNewPage = true;
    }
  }
  int left = writer->mSuperPageSize - buffer.size();
  if (forceNewPage || left <= MarginToFlush) {
    flushSuperPage();
  }
  pageCnt = 0;
  lastRDHoffset = pushBack(rdhn);
  auto& newrdh = *getLastRDH(); // dress new RDH with correct counters
  RDHUtils::setPacketCounter(newrdh, packetCounter++);
  RDHUtils::setPageCounter(newrdh, pageCnt++);
  RDHUtils::setStop(newrdh, 0);
  RDHUtils::setMemorySize(newrdh, sizeof(RDHAny));
  RDHUtils::setOffsetToNext(newrdh, sizeof(RDHAny));

  if (startOfRun && writer->isReadOutModeSet()) {
    auto trg = RDHUtils::getTriggerType(newrdh) | (writer->isContinuousReadout() ? o2::trigger::SOC : o2::trigger::SOT);
    RDHUtils::setTriggerType(newrdh, trg);
  }
  rdhCopy = newrdh;
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::flushSuperPage(bool keepLastPage)
{
  // write link superpage data to file (if requested, only up to the last page)
  size_t pgSize = (lastRDHoffset < 0 || !keepLastPage) ? buffer.size() : lastRDHoffset;
  if (writer->mVerbosity) {
    LOGF(info, "Flushing super page of %u bytes for %s", pgSize, describe());
  }
  writer->mFName2File.find(fileName)->second.write(buffer.data(), pgSize);
  auto toMove = buffer.size() - pgSize;
  if (toMove) { // is there something left in the buffer, move it to the beginning of the buffer
    if (toMove > pgSize) {
      memcpy(buffer.data(), &buffer[pgSize], toMove);
    } else {
      memmove(buffer.data(), &buffer[pgSize], toMove);
    }
    buffer.resize(toMove);
    lastRDHoffset -= pgSize;
  } else {
    buffer.clear();
    lastRDHoffset = -1;
  }
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::close(const IR& irf)
{
  // finalize link data
  // close open HBF, write empty HBFs until the end of the TF corresponding to irfin and detach from the stream
  if (writer->mFName2File.find(fileName) == writer->mFName2File.end()) {
    return; // already closed
  }
  if (writer->isCRUDetector()) { // finalize last TF
    int tf = writer->mHBFUtils.getTF(irf);
    auto finalIR = writer->mHBFUtils.getIRTF(tf + 1) - 1; // last IR of the current TF
    fillEmptyHBHs(finalIR, false);
  }
  closeHBFPage(); // close last HBF
  flushSuperPage();
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::fillEmptyHBHs(const IR& ir, bool dataAdded)
{
  // fill HBFs from last processed one to requested ir

  if (writer->isCRUDetector()) {
    std::vector<o2::InteractionRecord> irw;
    if (!writer->mHBFUtils.fillHBIRvector(irw, updateIR, ir)) {
      return;
    }
    for (const auto& irdummy : irw) {
      if (writer->mDontFillEmptyHBF &&
          writer->mHBFUtils.getTFandHBinTF(irdummy).second != 0 &&
          (!dataAdded || irdummy.orbit < ir.orbit)) {
        // even if requested, we skip empty HBF filling only if
        // 1) we are not at the new TF start
        // 2) method was called from addData and the current IR orbit is the one for which it was called (then it is not empty HB/trigger!)
        continue;
      }
      if (writer->mVerbosity > 2) {
        LOG(info) << "Adding HBF " << irdummy << " for " << describe();
      }
      closeHBFPage();                                                                 // close current HBF: add RDH with stop and update counters
      RDHUtils::setTriggerType(rdhCopy, 0);                                           // reset to avoid any detector specific flags in the dummy HBFs
      writer->mHBFUtils.updateRDH<RDHAny>(rdhCopy, irdummy, writer->isCRUDetector()); // update HBF orbit/bc and trigger flags
      openHBFPage(rdhCopy);                                                           // open new HBF
    }
    updateIR = irw.back() + o2::constants::lhc::LHCMaxBunches; //  new HBF will be generated at >= this IR
  } else {                                                     // RORC detector
    if (writer->mVerbosity > 2) {
      LOG(info) << "Adding HBF " << ir << " for " << describe();
    }
    closeHBFPage();                                          // close current HBF: add RDH with stop and update counters
    RDHUtils::setTriggerType(rdhCopy, 0);                    // reset to avoid any detector specific flags in the dummy HBFs
    writer->mHBFUtils.updateRDH<RDHAny>(rdhCopy, ir, false); // update HBF orbit/bc and trigger flags
    openHBFPage(rdhCopy);                                    // open new HBF
    updateIR = ir + 1;                                       // new Trigger in RORC detector will be generated at >= this IR
  }
}

//____________________________________________
std::string RawFileWriter::LinkData::describe() const
{
  std::stringstream ss;
  ss << "Link SubSpec=0x" << std::hex << std::setw(8) << std::setfill('0') << subspec << std::dec
     << '(' << std::setw(3) << int(RDHUtils::getCRUID(rdhCopy)) << ':' << std::setw(2) << int(RDHUtils::getLinkID(rdhCopy)) << ':'
     << int(RDHUtils::getEndPointID(rdhCopy)) << ") feeID=0x" << std::hex << std::setw(4) << std::setfill('0') << RDHUtils::getFEEID(rdhCopy);
  return ss.str();
}

//____________________________________________
void RawFileWriter::LinkData::print() const
{
  LOGF(info, "Summary for %s : NTF: %u NRDH: %u Nbytes: %u", describe(), nTFWritten, nRDHWritten, nBytesWritten);
}

//____________________________________________
size_t RawFileWriter::LinkData::pushBack(const char* ptr, size_t sz, bool keepLastOnFlash)
{
  if (!sz) {
    return buffer.size();
  }
  nBytesWritten += sz;
  // do we have a space one this superpage?
  if ((writer->mSuperPageSize - int(buffer.size())) < 0) { // need to flush
    flushSuperPage(keepLastOnFlash);
  }
  auto offs = expandBufferBy(sz);
  memmove(&buffer[offs], ptr, sz);
  return offs;
}

//================================================

//____________________________________________
void RawFileWriter::OutputFile::write(const char* data, size_t sz)
{
  std::lock_guard<std::mutex> lock(fileMtx);
  fwrite(data, 1, sz, handler); // flush to file
}

//____________________________________________
void RawFileWriter::DetLazinessCheck::acknowledge(LinkSubSpec_t s, const IR& _ir, bool _preformatted, uint32_t _trigger, uint32_t _detField)
{
  if (_ir != ir) { // unseen IR arrived
    ir = _ir;
    irSeen++;
    preformatted = _preformatted;
    trigger = _trigger;
    detField = _detField;
  }
  linksDone[s] = true;
}

//____________________________________________
void RawFileWriter::DetLazinessCheck::completeLinks(RawFileWriter* wr, const IR& _ir)
{
  if (wr->mSSpec2Link.size() == linksDone.size() || ir == _ir || ir.isDummy()) { // nothing to do
    return;
  }
  for (auto& it : wr->mSSpec2Link) {
    auto res = linksDone.find(it.first);
    if (res == linksDone.end()) {
      if (wr->mVerbosity > 10) {
        LOGP(info, "Complete {} for IR BCid:{} Orbit: {}", it.second.describe(), ir.bc, ir.orbit);
      }
      completeCount++;
      it.second.addData(ir, gsl::span<char>{}, preformatted, trigger, detField);
    }
  }
  clear();
}

void o2::raw::assertOutputDirectory(std::string_view outDirName)
{
  if (!std::filesystem::exists(outDirName)) {
#if defined(__clang__)
    // clang `create_directories` implementation is misbehaving and can
    // return false even if the directory is actually successfully created
    // so we work around that "feature" by not checking the
    // return value at all but using a second call to `exists`
    std::filesystem::create_directories(outDirName);
    if (!std::filesystem::exists(outDirName)) {
      LOG(fatal) << "could not create output directory " << outDirName;
    }
#else
    if (!std::filesystem::create_directories(outDirName)) {
      LOG(fatal) << "could not create output directory " << outDirName;
    }
#endif
  }
}
