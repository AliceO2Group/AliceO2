// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "DetectorsRaw/HBFUtils.h"
#include "CommonConstants/Triggers.h"
#include "Framework/Logger.h"

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

  if (!mFirstIRAdded.isDummy()) { // flushing and completing the last HBF makes sense only if data was added.
    auto irmax = getIRMax();
    for (auto& lnk : mSSpec2Link) {
      lnk.second.close(irmax);
      lnk.second.print();
    }
  }
  //
  // close all files
  for (auto& flh : mFName2File) {
    LOG(INFO) << "Closing output file " << flh.first;
    fclose(flh.second.handler);
    flh.second.handler = nullptr;
  }
  mFName2File.clear();
  mTimer.Stop();
  mTimer.Print();
}

//_____________________________________________________________________
void RawFileWriter::fillFromCache()
{
  LOG(INFO) << "Filling links from cached trees";
  mCachingStage = false;
  for (const auto& cache : mCacheMap) {
    for (const auto& entry : cache.second) {
      auto& link = getLinkWithSubSpec(entry.first);
      link.cacheTree->GetEntry(entry.second);
      link.addData(cache.first, link.cacheBuffer.payload, link.cacheBuffer.preformatted, link.cacheBuffer.trigger);
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
      LOG(ERROR) << "Failed to open output file " << outFileName;
      throw std::runtime_error(std::string("cannot open link output file ") + outFileName);
    }
  }
  if (!linkData.fileName.empty()) { // this link was already declared and associated with a file
    if (linkData.fileName == outFileName) {
      LOGF(INFO, "Link 0x%ux was already declared with same output, do nothing", sspec);
      return linkData;
    } else {
      LOGF(ERROR, "Link 0x%ux was already connected to different output file %s", sspec, linkData.fileName);
      throw std::runtime_error("redifinition of the link output file");
    }
  }
  linkData.fileName = outFileName;
  linkData.subspec = sspec;
  RDHUtils::setVersion(linkData.rdhCopy, mUseRDHVersion);
  RDHUtils::setFEEID(linkData.rdhCopy, fee);
  RDHUtils::setCRUID(linkData.rdhCopy, cru);
  RDHUtils::setLinkID(linkData.rdhCopy, link);
  RDHUtils::setEndPointID(linkData.rdhCopy, endpoint);
  if (mUseRDHVersion >= 6) {
    RDHUtils::setSourceID(linkData.rdhCopy, o2::header::DAQID::O2toDAQ(mOrigin));
  }
  linkData.writer = this;
  linkData.updateIR = mHBFUtils.getFirstIR();
  linkData.buffer.reserve(mSuperPageSize);
  RDHUtils::printRDH(linkData.rdhCopy);
  LOGF(INFO, "Registered %s with output to %s", linkData.describe(), outFileName);
  return linkData;
}

//_____________________________________________________________________
void RawFileWriter::addData(uint16_t feeid, uint16_t cru, uint8_t lnk, uint8_t endpoint, const IR& ir, const gsl::span<char> data, bool preformatted, uint32_t trigger)
{
  // add payload to relevant links
  if (data.size() % RDHUtils::GBTWord) {
    LOG(ERROR) << "provided payload size " << data.size() << " is not multiple of GBT word size";
    throw std::runtime_error("payload size is not mutiple of GBT word size");
  }
  auto sspec = RDHUtils::getSubSpec(cru, lnk, endpoint, feeid);
  auto& link = getLinkWithSubSpec(sspec);
  if (ir < mHBFUtils.getFirstIR()) {
    LOG(WARNING) << "provided " << ir << " precedes first TF " << mHBFUtils.getFirstIR() << " | discarding data for " << link.describe();
    return;
  }
  if (ir < mFirstIRAdded) {
    mFirstIRAdded = ir;
  }
  link.addData(ir, data, preformatted, trigger);
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
    LOGF(ERROR, "The link for SubSpec=0x%u was not registered", ss);
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
    cfgfile << "filePath = " << (fullPath ? o2::base::NameConf::getFullPath(getOutputFileName(i)) : getOutputFileName(i)) << std::endl;
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
  auto cachename = o2::utils::concat_string("_rawWriter_cache_", mOrigin.str, ::getpid(), ".root");
  mCacheFile.reset(TFile::Open(cachename.c_str(), "recreate"));
  LOG(INFO) << "Switched caching ON";
}

//===================================================================================

//___________________________________________________________________________________
void RawFileWriter::LinkData::cacheData(const IR& ir, const gsl::span<char> data, bool preformatted, uint32_t trigger)
{
  // cache data to temporary tree
  std::lock_guard<std::mutex> lock(writer->mCacheFileMtx);
  if (!cacheTree) {
    writer->mCacheFile->cd();
    cacheTree = std::make_unique<TTree>(o2::utils::concat_string("lnk", std::to_string(subspec)).c_str(), "cache");
    cacheTree->Branch("cache", &cacheBuffer);
  }
  cacheBuffer.preformatted = preformatted;
  cacheBuffer.trigger = trigger;
  cacheBuffer.payload.resize(data.size());
  if (!data.empty()) {
    memcpy(cacheBuffer.payload.data(), data.data(), data.size());
  }
  writer->mCacheMap[ir].emplace_back(subspec, cacheTree->GetEntries());
  cacheTree->Fill();
  return;
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::addData(const IR& ir, const gsl::span<char> data, bool preformatted, uint32_t trigger)
{
  // add payload corresponding to IR
  LOG(DEBUG) << "Adding " << data.size() << " bytes in IR " << ir << " to " << describe();
  std::lock_guard<std::mutex> lock(mtx);

  if (writer->mCachingStage) {
    cacheData(ir, data, preformatted, trigger);
    return;
  }

  if (startOfRun && writer->isRORCDetector()) { // in RORC mode we write separate RDH with SOX in the very beginning of the run
    writer->mHBFUtils.updateRDH<RDHAny>(rdhCopy, writer->mHBFUtils.getFirstIR(), false);
    RDHUtils::setTriggerType(rdhCopy, 0);
    openHBFPage(rdhCopy); // open new HBF just to report the SOX
    //    closeHBFPage();
  }

  int dataSize = data.size();
  if (ir >= updateIR) { // new IR exceeds or equal IR of next HBF to open, insert missed HBFs if needed
    fillEmptyHBHs(ir, true);
  }
  // we are guaranteed to be under the valid RDH + possibly some data

  if (trigger) {
    auto& rdh = *getLastRDH();
    RDHUtils::setTriggerType(rdh, RDHUtils::getTriggerType(rdh) | trigger);
  }

  if (!dataSize) {
    return;
  }
  if (preformatted) { // in case detectors wants to add new CRU page of predefined size
    addPreformattedCRUPage(data);
    return;
  }
  const char* ptr = &data[0];
  // in case particular detector CRU pages need to be self-consistent, when carrying-over
  // large payload to new CRU page we may need to write optional trailer and header before
  // and after the new RDH.
  bool carryOver = false;
  int splitID = 0;
  std::vector<char> carryOverHeader;
  while (dataSize > 0) {
    if (carryOver) { // check if there is carry-over header to write in the buffer
      addHBFPage();  // start new CRU page, if needed, the completed superpage is flushed
      // for sure after the carryOver we have space on the CRU page, no need to check
      LOG(DEBUG) << "Adding carryOverHeader " << carryOverHeader.size()
                 << " bytes in IR " << ir << " to " << describe();
      pushBack(carryOverHeader.data(), carryOverHeader.size());
      carryOverHeader.clear();
      carryOver = false;
    }
    int sizeLeftSupPage = writer->mSuperPageSize - buffer.size();
    int sizeLeftCRUPage = RDHUtils::MAXCRUPage - (int(buffer.size()) - lastRDHoffset);
    int sizeLeft = sizeLeftCRUPage < sizeLeftSupPage ? sizeLeftCRUPage : sizeLeftSupPage;
    if (!sizeLeft) { // this page is just over, open a new one
      addHBFPage();  // start new CRU page, if needed, the completed superpage is flushed
      continue;
    }
    if (dataSize <= sizeLeft) { // add all remaining data
      LOG(DEBUG) << "Adding payload " << dataSize << " bytes in IR " << ir << " (carryover=" << carryOver << " ) to " << describe();
      pushBack(ptr, dataSize);
      dataSize = 0;
    } else { // need to carryOver payload, determine 1st wsize bytes to write starting from ptr
      carryOver = true;
      int sizeActual = sizeLeft;
      std::vector<char> carryOverTrailer;
      if (writer->carryOverFunc) {
        sizeActual = writer->carryOverFunc(&rdhCopy, data, ptr, sizeLeft, splitID++, carryOverTrailer, carryOverHeader);
      }
      LOG(DEBUG) << "Adding carry-over " << splitID - 1 << " fitted payload " << sizeActual << " bytes in IR " << ir << " to " << describe();
      if (sizeActual < 0 || sizeActual + carryOverTrailer.size() > sizeLeft) {
        throw std::runtime_error(std::string("wrong carry-over data size provided by carryOverMethod") + std::to_string(sizeActual));
      }
      pushBack(ptr, sizeActual); // write payload fitting to this page
      dataSize -= sizeActual;
      ptr += sizeActual;
      LOG(DEBUG) << "Adding carryOverTrailer " << carryOverTrailer.size() << " bytes in IR "
                 << ir << " to " << describe();
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
    LOG(ERROR) << "Preformatted payload size of " << data.size() << " bytes for " << describe()
               << " exceeds max. size " << RDHUtils::MAXCRUPage - sizeof(RDHAny);
    throw std::runtime_error("preformatted payload exceeds max size");
  }
  if (int(buffer.size()) - lastRDHoffset > sizeof(RDHAny)) { // we must start from empty page
    addHBFPage();                                         // start new CRU page
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
  int psize = buffer.size() - lastRDHoffset;                  // set the size for the previous header RDH
  if (stop && psize == sizeof(RDHAny) && writer->emptyHBFFunc) { // we are closing an empty page, does detector want to add something?
    std::vector<char> emtyHBFFiller;                          // working space for optional empty HBF filler
    writer->emptyHBFFunc(&lastRDH, emtyHBFFiller);
    if (emtyHBFFiller.size()) {
      LOG(DEBUG) << "Adding empty HBF filler of size " << emtyHBFFiller.size() << " for " << describe();
      pushBack(emtyHBFFiller.data(), emtyHBFFiller.size());
      psize += emtyHBFFiller.size();
    }
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
    // check if the superpage reached the size where it hase to be flushed
    int left = writer->mSuperPageSize - buffer.size();
    if (left <= MarginToFlush) {
      flushSuperPage();
    }
    RDHUtils::setPacketCounter(rdhCopy, packetCounter++);
    RDHUtils::setPageCounter(rdhCopy, pageCnt++);
    RDHUtils::setStop(rdhCopy, stop);
    RDHUtils::setOffsetToNext(rdhCopy, sizeof(RDHAny));
    RDHUtils::setMemorySize(rdhCopy, sizeof(RDHAny));
    lastRDHoffset = pushBack(rdhCopy); // entry of the new RDH
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
void RawFileWriter::LinkData::openHBFPage(const RDHAny& rdhn, uint32_t trigger)
{
  /// create 1st page of the new HBF
  bool forceNewPage = false;

  // for RORC detectors the TF flag is absent, instead the 1st trigger after the start of TF will define the 1st be interpreted as 1st TF
  auto newTF_RORC = [this, &rdhn]() -> bool {
    auto tfhbPrev = writer->mHBFUtils.getTFandHBinTF(this->updateIR.bc ? this->updateIR - 1 : this->updateIR); // updateIR was advanced by 1 BC wrt IR of the previous update
    return this->writer->mHBFUtils.getTFandHBinTF(RDHUtils::getTriggerIR(rdhn)).first > tfhbPrev.first;        // new TF_ID exceeds old one
  };

  if ((RDHUtils::getTriggerType(rdhn) & o2::trigger::TF) || (writer->isRORCDetector() && newTF_RORC())) {
    if (writer->mVerbosity > -10) {
      LOGF(INFO, "Starting new TF for link FEEId 0x%04x", RDHUtils::getFEEID(rdhn));
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
    LOGF(INFO, "Flushing super page of %u bytes for %s", pgSize, describe());
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
    auto irfin = irf;
    if (irfin < updateIR) {
      irfin = updateIR;
    }
    int tf = writer->mHBFUtils.getTF(irfin);
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
          (!dataAdded || irdummy < ir)) {
        // even if requested, we skip empty HBF filling only if
        // 1) we are not at the new TF start
        // 2) method was called from addData and the current IR is the one for which it was called (then it is not empty HB/trigger!)
        continue;
      }
      if (writer->mVerbosity > 2) {
        LOG(INFO) << "Adding HBF " << irdummy << " for " << describe();
      }
      closeHBFPage();                                                                 // close current HBF: add RDH with stop and update counters
      RDHUtils::setTriggerType(rdhCopy, 0);                                           // reset to avoid any detector specific flags in the dummy HBFs
      writer->mHBFUtils.updateRDH<RDHAny>(rdhCopy, irdummy, writer->isCRUDetector()); // update HBF orbit/bc and trigger flags
      openHBFPage(rdhCopy);                                                           // open new HBF
    }
    updateIR = irw.back() + o2::constants::lhc::LHCMaxBunches; //  new HBF will be generated at >= this IR
  } else {                                                     // RORC detector
    if (writer->mVerbosity > 2) {
      LOG(INFO) << "Adding HBF " << ir << " for " << describe();
    }
    closeHBFPage();                                     // close current HBF: add RDH with stop and update counters
    RDHUtils::setTriggerType(rdhCopy, 0);               // reset to avoid any detector specific flags in the dummy HBFs
    writer->mHBFUtils.updateRDH<RDHAny>(rdhCopy, ir, false); // update HBF orbit/bc and trigger flags
    openHBFPage(rdhCopy);                               // open new HBF
    updateIR = ir + 1;                                  // new Trigger in RORC detector will be generated at >= this IR
  }
}

//____________________________________________
std::string RawFileWriter::LinkData::describe() const
{
  std::stringstream ss;
  ss << "Link SubSpec=0x" << std::hex << std::setw(8) << std::setfill('0')
     << RDHUtils::getSubSpec(rdhCopy) << std::dec
     << '(' << std::setw(3) << int(RDHUtils::getCRUID(rdhCopy)) << ':' << std::setw(2) << int(RDHUtils::getLinkID(rdhCopy)) << ':'
     << int(RDHUtils::getEndPointID(rdhCopy)) << ") feeID=0x" << std::hex << std::setw(4) << std::setfill('0') << RDHUtils::getFEEID(rdhCopy);
  return ss.str();
}

//____________________________________________
void RawFileWriter::LinkData::print() const
{
  LOGF(INFO, "Summary for %s : NTF: %u NRDH: %u Nbytes: %u", describe(), nTFWritten, nRDHWritten, nBytesWritten);
}

//================================================

void RawFileWriter::OutputFile::write(const char* data, size_t sz)
{
  std::lock_guard<std::mutex> lock(fileMtx);
  fwrite(data, 1, sz, handler); // flush to file
}
