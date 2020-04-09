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
  auto irmax = getIRMax();
  irmax--; // latest (among all links) HBF to open, we want just to close the last one
  for (auto& lnk : mSSpec2Link) {
    lnk.second.close(irmax);
    lnk.second.print();
  }
  //
  // close all files
  for (auto& flh : mFName2File) {
    LOG(INFO) << "Closing output file " << flh.first;
    fclose(flh.second.handler);
    flh.second.handler = nullptr;
  }
  mFName2File.clear();
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
  linkData.rdhCopy.feeId = fee;
  linkData.rdhCopy.cruID = cru;
  linkData.rdhCopy.linkID = link;
  linkData.rdhCopy.endPointID = endpoint;
  linkData.writer = this;
  linkData.updateIR = mHBFUtils.getFirstIR();
  linkData.buffer.reserve(mSuperPageSize);
  LOGF(INFO, "Registered %s with output to %s", linkData.describe(), outFileName);
  return linkData;
}

//_____________________________________________________________________
RawFileWriter::LinkData& RawFileWriter::registerLink(const RDH& rdh, std::string_view outFileName)
{
  // register the GBT link and its output file
  auto& linkData = registerLink(rdh.feeId, rdh.cruID, rdh.linkID, rdh.endPointID, outFileName);
  linkData.rdhCopy.detectorField = rdh.detectorField;
  return linkData;
}

//_____________________________________________________________________
void RawFileWriter::addData(uint16_t feeid, uint16_t cru, uint8_t lnk, uint8_t endpoint, const IR& ir, const gsl::span<char> data, bool preformatted)
{
  // add payload to relevant links
  if (data.size() % RDHUtils::GBTWord) {
    LOG(ERROR) << "provided payload size " << data.size() << " is not multiple of GBT word size";
    throw std::runtime_error("payload size is not mutiple of GBT word size");
  }
  auto sspec = RDHUtils::getSubSpec(cru, lnk, endpoint, feeid);
  auto& link = getLinkWithSubSpec(sspec);
  link.addData(ir, data, preformatted);
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
  for (int i = 0; i < getNOutputFiles(); i++) {
    cfgfile << std::endl
            << "[input-" << i << "]" << std::endl;
    cfgfile << "dataOrigin = " << origin << std::endl;
    cfgfile << "dataDescription = " << description << std::endl;
    cfgfile << "filePath = " << (fullPath ? o2::base::NameConf::getFullPath(getOutputFileName(i)) : getOutputFileName(i)) << std::endl;
  }
  cfgfile.close();
}

//===================================================================================

//___________________________________________________________________________________
void RawFileWriter::LinkData::addData(const IR& ir, const gsl::span<char> data, bool preformatted)
{
  // add payload corresponding to IR
  LOG(DEBUG) << "Adding " << data.size() << " bytes in IR " << ir << " to " << describe();
  std::lock_guard<std::mutex> lock(mtx);
  if (ir >= updateIR) { // new IR exceeds or equal IR of next HBF to open, insert missed HBFs if needed
    fillEmptyHBHs(ir);
  }
  // we are guaranteed to be under the valid RDH + possibly some data
  int dataSize = data.size();
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
        sizeActual = writer->carryOverFunc(rdhCopy, data, ptr, sizeLeft, splitID++, carryOverTrailer, carryOverHeader);
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
  if (data.size() > RDHUtils::MAXCRUPage - sizeof(RDH)) {
    LOG(ERROR) << "Preformatted payload size of " << data.size() << " bytes for " << describe()
               << " exceeds max. size " << RDHUtils::MAXCRUPage - sizeof(RDH);
    throw std::runtime_error("preformatted payload exceeds max size");
  }
  if (int(buffer.size()) - lastRDHoffset > sizeof(RDH)) { // we must start from empty page
    addHBFPage();                                         // start new CRU page
  }
  pushBack(&data[0], data.size());
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::addHBFPage(bool stop)
{
  /// Add new page (RDH) to existing one for the link (possibly stop page)

  // check if the superpage reached the size where it hase to be flushed
  if (lastRDHoffset < 0) {
    return; // no page was open
  }
  // finalize last RDH
  auto* lastRDH = getLastRDH();
  int psize = buffer.size() - lastRDHoffset;                  // set the size for the previous header RDH
  if (stop && psize == sizeof(RDH) && writer->emptyHBFFunc) { // we are closing an empty page, does detector want to add something?
    std::vector<char> emtyHBFFiller;                          // working space for optional empty HBF filler
    writer->emptyHBFFunc(*lastRDH, emtyHBFFiller);
    if (emtyHBFFiller.size()) {
      LOG(DEBUG) << "Adding empty HBF filler of size " << emtyHBFFiller.size() << " for " << describe();
      pushBack(emtyHBFFiller.data(), emtyHBFFiller.size());
      psize += emtyHBFFiller.size();
    }
  }
  lastRDH->offsetToNext = lastRDH->memorySize = psize;

  if (writer->mVerbosity > 2) {
    RDHUtils::printRDH(*lastRDH);
  }
  rdhCopy = *lastRDH;
  int left = writer->mSuperPageSize - buffer.size();
  if (left <= MarginToFlush) {
    flushSuperPage();
  }
  rdhCopy.packetCounter = packetCounter++;
  rdhCopy.pageCnt = pageCnt++;
  rdhCopy.stop = stop;
  rdhCopy.offsetToNext = rdhCopy.memorySize = sizeof(RDH);
  lastRDHoffset = pushBack(rdhCopy); // entry of the new RDH
  if (stop) {
    if (rdhCopy.triggerType & o2::trigger::TF) {
      nTFWritten++;
    }
    if (writer->mVerbosity > 2) {
      RDHUtils::printRDH(rdhCopy);
    }
    lastRDHoffset = -1; // after closing, the previous RDH is not valid anymore
    startOfRun = false; // signal that we are definitely not in the beginning of the run
  }
  //
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::openHBFPage(const RDH& rdhn)
{
  /// create 1st page of the new HBF
  bool forceNewPage = false;
  if (rdhn.triggerType & o2::trigger::TF) {
    if (writer->mVerbosity > 0) {
      LOGF(INFO, "Starting new TF for link FEEId 0x%04x", rdhn.feeId);
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
  RDH* newRDH = getLastRDH(); // fetch new RDH
  newRDH->packetCounter = packetCounter++;
  newRDH->pageCnt = pageCnt++;
  newRDH->stop = 0;
  newRDH->memorySize = newRDH->offsetToNext = sizeof(RDH);
  if (startOfRun && writer->isReadOutModeSet()) {
    newRDH->triggerType |= writer->isContinuousReadout() ? o2::trigger::SOC : o2::trigger::SOT;
  }
  rdhCopy = *newRDH;
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
  auto irfin = irf;
  if (irfin < updateIR) {
    irfin = updateIR;
  }
  int tf = writer->mHBFUtils.getTF(irfin);
  auto finalIR = writer->mHBFUtils.getIRTF(tf + 1) - 1; // last IR of the current TF
  fillEmptyHBHs(finalIR);
  closeHBFPage(); // close last HBF
  flushSuperPage();
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::fillEmptyHBHs(const IR& ir)
{
  // fill HBFs from last processed one to requested ir
  std::vector<o2::InteractionRecord> irw;
  if (!writer->mHBFUtils.fillHBIRvector(irw, updateIR, ir)) {
    return;
  }
  for (const auto& irdummy : irw) {
    if (writer->mVerbosity > 2) {
      LOG(INFO) << "Adding HBF " << irdummy << " for " << describe();
    }
    closeHBFPage();                                     // close current HBF: add RDH with stop and update counters
    rdhCopy.triggerType = 0;                            // reset to avoid any detector specific flags in the dummy HBFs
    writer->mHBFUtils.updateRDH<RDH>(rdhCopy, irdummy); // update HBF orbit/bc and trigger flags
    openHBFPage(rdhCopy);                               // open new HBF
  }
  updateIR = irw.back() + o2::constants::lhc::LHCMaxBunches; // new HBF will be generated at >= this IR
}

//____________________________________________
std::string RawFileWriter::LinkData::describe() const
{
  std::stringstream ss;
  ss << "Link SubSpec=0x" << std::hex << std::setw(8) << std::setfill('0')
     << RDHUtils::getSubSpec(rdhCopy.cruID, rdhCopy.linkID, rdhCopy.endPointID, rdhCopy.feeId) << std::dec
     << '(' << std::setw(3) << int(rdhCopy.cruID) << ':' << std::setw(2) << int(rdhCopy.linkID) << ':'
     << int(rdhCopy.endPointID) << ") feeID=0x" << std::hex << std::setw(4) << std::setfill('0') << rdhCopy.feeId;
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
