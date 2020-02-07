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
#include "DetectorsRaw/RawFileWriter.h"
#include "CommonConstants/Triggers.h"
#include "Framework/Logger.h"

using namespace o2::raw;
namespace o2h = o2::header;

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
  mIRMax--; // mIRMax is pointing to the beginning of latest (among all links) HBF to open, we want just to close the last one
  for (auto& lnk : mSSpec2Link) {
    lnk.second.updateIR = mIRMax;
    lnk.second.close(mIRMax);
    lnk.second.print();
  }
  //
  // close all files
  for (auto& flh : mFName2File) {
    LOG(INFO) << "Closing output file " << flh.first;
    fclose(flh.second);
  }
  mFName2File.clear();
}

//_____________________________________________________________________
void RawFileWriter::registerLink(uint16_t fee, uint16_t cru, uint8_t link, uint8_t endpoint, const std::string& outFileName)
{
  // register the GBT link and its output file

  auto sspec = HBFUtils::getSubSpec(cru, link, endpoint);
  auto& linkData = getLinkWithSubSpec(sspec);
  auto* file = mFName2File[outFileName];
  if (!file) {
    if ((file = fopen(outFileName.c_str(), "wb"))) { // if file does not exist, create it
      mFName2File[outFileName] = file;
    } else {
      LOG(ERROR) << "Failed to open output file " << outFileName;
      throw std::runtime_error(std::string("cannot open link output file ") + outFileName);
    }
  }
  if (linkData.file) { // this link was already declared and associated with a file
    if (linkData.file == file) {
      LOGF(INFO, "Link 0x%ux was already declared with same output, do nothing", sspec);
      return;
    } else {
      LOGF(ERROR, "Link 0x%ux was already connected to different output file %s", sspec, linkData.fileName);
      throw std::runtime_error("redifinition of the link output file");
    }
  }
  linkData.fileName = outFileName;
  linkData.file = file;
  linkData.subspec = sspec;
  linkData.rdhCopy.feeId = fee;
  linkData.rdhCopy.cruID = cru;
  linkData.rdhCopy.linkID = link;
  linkData.rdhCopy.endPointID = endpoint;
  linkData.writer = this;
  linkData.updateIR = mHBFUtils.getFirstIR();
  linkData.buffer.reserve(mSuperPageSize);
  LOGF(INFO, "Registered %s with output to %s", linkData.describe(), outFileName);
}

//_____________________________________________________________________
void RawFileWriter::registerLink(const RDH& rdh, const std::string& outFileName)
{
  // register the GBT link and its output file
  registerLink(rdh.feeId, rdh.cruID, rdh.linkID, rdh.endPointID, outFileName);
  auto& linkData = getLinkWithSubSpec(rdh);
  linkData.rdhCopy.detectorField = rdh.detectorField;
}

//_____________________________________________________________________
void RawFileWriter::addData(uint16_t cru, uint8_t lnk, uint8_t endpoint, const IR& ir, const gsl::span<char> data)
{
  // add payload to relevant links
  if (data.size() % HBFUtils::GBTWord) {
    LOG(ERROR) << "provided payload size " << data.size() << " is not multiple of GBT word size";
    throw std::runtime_error("payload size is not mutiple of GBT word size");
  }
  auto sspec = HBFUtils::getSubSpec(cru, lnk, endpoint);
  if (!isLinkRegistered(sspec)) {
    LOGF(ERROR, "The link for SubSpec=0x%ux(%u:%u:%u) was not registered", sspec, cru, lnk, endpoint);
    throw std::runtime_error("data for non-registered GBT link supplied");
  }
  auto& link = getLinkWithSubSpec(sspec);
  link.addData(ir, data);

  if (mIRMax < link.updateIR) { // remember highest IR seen
    mIRMax = link.updateIR;
  }
}

//===================================================================================
std::vector<char> RawFileWriter::LinkData::sCarryOverTrailer;
std::vector<char> RawFileWriter::LinkData::sCarryOverHeader;
std::vector<char> RawFileWriter::LinkData::sEmptyHBFFiller;
std::vector<o2::InteractionRecord> RawFileWriter::LinkData::sIRWork;

RawFileWriter::LinkData::~LinkData()
{
  // updateIR is pointing to the beginning of next HBF to open, we want just to close the last one
  close(--updateIR);
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::addData(const IR& ir, const gsl::span<char> data)
{
  // add payload corresponding to IR
  LOG(DEBUG) << "Adding " << data.size() << " bytes in IR " << ir << " to " << describe();

  if (ir >= updateIR) { // new IR exceeds or equal IR of next HBF to open, insert missed HBFs if needed
    fillEmptyHBHs(ir);
  }
  // we are guaranteed to be under the valid RDH + possibly some data
  int dataSize = data.size();
  if (!dataSize) {
    return;
  }
  const char* ptr = &data[0];
  // in case particular detector CRU pages need to be self-consistent, when carrying-over
  // large payload to new CRU page we may need to write optional trailer and header before
  // and after the new RDH.
  bool carryOver = false;
  int splitID = 0;
  while (dataSize > 0) {
    if (carryOver) { // check if there is carry-over header to write in the buffer
      addHBFPage();  // start new CRU page, if needed, the completed superpage is flushed
      // for sure after the carryOver we have space on the CRU page, no need to check
      LOG(DEBUG) << "Adding carryOverHeader " << sCarryOverHeader.size()
                 << " bytes in IR " << ir << " to " << describe();
      pushBack(sCarryOverHeader.data(), sCarryOverHeader.size());
      sCarryOverHeader.clear();
      carryOver = false;
    }
    int sizeLeftSupPage = writer->mSuperPageSize - buffer.size();
    int sizeLeftCRUPage = HBFUtils::MAXCRUPage - (int(buffer.size()) - lastRDHoffset);
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
      if (writer->carryOverFunc) {
        sizeActual = writer->carryOverFunc(rdhCopy, data, ptr, sizeLeft, splitID++, sCarryOverTrailer, sCarryOverHeader);
      }
      LOG(DEBUG) << "Adding carry-over " << splitID - 1 << " fitted payload " << sizeActual << " bytes in IR " << ir << " to " << describe();
      if (sizeActual < 0 || sizeActual + sCarryOverTrailer.size() > sizeLeft) {
        throw std::runtime_error(std::string("wrong carry-over data size provided by carryOverMethod") + std::to_string(sizeActual));
      }
      pushBack(ptr, sizeActual); // write payload fitting to this page
      dataSize -= sizeActual;
      ptr += sizeActual;
      LOG(DEBUG) << "Adding carryOverTrailer " << sCarryOverTrailer.size() << " bytes in IR "
                 << ir << " to " << describe();
      pushBack(sCarryOverTrailer.data(), sCarryOverTrailer.size());
      sCarryOverTrailer.clear();
    }
  }
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
    writer->emptyHBFFunc(*lastRDH, sEmptyHBFFiller);
    if (sEmptyHBFFiller.size()) {
      LOG(DEBUG) << "Adding empty HBF filler of size " << sEmptyHBFFiller.size() << " for " << describe();
      pushBack(sEmptyHBFFiller.data(), sEmptyHBFFiller.size());
      psize += sEmptyHBFFiller.size();
      sEmptyHBFFiller.clear();
    }
  }
  lastRDH->offsetToNext = lastRDH->memorySize = psize;

  if (writer->mVerbosity > 2) {
    HBFUtils::printRDH(*lastRDH);
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
      HBFUtils::printRDH(rdhCopy);
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
  // write link superpage data to file
  size_t pgSize = (lastRDHoffset < 0 || !keepLastPage) ? buffer.size() : lastRDHoffset;
  if (writer->mVerbosity) {
    LOGF(INFO, "Flushing super page of %u bytes for %s", pgSize, describe());
  }
  fwrite(buffer.data(), 1, pgSize, file); // write to file
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
  if (!file) {
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
  file = nullptr;
}

//___________________________________________________________________________________
void RawFileWriter::LinkData::fillEmptyHBHs(const IR& ir)
{
  // fill HBFs from last processed one to requested ir
  if (!writer->mHBFUtils.fillHBIRvector(sIRWork, updateIR, ir)) {
    return;
  }
  for (const auto& irdummy : sIRWork) {
    if (writer->mVerbosity > 2) {
      LOG(INFO) << "Adding HBF " << irdummy << " for " << describe();
    }
    closeHBFPage();                                     // close current HBF: add RDH with stop and update counters
    rdhCopy.triggerType = 0;                            // reset to avoid any detector specific flags in the dummy HBFs
    writer->mHBFUtils.updateRDH<RDH>(rdhCopy, irdummy); // update HBF orbit/bc and trigger flags
    openHBFPage(rdhCopy);                               // open new HBF
  }
  updateIR = sIRWork.back() + o2::constants::lhc::LHCMaxBunches; // new HBF will be generated at >= this IR
}

//____________________________________________
std::string RawFileWriter::LinkData::describe() const
{
  std::stringstream ss;
  ss << "Link SubSpec=0x" << std::hex << std::setw(8) << std::setfill('0')
     << HBFUtils::getSubSpec(rdhCopy.cruID, rdhCopy.linkID, rdhCopy.endPointID) << std::dec
     << '(' << std::setw(3) << int(rdhCopy.cruID) << ':' << std::setw(2) << int(rdhCopy.linkID) << ':'
     << int(rdhCopy.endPointID) << ") feeID=0x" << std::hex << std::setw(4) << std::setfill('0') << rdhCopy.feeId;
  return ss.str();
}

//____________________________________________
void RawFileWriter::LinkData::print() const
{
  LOGF(INFO, "Summary for %s : NTF: %u NRDH: %u Nbytes: %u", describe(), nTFWritten, nRDHWritten, nBytesWritten);
}
