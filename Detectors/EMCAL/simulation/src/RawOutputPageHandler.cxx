// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <fmt/format.h>
#include <Headers/RAWDataHeader.h>
#include <DetectorsRaw/HBFUtils.h>
#include <EMCALSimulation/RawOutputPageHandler.h>

using namespace o2::emcal;

RawOutputPageHandler::RawOutputPageHandler(const char* rawfilename) : mOutputStream(rawfilename), mCurrentTimeframe(-1)
{
  // create page buffers for all (40) links
  for (int ilink = 0; ilink < 40; ilink++)
    mRawPages[ilink] = RawPageBuffer();
  mOutputStream.open();
}

RawOutputPageHandler::~RawOutputPageHandler()
{
  // write pages for last timeframe to file
  for (const auto& [linkID, pagebuffer] : mRawPages) {
    // only write pages for links which send data
    if (pagebuffer.getPages().size())
      writeTimeframe(linkID, pagebuffer);
  }
}

void RawOutputPageHandler::initTrigger(const o2::InteractionRecord& irtrigger)
{
  auto currenttimeframe = raw::HBFUtils::Instance().getTF(irtrigger);
  if (currenttimeframe != mCurrentTimeframe) {
    // write pages to file
    for (auto& [linkID, pagebuffer] : mRawPages) {
      // only write pages for links which send data
      if (pagebuffer.getPages().size())
        writeTimeframe(linkID, pagebuffer);
      pagebuffer.flush();
    }
    // set the new timeframe
    mCurrentTimeframe = currenttimeframe;
  }
}

void RawOutputPageHandler::writeTimeframe(int linkID, const RawPageBuffer& pagebuffer)
{
  // write pages to file
  // Write timeframe open
  RawHeader timeframeheader;
  timeframeheader.linkID = linkID;
  mOutputStream.writeSingleHeader(timeframeheader);
  int pagecounter = 1;
  for (const auto& page : pagebuffer.getPages()) {
    auto header = page.mHeader;
    header.pageCnt = pagecounter;
    pagecounter = mOutputStream.writeData(header, gsl::span<const char>(page.mPage.data(), page.mPage.size()));
  }

  // end of timeframe
  timeframeheader.pageCnt = pagecounter;
  timeframeheader.stop = 1;
  timeframeheader.triggerType = o2::trigger::TF;
  mOutputStream.writeSingleHeader(timeframeheader);
}

void RawOutputPageHandler::addPageForLink(int linkID, const RawHeader& header, const std::vector<char>& page)
{
  if (linkID > 40)
    throw LinkIDException(linkID);
  mRawPages[linkID].addPage(header, page);
}

RawOutputPageHandler::LinkIDException::LinkIDException(int linkID) : std::exception(), mLinkID(linkID), mMessage()
{
  mMessage = fmt::format("Link ID invalid: %d (max 40)", linkID);
}