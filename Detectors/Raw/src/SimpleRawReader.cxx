// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SimpleRawReader.cxx
/// \brief Simple reader for non-DPL tests
#include "Framework/Logger.h"
#include "DetectorsRaw/SimpleRawReader.h"

namespace o2
{
namespace itsmft
{

///======================================================================
///                 Simple reader for non-DPL tests
///
/// Allows to load the whole TF to the memory and navigate over CRU pages
/// of individual GBT links
///
///======================================================================

///_________________________________________________________________
/// Init the reader from the config string, create buffer
void SimpleRawReader::init()
{
  if (initDone) {
    return;
  }
  reader = std::make_unique<o2::raw::RawFileReader>(cfgName); // init from configuration file
  uint32_t errCheck = 0xffffffff;
  errCheck ^= 0x1 << o2::raw::RawFileReader::ErrNoSuperPageForTF; // makes no sense for superpages not interleaved by others
  reader->setCheckErrors(errCheck);
  reader->init();
  buffers.resize(reader->getNLinks());
  initDone = true;
}

///_________________________________________________________________
/// read data of all links for next TF, return number of links with data
int SimpleRawReader::loadNextTF()
{
  if (!reader) {
    init();
  }
  int nLinks = reader->getNLinks(), nread = 0;
  for (int il = 0; il < nLinks; il++) {
    auto& buff = buffers[il].first;
    buffers[il].second = 0; // reset position in the buffer to its head
    buff.clear();
    auto& lnk = reader->getLink(il);
    auto sz = lnk.getNextTFSize();
    if (!sz) {
      continue;
    }
    buff.resize(sz);
    lnk.readNextTF(buff.data());
    LOG(INFO) << "Loaded " << sz << " bytes for " << lnk.describe();
    nread++;
  }
  return nread;
}

///_________________________________________________________________
/// provide access to the next CRU page of the load TF data for the link il
const gsl::span<char> SimpleRawReader::getNextPage(int il)
{
  auto& buff = buffers[il].first;
  auto& pos = buffers[il].second;
  if (pos >= buff.size()) {
    return gsl::span<char>();
  }
  auto rdh = reinterpret_cast<RDH*>(&buff[pos]);
  gsl::span<char> page{&buff[pos], rdh->memorySize};
  pos += rdh->offsetToNext;
  return page;
}

} // namespace itsmft
} // namespace o2
