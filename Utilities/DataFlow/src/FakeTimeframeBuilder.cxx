// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFlow/FakeTimeframeBuilder.h"
#include "TimeFrame/TimeFrame.h"
#include "Headers/DataHeader.h"
#include <string>
#include <vector>
#include <functional>
#include <cstring>

using DataHeader = o2::header::DataHeader;
using DataDescription = o2::header::DataDescription;
using DataOrigin = o2::header::DataOrigin;
using IndexElement = o2::DataFormat::IndexElement;

namespace {
  o2::header::DataDescription lookupDataDescription(const char *key) {
    if (strcmp(key, "RAWDATA") == 0)
      return o2::header::gDataDescriptionRawData;
    else if (strcmp(key, "CLUSTERS") == 0)
      return o2::header::gDataDescriptionClusters;
    else if (strcmp(key, "TRACKS") == 0)
      return o2::header::gDataDescriptionTracks;
    else if (strcmp(key, "CONFIG") == 0)
      return o2::header::gDataDescriptionConfig;
    else if (strcmp(key, "INFO") == 0)
      return o2::header::gDataDescriptionInfo;
    return o2::header::gDataDescriptionInvalid;
  }

  o2::header::DataOrigin lookupDataOrigin(const char *key) {
    if (strcmp(key, "TPC") == 0)
      return o2::header::gDataOriginTPC;
    if (strcmp(key, "TRD") == 0)
      return o2::header::gDataOriginTRD;
    if (strcmp(key, "TOF") == 0)
      return o2::header::gDataOriginTOF;
    if (strcmp(key, "ITS") == 0)
      return o2::header::gDataOriginITS;
    return o2::header::gDataOriginInvalid;
  }


}

namespace o2 { namespace DataFlow {

std::unique_ptr<char[]> fakeTimeframeGenerator(std::vector<FakeTimeframeSpec> &specs, std::size_t &totalSize) {
  // Calculate the total size of your timeframe. This is
  // given by:
  // - N*The size of the data header (this should actually depend on the
  //   kind of data as different dataDescriptions will probably have
  //   different headers).
  // - Sum_N(The size of the buffer_i)
  // - The size of the index header
  // - N*sizeof(dataheader)
  // Assuming all the data header
  size_t sizeOfHeaders = specs.size()*sizeof(DataHeader);
  size_t sizeOfBuffers = 0;
  for (auto && spec : specs) {
    sizeOfBuffers += spec.bufferSize;
  }
  size_t sizeOfIndexHeader = sizeof(DataHeader);
  size_t sizeOfIndex = sizeof(IndexElement)*specs.size();
  totalSize = sizeOfHeaders + sizeOfBuffers + sizeOfIndexHeader + sizeOfIndex;

  // Add the actual - data
  auto buffer = std::make_unique<char[]>(totalSize);
  char *bi = buffer.get();
  std::vector<IndexElement> headers;
  int count = 0;
  for (auto &&spec : specs) {
    IndexElement el;
    el.first.dataDescription = lookupDataDescription(spec.dataDescription);
    el.first.dataOrigin = lookupDataOrigin(spec.origin);
    el.first.payloadSize = spec.bufferSize;
    el.first.headerSize = sizeof(el.first);
    el.second = count++;
    // Let's zero at least the header...
    memset(bi, 0, sizeof(el.first));
    memcpy(bi, &el, sizeof(el.first));
    headers.push_back(el);
    bi += sizeof(el.first);
    spec.bufferFiller(bi, spec.bufferSize);
    bi += spec.bufferSize;
  }

  // Add the index
  DataHeader index;
  index.dataDescription = DataDescription("TIMEFRAMEINDEX");
  index.dataOrigin = DataOrigin("FKE");
  index.headerSize = sizeOfIndexHeader;
  index.payloadSize = sizeOfIndex;
  memcpy(bi, &index, sizeof(index));
  memcpy(bi+sizeof(index), headers.data(), headers.size() * sizeof(IndexElement));
  return std::move(buffer);
}

}} // o2::Headers
