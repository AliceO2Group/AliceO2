// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SimpleRawReader.h
/// \brief Definition of the simple reader for non-DPL tests
#ifndef ALICEO2_ITSMFT_SIMPLERAWREADER_H_
#define ALICEO2_ITSMFT_SIMPLERAWREADER_H_

#include <vector>
#include <gsl/gsl>
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RawFileReader.h"

namespace o2
{

namespace itsmft
{

/// Simple reader for non-DPL tests
struct SimpleRawReader {                                   // simple class to read detector raw data for multiple links
  using LinkBuffer = std::pair<std::vector<char>, size_t>; // buffer for the link TF data and running position index
  using RDH = o2::header::RAWDataHeader;

  std::unique_ptr<o2::raw::RawFileReader> reader;
  std::vector<LinkBuffer> buffers;
  std::string cfgName{};

  SimpleRawReader() = default;
  SimpleRawReader(const std::string& cfg) : cfgName(cfg) {}
  void init();
  int loadNextTF();
  int getNLinks() const { return reader ? reader->getNLinks() : 0; }

  const gsl::span<char> getNextPage(int il);
  bool initDone = false;

  ClassDefNV(SimpleRawReader, 1);
};

} // namespace itsmft
} // namespace o2

#endif
