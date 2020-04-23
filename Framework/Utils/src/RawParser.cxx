// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RawParser.cxx
/// @author Matthias Richter
/// @since  2019-10-15
/// @brief  Generic parser for consecutive raw pages

#include "DPLUtils/RawParser.h"
#include "fmt/format.h"
#include <iostream>

namespace o2::framework::raw_parser
{

const char* RDHFormatter<V6>::sFormatString = "{:>5} {:>4} {:>4} {:>4} {:>4} {:>3} {:>3} {:>3}  {:>1}";
void RDHFormatter<V6>::apply(std::ostream& os, V6 const& header, FormatSpec choice, const char* delimiter)
{
  if (choice == FormatSpec::Info) {
    os << "RDH v6";
  } else if (choice == FormatSpec::TableHeader) {
    os << fmt::format(sFormatString, "PkC", "pCnt", "fId", "sId", "Mem", "CRU", "EP", "LID", "s");
  } else if (choice == FormatSpec::Entry) {
    os << fmt::format(sFormatString,
                      header.packetCounter,
                      header.pageCnt,
                      header.feeId,
                      header.sourceID,
                      header.memorySize,
                      header.cruID,
                      header.endPointID,
                      header.linkID,
                      header.stop);
  }
  os << delimiter;
}

const char* RDHFormatter<V5>::sFormatString = "{:>5} {:>4} {:>4} {:>4} {:>3} {:>3} {:>3}  {:>1}";
void RDHFormatter<V5>::apply(std::ostream& os, V5 const& header, FormatSpec choice, const char* delimiter)
{
  if (choice == FormatSpec::Info) {
    os << "RDH v5";
  } else if (choice == FormatSpec::TableHeader) {
    os << fmt::format(sFormatString, "PkC", "pCnt", "fId", "Mem", "CRU", "EP", "LID", "s");
  } else if (choice == FormatSpec::Entry) {
    os << fmt::format(sFormatString,
                      header.packetCounter,
                      header.pageCnt,
                      header.feeId,
                      header.memorySize,
                      header.cruID,
                      header.endPointID,
                      header.linkID,
                      header.stop);
  }
  os << delimiter;
}

const char* RDHFormatter<V4>::sFormatString = "{:>5} {:>4} {:>4} {:>4} {:>3} {:>3} {:>3} {:>10} {:>5}  {:>1}";
void RDHFormatter<V4>::apply(std::ostream& os, V4 const& header, FormatSpec choice, const char* delimiter)
{
  if (choice == FormatSpec::Info) {
    os << "RDH v4";
  } else if (choice == FormatSpec::TableHeader) {
    os << fmt::format(sFormatString, "PkC", "pCnt", "fId", "Mem", "CRU", "EP", "LID", "HBOrbit", "HBBC", "s");
  } else if (choice == FormatSpec::Entry) {
    os << fmt::format(sFormatString,
                      header.packetCounter,
                      header.pageCnt,
                      header.feeId,
                      header.memorySize,
                      header.cruID,
                      header.endPointID,
                      header.linkID,
                      header.heartbeatOrbit,
                      header.heartbeatBC,
                      header.stop);
  }
  os << delimiter;
}

} // namespace o2::framework::raw_parser
