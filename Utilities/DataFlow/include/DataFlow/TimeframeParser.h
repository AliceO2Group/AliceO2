// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef TIMEFRAME_PARSER_H_
#define TIMEFRAME_PARSER_H_

#include <iosfwd>
#include <functional>

class FairMQParts;

namespace o2
{
namespace data_flow
{

/// An helper function which takes a std::istream pointing
/// to a naively persisted timeframe and pumps its parts to
/// FairMQParts, ready to be shipped via FairMQ.
void streamTimeframe(std::istream& stream,
                     std::function<void(FairMQParts& parts, char* buffer, size_t size)> onAddPart,
                     std::function<void(FairMQParts& parts)> onSend);

void streamTimeframe(std::ostream& stream, FairMQParts& parts);

} // namespace data_flow
} // namespace o2

#endif // TIMEFRAME_PARSER_H
