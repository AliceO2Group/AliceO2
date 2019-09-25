// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TimeframeValidatorDevice.cxx
/// @author Giulio Eulisse, Matthias Richter, Sandro Wenzel
/// @since  2017-02-07
/// @brief  Validator device for a full time frame

#include <thread> // this_thread::sleep_for
#include <chrono>
#include <cstring>

#include "DataFlow/TimeframeParser.h"
#include "Headers/SubframeMetadata.h"
#include "Headers/DataHeader.h"
#include "TimeFrame/TimeFrame.h"

#include <options/FairMQProgOptions.h>
#include <FairMQParts.h>

using DataHeader = o2::header::DataHeader;
using DataDescription = o2::header::DataDescription;
using IndexElement = o2::dataformats::IndexElement;

namespace o2
{
namespace data_flow
{

// Possible states for the parsing of a timeframe
// PARSE_BEGIN_STREAM ->
enum ParsingState {
  PARSE_BEGIN_STREAM = 0,
  PARSE_BEGIN_TIMEFRAME,
  PARSE_BEGIN_PAIR,
  PARSE_DATA_HEADER,
  PARSE_CONCRETE_HEADER,
  PARSE_PAYLOAD,
  PARSE_END_PAIR,
  PARSE_END_TIMEFRAME,
  PARSE_END_STREAM,
  ERROR
};

struct StreamingState {
  StreamingState() = default;

  ParsingState state = PARSE_BEGIN_STREAM;
  bool hasDataHeader = false;
  bool hasConcreteHeader = false;
  void* payloadBuffer = nullptr;
  void* headerBuffer = nullptr;
  DataHeader dh; // The current DataHeader being parsed
};

void streamTimeframe(std::istream& stream,
                     std::function<void(FairMQParts& parts, char* buffer, size_t size)> onAddPart,
                     std::function<void(FairMQParts& parts)> onSend)
{
  FairMQParts parts;
  StreamingState state;
  assert(state.state == PARSE_BEGIN_STREAM);
  while (true) {
    switch (state.state) {
      case PARSE_BEGIN_STREAM:
        LOG(INFO) << "In PARSE_BEGIN_STREAM\n";
        state.state = PARSE_BEGIN_TIMEFRAME;
        break;
      case PARSE_BEGIN_TIMEFRAME:
        LOG(INFO) << "In PARSE_BEGIN_TIMEFRAME\n";
        state.state = PARSE_BEGIN_PAIR;
        break;
      case PARSE_BEGIN_PAIR:
        LOG(INFO) << "In PARSE_BEGIN_PAIR\n";
        state.state = PARSE_DATA_HEADER;
        state.hasDataHeader = false;
        state.payloadBuffer = nullptr;
        state.headerBuffer = nullptr;
        break;
      case PARSE_DATA_HEADER:
        LOG(INFO) << "In PARSE_DATA_HEADER\n";
        if (state.hasDataHeader) {
          throw std::runtime_error("DataHeader already present.");
        } else if (state.payloadBuffer) {
          throw std::runtime_error("Unexpected payload.");
        }
        LOG(INFO) << "Reading dataheader of " << sizeof(state.dh) << " bytes\n";
        stream.read(reinterpret_cast<char*>(&state.dh), sizeof(state.dh));
        // If we have a TIMEFRAMEINDEX part and we find the eof, we are done.
        if (stream.eof()) {
          throw std::runtime_error("Premature end of stream");
        }

        // Otherwise we move to the state which is responsible for parsing the
        // kind of header.
        state.state = PARSE_CONCRETE_HEADER;
        break;
      case PARSE_CONCRETE_HEADER:
        LOG(INFO) << "In PARSE_CONCRETE_HEADER\n";
        if (state.headerBuffer) {
          throw std::runtime_error("File has two consecutive headers");
        }
        if (state.dh.headerSize < sizeof(DataHeader)) {
          std::ostringstream str;
          str << "Bad header size. Should be greater then "
              << sizeof(DataHeader)
              << ". Found " << state.dh.headerSize << "\n";
          throw std::runtime_error(str.str());
        }
        // We get the full header size and read the rest of the header
        state.headerBuffer = malloc(state.dh.headerSize);
        memcpy(state.headerBuffer, &state.dh, sizeof(state.dh));
        LOG(INFO) << "Reading rest of the header of " << state.dh.headerSize - sizeof(state.dh) << " bytes\n";
        stream.read(reinterpret_cast<char*>(state.headerBuffer) + sizeof(state.dh),
                    state.dh.headerSize - sizeof(state.dh));
        // Handle the case the file was truncated.
        if (stream.eof()) {
          throw std::runtime_error("Unexpected end of file");
        }
        onAddPart(parts, reinterpret_cast<char*>(state.headerBuffer), state.dh.headerSize);
        // Move to parse the payload
        state.state = PARSE_PAYLOAD;
        break;
      case PARSE_PAYLOAD:
        LOG(INFO) << "In PARSE_PAYLOAD\n";
        if (state.payloadBuffer) {
          throw std::runtime_error("File has two consecutive payloads");
        }
        state.payloadBuffer = new char[state.dh.payloadSize];
        LOG(INFO) << "Reading payload of " << state.dh.payloadSize << " bytes\n";
        stream.read(reinterpret_cast<char*>(state.payloadBuffer), state.dh.payloadSize);
        if (stream.eof()) {
          throw std::runtime_error("Unexpected end of file");
        }
        onAddPart(parts, reinterpret_cast<char*>(state.payloadBuffer), state.dh.payloadSize);
        state.state = PARSE_END_PAIR;
        break;
      case PARSE_END_PAIR:
        LOG(INFO) << "In PARSE_END_PAIR\n";
        state.state = state.dh == DataDescription("TIMEFRAMEINDEX") ? PARSE_END_TIMEFRAME : PARSE_BEGIN_PAIR;
        break;
      case PARSE_END_TIMEFRAME:
        LOG(INFO) << "In PARSE_END_TIMEFRAME\n";
        onSend(parts);
        // Check if we have more. If not, we can declare success.
        stream.peek();
        if (stream.eof()) {
          state.state = PARSE_END_STREAM;
        } else {
          state.state = PARSE_BEGIN_TIMEFRAME;
        }
        break;
      case PARSE_END_STREAM:
        return;
        break;
      default:
        break;
    }
  }
}

void streamTimeframe(std::ostream& stream, FairMQParts& parts)
{
  if (parts.Size() < 2) {
    throw std::runtime_error("Expecting at least 2 parts\n");
  }

  auto indexHeader = o2::header::get<DataHeader*>(parts.At(parts.Size() - 2)->GetData());
  // FIXME: Provide iterator pair API for the index
  //        Index should really be something which provides an
  //        iterator pair API so that we can sort / find / lower_bound
  //        easily. Right now we simply use it a C-style array.
  auto index = reinterpret_cast<IndexElement*>(parts.At(parts.Size() - 1)->GetData());

  LOG(INFO) << "This time frame has " << parts.Size() << " parts.\n";
  auto indexEntries = indexHeader->payloadSize / sizeof(IndexElement);
  if (indexHeader->dataDescription != DataDescription("TIMEFRAMEINDEX")) {
    throw std::runtime_error("Could not find a valid index header\n");
  }
  LOG(INFO) << indexHeader->dataDescription.str << "\n";
  LOG(INFO) << "This time frame has " << indexEntries << "entries in the index.\n";
  if ((indexEntries * 2 + 2) != (parts.Size())) {
    std::stringstream err;
    err << "Mismatched index and received parts. Expected "
        << (parts.Size() - 2 * 2) << " found " << indexEntries;
    throw std::runtime_error(err.str());
  }

  LOG(INFO) << "Everything is fine with received timeframe\n";
  for (size_t i = 0; i < parts.Size(); ++i) {
    stream.write(reinterpret_cast<const char*>(parts.At(i)->GetData()),
                 parts.At(i)->GetSize());
  }
}

} // namespace data_flow
} // namespace o2
