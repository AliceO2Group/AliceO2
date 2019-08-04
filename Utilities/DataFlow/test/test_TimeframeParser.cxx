// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFlow/TimeframeParser.h"
#include "DataFlow/FakeTimeframeBuilder.h"
#include "Headers/DataHeader.h"
#include <FairMQParts.h>
#include <istream>
#include <cstdlib>

struct OneShotReadBuf : public std::streambuf {
  OneShotReadBuf(char* s, std::size_t n)
  {
    setg(s, s, s + n);
  }
};

using DataHeader = o2::header::DataHeader;

int main(int argc, char** argv)
{
  // Construct a dummy timeframe.
  // Stream it and get the parts
  FairMQParts parts;
  auto onAddParts = [](FairMQParts& p, char* buffer, size_t size) {
    LOG(INFO) << "Adding part to those to be sent.\n";
  };
  auto onSend = [](FairMQParts& p) {
    LOG(INFO) << "Everything OK. Sending parts\n";
  };

  // Prepare a test timeframe to be streamed
  auto zeroFiller = [](char* b, size_t s) { memset(b, 0, s); };
  std::vector<o2::data_flow::FakeTimeframeSpec> specs = {
    {.origin = "TPC",
     .dataDescription = "CLUSTERS",
     .bufferFiller = zeroFiller,
     .bufferSize = 1000}};

  size_t testBufferSize;
  auto testBuffer = fakeTimeframeGenerator(specs, testBufferSize);

  OneShotReadBuf osrb(testBuffer.get(), testBufferSize);
  std::istream s(&osrb);

  try {
    o2::data_flow::streamTimeframe(s, onAddParts, onSend);
  } catch (std::runtime_error& e) {
    LOG(ERROR) << e.what() << std::endl;
    exit(1);
  }
}
