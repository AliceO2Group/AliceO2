// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TimeFrame TimeFrametest
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TimeFrame/TimeFrame.h"
#include <fairmq/Message.h>
#include <fairmq/Parts.h>
#include <fairmq/TransportFactory.h>
#include "Headers/DataHeader.h"
#include "TFile.h"

namespace o2
{
namespace dataformats
{
BOOST_AUTO_TEST_CASE(MessageSizePair_test)
{
  size_t S = 100;
  auto* buffer = new char[S];

  // fill the buffer with something
  buffer[0] = 'a';
  buffer[S - 1] = 'z';

  MessageSizePair mes(S, buffer);

  // assert some properties on MessageSizePair
  BOOST_CHECK(mes.size == S);
  BOOST_CHECK(mes.buffer = buffer);

  // check ROOT IO of MessageSizePair
  TFile* file = TFile::Open("mspair.root", "RECREATE");
  if (file) {
    file->WriteObject(&mes, "MessageSizePair");
    file->Close();
  }

  MessageSizePair* ptr = nullptr;
  TFile* infile = TFile::Open("mspair.root", "READ");
  if (infile) {
    infile->GetObject("MessageSizePair", ptr);
    infile->Close();
    BOOST_CHECK(ptr != nullptr);
  }
  BOOST_CHECK(ptr->size == S);
  // test first and last characters in buffer to see if correctly streamed
  BOOST_CHECK(ptr->buffer[0] == buffer[0]);
  BOOST_CHECK(ptr->buffer[S - 1] == buffer[S - 1]);
}

BOOST_AUTO_TEST_CASE(TimeFrame_test)
{
  fair::mq::Parts messages;

  // we use the ZMQ Factory to create some messages
  auto zmq(fair::mq::TransportFactory::CreateTransportFactory("zeromq"));
  BOOST_CHECK(zmq);
  messages.AddPart(zmq->CreateMessage(1000));

  o2::header::DataHeader dh;
  messages.AddPart(zmq->NewSimpleMessage(dh));

  TimeFrame frame(messages);
  BOOST_CHECK(frame.GetNumParts() == 2);

  // test ROOT IO
  TFile* file = TFile::Open("timeframe.root", "RECREATE");
  if (file) {
    file->WriteObject(&frame, "Timeframe");
    file->Close();
  }

  TimeFrame* ptr = nullptr;
  TFile* infile = TFile::Open("timeframe.root", "READ");
  if (infile) {
    infile->GetObject("Timeframe", ptr);
    infile->Close();
    BOOST_CHECK(ptr != nullptr);
  }

  // test access to data
  BOOST_CHECK(ptr->GetNumParts() == 2);
  for (int i = 0; i < ptr->GetNumParts(); ++i) {
    BOOST_CHECK(ptr->GetPart(i).size == messages[i].GetSize());
  }
}

} // namespace dataformats
} // namespace o2
