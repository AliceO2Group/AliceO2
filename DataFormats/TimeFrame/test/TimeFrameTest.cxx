// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TimeFrame TimeFrametest
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TimeFrame/TimeFrame.h"
#include <FairMQMessage.h>
#include <FairMQParts.h>
#include <FairMQTransportFactory.h>
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

// some (modified) convenience functions to be able to create messages
// copied from FairRoot
template <typename T>
inline static void SimpleMsgCleanup(void* /*data*/, void* obj)
{
  delete static_cast<T*>(obj);
}

template <typename Factory, typename T>
inline static FairMQMessagePtr NewSimpleMessage(Factory& f, const T& data)
{
  auto* dataCopy = new T(data);
  return f.CreateMessage(dataCopy, sizeof(T), SimpleMsgCleanup<T>, dataCopy);
}

BOOST_AUTO_TEST_CASE(TimeFrame_test)
{
  FairMQParts messages;

  // we use the ZMQ Factory to create some messages
  std::shared_ptr<FairMQTransportFactory> zmq(FairMQTransportFactory::CreateTransportFactory("zeromq"));
  BOOST_CHECK(zmq);
  messages.AddPart(zmq->CreateMessage(1000));

  o2::header::DataHeader dh;
  messages.AddPart(NewSimpleMessage(*zmq, dh));

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
