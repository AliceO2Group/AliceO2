/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

/// @file FairMQmonitor.h
///
/// @since 2014-12-10
/// @author M. Krzewicki <mkrzewic@cern.ch>

#ifndef FAIRMQMONITOR_H_
#define FAIRMQMONITOR_H_

#include "O2device/O2device.h"
#include "Headers/DataHeader.h"

/// This is a simple FairMQ monitoring class
/// assumption is the messages are O@ messages (constructed supported
/// methods in O2device).
/// The "data" channel can be configured as any socket type,
/// it will appropriately send requests, send/receive messages or send replies.
/// All incoming traffic is dumped on screen in the form of a hex dump for both
/// the header block and payload block.
class FairMQmonitor : public AliceO2::Base::O2device
{
public:
  FairMQmonitor();
  virtual ~FairMQmonitor() {};

protected:
  virtual void Run();
  void InitTask();
  bool HandleData(AliceO2::Base::O2message& parts, int index);
  bool HandleO2frame(const byte* headerBuffer, size_t headerBufferSize,
      const byte* dataBuffer,   size_t dataBufferSize);

private:
  AliceO2::Header::DataHeader mDataHeader;
  std::string mPayload;
  std::string mName;
  long long mDelay;
  long long mIterations;
  long long mLimitOutputCharacters;
};

#endif /* FAIRMQMONITOR_H_ */
