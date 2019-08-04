// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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

/// @file O2MessageMonitor.h
///
/// @since 2014-12-10
/// @author M. Krzewicki <mkrzewic@cern.ch>

#ifndef O2MESSAGEMONITOR_H_
#define O2MESSAGEMONITOR_H_

#include "O2Device/O2Device.h"

/// This is a simple FairMQ monitoring class
/// assumption is the messages are O@ messages (constructed supported
/// methods in O2Device).
/// The "data" channel can be configured as any socket type,
/// it will appropriately send requests, send/receive messages or send replies.
/// All incoming traffic is dumped on screen in the form of a hex dump for both
/// the header block and payload block.
class O2MessageMonitor : public o2::base::O2Device
{
 public:
  O2MessageMonitor() = default;
  ~O2MessageMonitor() override = default;

 protected:
  void Run() override;
  void InitTask() override;

 private:
  std::string mPayload{"I am the info payload"};
  std::string mName{R"(My name is "gDataDescriptionInfo")"};
  long long mDelay{1000};
  long long mIterations{10};
  long long mLimitOutputCharacters{1024};
};

#endif /* O2MESSAGEMONITOR_H_ */
