// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef WRAPPERDEVICE_H
#define WRAPPERDEVICE_H
//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   WrapperDevice.h
//  @author Matthias Richter
//  @since  2014-05-08
//  @brief  FairRoot/ALFA device running ALICE HLT code

#include <FairMQDevice.h>
#include <vector>
#include <boost/program_options.hpp>

namespace bpo = boost::program_options;

class FairMQMessage;

namespace o2
{
namespace alice_hlt
{
class Component;

/// @class WrapperDevice
/// A FairMQ device class supporting the HLT component interface
/// for processing.
///
/// The device class implements the interface functions of FairMQ, and it
/// receives and send messages. The data of the messages are processed
/// using the Component class.
class WrapperDevice : public FairMQDevice
{
 public:
  /// default constructor
  WrapperDevice(int verbosity = 0);
  /// destructor
  ~WrapperDevice() override;

  /// get description of options
  static bpo::options_description GetOptionsDescription();

  enum /*class*/ OptionKeyIds /*: int*/ {
    OptionKeyPollPeriod = 0,
    OptionKeyDryRun,
    OptionKeyLast
  };

  constexpr static const char* OptionKeys[] = {
    "poll-period",
    "dry-run",
    nullptr};

  /////////////////////////////////////////////////////////////////
  // the FairMQDevice interface

  /// inherited from FairMQDevice
  void InitTask() override;
  /// inherited from FairMQDevice
  void Run() override;

 protected:
 private:
  // copy constructor prohibited
  WrapperDevice(const WrapperDevice&);
  // assignment operator prohibited
  WrapperDevice& operator=(const WrapperDevice&);

  /// create a new message with data buffer of specified size
  unsigned char* createMessageBuffer(unsigned size);

  Component* mComponent;                   // component instance
  std::vector<FairMQMessagePtr> mMessages; // array of output messages

  int mPollingPeriod;        // period of polling on input sockets in ms
  int mSkipProcessing;       // skip component processing
  int mLastCalcTime;         // start time of current statistic period
  int mLastSampleTime;       // time of last data sample
  int mMinTimeBetweenSample; // min time between data samples in statistic period
  int mMaxTimeBetweenSample; // max time between data samples in statistic period
  int mTotalReadCycles;      // tot number of read cycles in statistic period
  int mMaxReadCycles;        // max number of read cycles in statistic period
  int mNSamples;             // number of samples in statistic period
  int mVerbosity;            // verbosity level
};

} // namespace alice_hlt
} // namespace o2
#endif // WRAPPERDEVICE_H
