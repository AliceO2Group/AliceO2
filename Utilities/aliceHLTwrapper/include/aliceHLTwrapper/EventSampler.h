// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef EVENTSAMPLER_H
#define EVENTSAMPLER_H
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

//  @file   EventSampler.h
//  @author Matthias Richter
//  @since  2015-03-15
//  @brief  Sampler device for Alice HLT events in FairRoot/ALFA

#include <FairMQDevice.h>
#include <vector>
#include <boost/program_options.hpp>

namespace bpo = boost::program_options;

namespace o2
{
namespace alice_hlt
{
class Component;

/// @class EventSampler
/// Sampler device for Alice HLT events in FairRoot/ALFA.
///
/// The device sends the event descriptor to downstream devices and can
/// measure latency though a feedback channel
class EventSampler : public FairMQDevice
{
 public:
  /// default constructor
  EventSampler(int verbosity = 0);
  /// destructor
  ~EventSampler() override;

  /// get description of options
  static bpo::options_description GetOptionsDescription();

  enum /*class*/ OptionKeyIds /*: int*/ {
    OptionKeyEventPeriod = 0,
    OptionKeyInitialDelay,
    OptionKeyPollTimeout,
    OptionKeyDryRun,
    OptionKeyLatencyLog,
    OptionKeyLast
  };

  constexpr static const char* OptionKeys[] = {
    "eventperiod",
    "initialdelay",
    "polltimeout",
    "dry-run",
    "latency-log",
    nullptr};

  /////////////////////////////////////////////////////////////////
  // the FairMQDevice interface

  /// inherited from FairMQDevice
  void InitTask() override;
  /// inherited from FairMQDevice
  void Run() override;

  /// sampler loop started in a separate thread
  void samplerLoop();

 protected:
 private:
  // copy constructor prohibited
  EventSampler(const EventSampler&);
  // assignment operator prohibited
  EventSampler& operator=(const EventSampler&);

  int mEventPeriod;                // event rate in us
  int mInitialDelay;               // initial delay in ms before sending first event
  int mNEvents;                    // number of generated events
  int mPollingTimeout;             // period of polling on input sockets in ms
  int mSkipProcessing;             // skip component processing
  int mVerbosity;                  // verbosity level
  std::string mLatencyLogFileName; // output file for logging of latency
};

} // namespace alice_hlt
} // namespace o2
#endif // EVENTSAMPLER_H
