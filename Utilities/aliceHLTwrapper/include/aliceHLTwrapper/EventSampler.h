//-*- Mode: C++ -*-

#ifndef EVENTSAMPLER_H
#define EVENTSAMPLER_H
//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or	     *
//* (at your option) any later version.					     *
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

#include "FairMQDevice.h"
#include <vector>

namespace ALICE {
namespace HLT {
class Component;

/// @class EventSampler
/// Sampler device for Alice HLT events in FairRoot/ALFA.
///
/// The device sends the event descriptor to downstream devices and can
/// measure latency though a feedback channel
class EventSampler : public FairMQDevice {
public:
  /// default constructor
  EventSampler(int verbosity=0);
  /// destructor
  ~EventSampler();

  /////////////////////////////////////////////////////////////////
  // the FairMQDevice interface

  /// inherited from FairMQDevice
  virtual void Init();
  /// inherited from FairMQDevice
  virtual void Run();
  /// inherited from FairMQDevice
  virtual void Pause();
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::SetProperty
  virtual void SetProperty(const int key, const std::string& value);
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::GetProperty
  virtual std::string GetProperty(const int key, const std::string& default_ = "");
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::SetProperty
  virtual void SetProperty(const int key, const int value);
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::GetProperty
  virtual int GetProperty(const int key, const int default_ = 0);

  /// sampler loop started in a separate thread
  void samplerLoop();

  /////////////////////////////////////////////////////////////////
  // device property identifier
  enum { Id = FairMQDevice::Last, PollingTimeout, SkipProcessing, EventPeriod, InitialDelay, OutputFile, Last };

protected:

private:
  // copy constructor prohibited
  EventSampler(const EventSampler&);
  // assignment operator prohibited
  EventSampler& operator=(const EventSampler&);

  int mEventPeriod;          // event rate in us
  int mInitialDelay;         // initial delay in ms before sending first event
  int mNEvents;              // number of generated events
  int mPollingTimeout;       // period of polling on input sockets in ms
  int mSkipProcessing;       // skip component processing
  int mVerbosity;            // verbosity level
  std::string mOutputFile;   // output file for logging of latency
};

} // namespace hlt
} // namespace alice
#endif // EVENTSAMPLER_H
