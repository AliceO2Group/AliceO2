//-*- Mode: C++ -*-

#ifndef WRAPPERDEVICE_H
#define WRAPPERDEVICE_H
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

//  @file   WrapperDevice.h
//  @author Matthias Richter
//  @since  2014-05-08
//  @brief  FairRoot/ALFA device running ALICE HLT code

#include <FairMQDevice.h>
#include <vector>

class FairMQMessage;

namespace ALICE {
namespace HLT {
class Component;

/// @class WrapperDevice
/// A FairMQ device class supporting the HLT component interface
/// for processing.
///
/// The device class implements the interface functions of FairMQ, and it
/// receives and send messages. The data of the messages are processed
/// using the Component class.
class WrapperDevice : public FairMQDevice {
public:
  /// default constructor
  WrapperDevice(int argc, char** argv, int verbosity = 0);
  /// destructor
  ~WrapperDevice() override;

  /////////////////////////////////////////////////////////////////
  // the FairMQDevice interface

  /// inherited from FairMQDevice
  void Init() override;
  /// inherited from FairMQDevice
  void InitTask() override;
  /// inherited from FairMQDevice
  void Run() override;
  /// inherited from FairMQDevice
  void Pause() override;
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::SetProperty
  void SetProperty(const int key, const std::string& value) override;
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::GetProperty
  std::string GetProperty(const int key, const std::string& default_ = "") override;
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::SetProperty
  void SetProperty(const int key, const int value) override;
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::GetProperty
  int GetProperty(const int key, const int default_ = 0) override;

  /////////////////////////////////////////////////////////////////
  // device property identifier
  enum { Id = FairMQDevice::Last, PollingPeriod, SkipProcessing, Last };

protected:

private:
  // copy constructor prohibited
  WrapperDevice(const WrapperDevice&);
  // assignment operator prohibited
  WrapperDevice& operator=(const WrapperDevice&);

  /// create a new message with data buffer of specified size
  unsigned char* createMessageBuffer(unsigned size);

  Component* mComponent;     // component instance
  std::vector<char*> mArgv;       // array of arguments for the component
  std::vector<std::unique_ptr<FairMQMessage>> mMessages; // array of output messages

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

} // namespace hlt
} // namespace alice
#endif // WRAPPERDEVICE_H
