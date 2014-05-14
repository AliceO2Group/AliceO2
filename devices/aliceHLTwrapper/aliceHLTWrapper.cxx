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

//  @file   aliHLTWrapper.cxx
//  @author Matthias Richter
//  @since  2014-05-07 
//  @brief  FairRoot/ALFA device running ALICE HLT code

#include "WrapperDevice.h"
#include <iostream>
#include <getopt.h>
#include <memory>
#include <cstring>
#ifdef NANOMSG
  #include "FairMQTransportFactoryNN.h"
#else
  #include "FairMQTransportFactoryZMQ.h"
#endif

using std::cout;
using std::cerr;
using std::stringstream;

int main(int argc, char** argv)
{
  int iResult=0;

  // parse options
  int iArg=1;
  bool bPrintUsage=true;
  std::string id;
  int numIoThreads=0;
  int numInputs=0;
  int numOutputs=0;
  std::string socketType;
  int outputBufferSize=0;
  std::string connectMethod;
  std::string address;
  for (; iArg<argc; iArg++) {
    char* arg=argv[iArg];
    switch (iArg) {
    case 1: id=arg; break;
    case 2: std::stringstream(arg) >> numIoThreads; break;
    case 3: std::stringstream(arg) >> numInputs; break;
    case 4: std::stringstream(arg) >> numOutputs; break;
    case 5: socketType=arg; break;
    case 6: std::stringstream(arg) >> outputBufferSize; break;
    case 7: connectMethod=arg; break;
    case 8: address=arg; break;
    default:
      bPrintUsage=false;
      break;
    }
    if (arg[0]=='-') {
      // all options after the first one starting with '-' are propagated
      // to the HLT component
      break;
    }
  }

  if (bPrintUsage) {
    cout << "Usage : " << argv[0] << " ID numIoThreads numInputs numOutputs socketType bufferSize connectMethod address componentArguments" << endl;
    return 0;
  }

  vector<char*> deviceArgs;
  deviceArgs.push_back(argv[0]);
  deviceArgs.insert(deviceArgs.end(), argv+iArg, argv+argc);
  ALICE::HLT::WrapperDevice device(deviceArgs.size(), &deviceArgs[0]);

#ifdef NANOMSG
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryNN();
#else
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();
#endif

  device.SetTransport(transportFactory);
  device.SetProperty(FairMQDevice::Id, id.c_str());
  device.SetProperty(FairMQDevice::NumIoThreads, numIoThreads);
  device.SetProperty(FairMQDevice::NumInputs, numInputs);
  device.SetProperty(FairMQDevice::NumOutputs, numOutputs);
  device.ChangeState(FairMQDevice::INIT);
  device.SetProperty(FairMQDevice::OutputSocketType, socketType.c_str());
  device.SetProperty(FairMQDevice::OutputSndBufSize, outputBufferSize, 0);
  device.SetProperty(FairMQDevice::OutputMethod, connectMethod.c_str(), 0);
  device.SetProperty(FairMQDevice::OutputAddress, address.c_str(), 0);

  device.ChangeState(FairMQDevice::SETOUTPUT);
  device.ChangeState(FairMQDevice::SETINPUT);
  device.ChangeState(FairMQDevice::RUN);

  char ch;
  cin.get(ch);

  device.ChangeState(FairMQDevice::STOP);
  device.ChangeState(FairMQDevice::END);

  return iResult;
}
