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

  struct SocketProperties_t {
    std::string type;
    int         size;
    std::string method;
    std::string address;
  };

int main(int argc, char** argv)
{
  int iResult=0;

  // parse options
  int iArg=0;
  int iDeviceArg=-1;
  bool bPrintUsage=false;
  std::string id;
  int numIoThreads=0;
  int numInputs=0;
  int numOutputs=0;

  vector<SocketProperties_t> inputSockets;
  vector<SocketProperties_t> outputSockets;

  static struct option programOptions[] = {
    {"input",       required_argument, 0, 'i'},
    {"output",      required_argument, 0, 'o'},
    {0, 0, 0, 0}
  };

  enum socketkeyids{
    TYPE = 0,
    SIZE,
    METHOD,
    ADDRESS,
    lastsocketkey
  };

  const char *socketkeys[] = {
    /*[TYPE]    = */ "type",
    /*[SIZE]    = */ "size",
    /*[METHOD]  = */ "method",
    /*[ADDRESS] = */ "address",
    NULL
  };

  char c=0;
  int iOption = 0;
  opterr=false;
  optind=1; // indicate new start of scanning
  while ((c=getopt_long(argc, argv, "-i:o:", programOptions, &iOption)) != -1
	 && bPrintUsage==false
	 && iDeviceArg<0) {
    switch (c) {
    case 'i':
    case 'o':
      {
	char* subopts = optarg;
	char* value=NULL;
	int keynum=0;
	SocketProperties_t prop;
	while (subopts && *subopts!=0 && *subopts != ' ') {
	  char *saved = subopts;
	  switch(getsubopt(&subopts, (char **)socketkeys, &value)) {
	  case TYPE:    keynum++; prop.type=value;                       break;
	  case SIZE:    keynum++; std::stringstream(value) >> prop.size; break;
	  case METHOD:  keynum++; prop.method=value;                     break;
	  case ADDRESS: keynum++; prop.address=value;                    break;
	  default:
	    keynum=0;
	    break;
	  }
	}
	if (bPrintUsage=(keynum<lastsocketkey)) {
	  cerr << "invalid socket description format: required 'type=value,size=value,method=value,address=value'" << endl;
	} else {
	  if (c=='i') inputSockets.push_back(prop);
	  else outputSockets.push_back(prop);
	}
      }
      break;
    case '?': // all remaining arguments passed to the device instance
      iDeviceArg=optind-1;
      break;
    case '\1': // the first required arguments are without hyphens and in fixed order
      switch (++iArg) {
      case 1: id=optarg; break;
      case 2: std::stringstream(optarg) >> numIoThreads; break;
      default:
	bPrintUsage=true;
      }
      break;
    default:
      cerr << "unknown option: '"<< c << "'" << endl;
    }
  }

  numInputs=inputSockets.size();
  numOutputs=outputSockets.size();
  if (bPrintUsage || iDeviceArg<0 ||
      (numInputs==0 && numOutputs==0)) {
    cout << endl << argv[0] << ":" << endl;
    cout << "        wrapper to run an ALICE HLT component in a FairRoot/ALFA device" << endl;
    cout << "Usage : " << argv[0] << " ID numIoThreads [--input|--output type=value,size=value,method=value,address=value] componentArguments" << endl;
    cout << "        Multiple slots can be defined by --input/--output options" << endl;
    return 0;
  }

  vector<char*> deviceArgs;
  deviceArgs.push_back(argv[0]);
  if (iDeviceArg>0)
    deviceArgs.insert(deviceArgs.end(), argv+iDeviceArg, argv+argc);
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
  for (unsigned iInput=0; iInput<numInputs; iInput++) {
    device.SetProperty(FairMQDevice::InputSocketType, inputSockets[iInput].type.c_str(), iInput);
    device.SetProperty(FairMQDevice::InputSndBufSize, inputSockets[iInput].size, iInput);
    device.SetProperty(FairMQDevice::InputMethod,     inputSockets[iInput].method.c_str(), iInput);
    device.SetProperty(FairMQDevice::InputAddress,    inputSockets[iInput].address.c_str(), iInput);
  }
  for (unsigned iOutput=0; iOutput<numOutputs; iOutput++) {
    device.SetProperty(FairMQDevice::OutputSocketType, outputSockets[iOutput].type.c_str(), iOutput);
    device.SetProperty(FairMQDevice::OutputSndBufSize, outputSockets[iOutput].size, iOutput);
    device.SetProperty(FairMQDevice::OutputMethod,     outputSockets[iOutput].method.c_str(), iOutput);
    device.SetProperty(FairMQDevice::OutputAddress,    outputSockets[iOutput].address.c_str(), iOutput);
  }

  device.ChangeState(FairMQDevice::SETOUTPUT);
  device.ChangeState(FairMQDevice::SETINPUT);
  device.ChangeState(FairMQDevice::RUN);

  char ch;
  cin.get(ch);

  device.ChangeState(FairMQDevice::STOP);
  device.ChangeState(FairMQDevice::END);

  return iResult;
}
