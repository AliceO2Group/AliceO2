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
#include <csignal>
#include <getopt.h>
#include <memory>
#include <cstring>
#ifdef NANOMSG
  #include "FairMQTransportFactoryNN.h"
#endif
#include "FairMQTransportFactoryZMQ.h"

using std::cout;
using std::cerr;
using std::stringstream;

  struct SocketProperties_t {
    std::string type;
    int         size;
    std::string method;
    std::string address;
  };

FairMQDevice* gDevice=NULL;
static void s_signal_handler (int signal)
{
  cout << endl << "Caught signal " << signal << endl;

  if (gDevice) {
    gDevice->ChangeState(FairMQDevice::STOP);
    gDevice->ChangeState(FairMQDevice::END);
    cout << "Shutdown complete. Bye!" << endl;
  } else {
    cerr << "No device to shut down, ignoring signal ..." << endl;
  }

  exit(1);
}

static void s_catch_signals (void)
{
  struct sigaction action;
  action.sa_handler = s_signal_handler;
  action.sa_flags = 0;
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, NULL);
  sigaction(SIGQUIT, &action, NULL);
  sigaction(SIGTERM, &action, NULL);
}

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
  const char* factoryType="zmq";
  int verbosity=-1;
  int deviceLogInterval=10000;
  int pollingPeriod=-1;
  int skipProcessing=0;

  static struct option programOptions[] = {
    {"input",       required_argument, 0, 'i'}, // input socket
    {"output",      required_argument, 0, 'o'}, // output socket
    {"factory-type",required_argument, 0, 't'}, // type of the factory "zmq", "nanomsg"
    {"verbosity",   required_argument, 0, 'v'}, // verbosity
    {"loginterval", required_argument, 0, 'l'}, // logging interval
    {"poll-period", required_argument, 0, 'p'}, // polling period of the device in ms
    {"dry-run",     no_argument      , 0, 'n'}, // skip the component processing
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

  // build the string from the option definition
  // hyphen in the beginning indicates custom handling for non-option elements
  // a colon after the option indicates a required argument to that option
  // two colons after the option indicate an optional argument
  std::string optstring="-";
  for (struct option* programOption=programOptions;
       programOption!=NULL && programOption->name!=NULL;
       programOption++) {
    if (programOption->flag==NULL) {
      // programOption->val uniquely identifies particular long option
      optstring+=((char)programOption->val);
      if (programOption->has_arg==required_argument)
	optstring+=":"; // one colon to indicate required argument
      if (programOption->has_arg==optional_argument)
	optstring+="::"; // two colons to indicate optional argument
    } else {
      throw std::runtime_error("handling of program option flag is not yet implemented, please check option definitions");
    }
  }
  while ((c=getopt_long(argc, argv, optstring.c_str(), programOptions, &iOption)) != -1
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
    case 't':
      factoryType=optarg;
      break;
    case 'v':
      std::stringstream(optarg) >> std::hex >> verbosity;
      break;
    case 'l':
      std::stringstream(optarg) >> deviceLogInterval;
      break;
    case 'p':
      std::stringstream(optarg) >> pollingPeriod;
      break;
    case 'n':
      skipProcessing=1;
      break;
    case '?': // all remaining arguments passed to the device instance
      iDeviceArg=optind-1;
      break;
    case '\1': // the first required arguments are without hyphens and in fixed order
      // special treatment of elements not defined in the option string is
      // indicated by the leading hyphen in the option string, that makes
      // getopt to return value '1' for any of those elements allowing
      // for a custom handling, matches only elemments not starting with a hyphen
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
    cout << "Usage : " << argv[0] << " ID numIoThreads [--factory type] [--input|--output type=value,size=value,method=value,address=value] componentArguments" << endl;
    cout << "        The first two arguments are in fixed order, followed by optional arguments: " << endl;
    cout << "        --factory-type,-t nanomsg|zmq" << endl;
    cout << "        --poll-period,-p             period_in_ms" << endl;
    cout << "        --loginterval,-l             period_in_ms" << endl;
    cout << "        --verbosity,-v 0xhexval      verbosity level" << endl;
    cout << "        --dry-run,-n                 skip the component processing" << endl;
    cout << "        Multiple slots can be defined by --input/--output options" << endl;
    cout << "        HLT component arguments at the end of the list" << endl;
    cout << "        --library,-l     componentLibrary" << endl;
    cout << "        --component,-c   componentId"	<< endl;
    cout << "        --parameter,-p   parameter"	<< endl;
    cout << "        --run,-r         runNo"            << endl;

    return 0;
  }

  FairMQTransportFactory* transportFactory=NULL;
  if (strcmp(factoryType, "nanomsg")==0) {
#ifdef NANOMSG
    transportFactory = new FairMQTransportFactoryNN();
#else
    cerr << "can not create factory for NANOMSG: not enabled in build" << endl;
    return -ENODEV;
#endif
  } else if (strcmp(factoryType, "zmq")==0) {
    transportFactory = new FairMQTransportFactoryZMQ();
  } else {
    cerr << "invalid factory type: " << factoryType << endl;
    return -ENODEV;
  }

  if (verbosity>=0) {
    std::ios::fmtflags oldflags = std::cout.flags();
    std::cout << "Verbosity: 0x" << std::setfill('0') << std::setw(2) << std::hex << verbosity << std::endl;
    std::cout.flags(oldflags);
    // verbosity option is propagated to device, the verbosity level
    // for the HLT component code can be specified as extra parameter,
    // note that this is using a mask, where each bit corresponds to
    // a message catagory, e.g. 0x78 is commonly used to get all
    // above "Info"
  }

  vector<char*> deviceArgs;
  deviceArgs.push_back(argv[0]);
  if (iDeviceArg>0)
    deviceArgs.insert(deviceArgs.end(), argv+iDeviceArg, argv+argc);

  gDevice=new ALICE::HLT::WrapperDevice(deviceArgs.size(), &deviceArgs[0], verbosity);
  if (!gDevice) {
    cerr << "failed to create device"  << endl;
    return -ENODEV;
  }
  s_catch_signals();

  { // scope for the device reference variable
  FairMQDevice& device=*gDevice;

  device.SetTransport(transportFactory);
  device.SetProperty(FairMQDevice::Id, id.c_str());
  device.SetProperty(FairMQDevice::NumIoThreads, numIoThreads);
  device.SetProperty(FairMQDevice::NumInputs, numInputs);
  device.SetProperty(FairMQDevice::NumOutputs, numOutputs);
  device.SetProperty(FairMQDevice::LogIntervalInMs, deviceLogInterval);
  if (pollingPeriod>0) device.SetProperty(ALICE::HLT::WrapperDevice::PollingPeriod, pollingPeriod);
  if (skipProcessing)  device.SetProperty(ALICE::HLT::WrapperDevice::SkipProcessing, skipProcessing);
  device.ChangeState(FairMQDevice::INIT);
  for (unsigned iInput=0; iInput<numInputs; iInput++) {
    device.SetProperty(FairMQDevice::InputSocketType, inputSockets[iInput].type.c_str(), iInput);
    // set High-water-mark for the sockets. in ZMQ, depending on the socket type, some
    // have only send buffers (PUB, PUSH), some only receive buffers (SUB, PULL), and
    // some have both (DEALER, ROUTER, PAIR, REQ, REP)
    // we set both snd and rcv to the same value for the moment
    device.SetProperty(FairMQDevice::InputSndBufSize, inputSockets[iInput].size, iInput);
    device.SetProperty(FairMQDevice::InputRcvBufSize, inputSockets[iInput].size, iInput);
    device.SetProperty(FairMQDevice::InputMethod,     inputSockets[iInput].method.c_str(), iInput);
    device.SetProperty(FairMQDevice::InputAddress,    inputSockets[iInput].address.c_str(), iInput);
  }
  for (unsigned iOutput=0; iOutput<numOutputs; iOutput++) {
    device.SetProperty(FairMQDevice::OutputSocketType, outputSockets[iOutput].type.c_str(), iOutput);
    // we set both snd and rcv to the same value for the moment, see above
    device.SetProperty(FairMQDevice::OutputSndBufSize, outputSockets[iOutput].size, iOutput);
    device.SetProperty(FairMQDevice::OutputRcvBufSize, outputSockets[iOutput].size, iOutput);
    device.SetProperty(FairMQDevice::OutputMethod,     outputSockets[iOutput].method.c_str(), iOutput);
    device.SetProperty(FairMQDevice::OutputAddress,    outputSockets[iOutput].address.c_str(), iOutput);
  }

  device.ChangeState(FairMQDevice::SETOUTPUT);
  device.ChangeState(FairMQDevice::SETINPUT);
  device.ChangeState(FairMQDevice::RUN);

  boost::unique_lock<boost::mutex> lock(device.fRunningMutex);
  while (!device.fRunningFinished)
  {
      device.fRunningCondition.wait(lock);
  }
  } // scope for the device reference variable

  FairMQDevice* almostdead=gDevice;
  gDevice=NULL;
  delete almostdead;

  return iResult;
}
