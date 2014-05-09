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

#include "SystemInterface.h"
#include "WrapperDevice.h"
#include <iostream>
#include <getopt.h>
#include <vector>
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
  static struct option programOptions[] = {
    {"library",     required_argument, 0, 'l'},
    {"component",   required_argument, 0, 'c'},
    {"parameter",   required_argument, 0, 'p'},
    {"run",         required_argument, 0, 'r'},
    {0, 0, 0, 0}
  };

  /* getopt_long stores the option index here. */
  char c=0;
  int iOption = 0;

  // HLT components are implemented in shared libraries, the library name
  // and component id are used to factorize a component
  // optionally, a list of configuration parameters can be specified as
  // one single string which is translated in an array of string in the
  // argc/argv format
  const char* componentLibrary="";
  const char* componentId="";
  const char* componentParameter="";

  // the configuration and calibration is fixed for every run and identified
  // by the run no
  int runNumber=0;

  while ((c=getopt_long(argc, argv, "l:c:p:r:", programOptions, &iOption)) != -1) {
    switch (c) {
    case 'l':
      componentLibrary=optarg;
      break;
    case 'c':
      componentId=optarg;
      break;
    case 'p':
      componentParameter=optarg;
      break;
    case 'r':
      stringstream(optarg) >> runNumber;
    case '?':
      // TODO: more error handling
      break;
    default:
      cerr << "unknown option: '"<< c << "'" << endl;
    }
  }

  cout << "Library: " << componentLibrary << " - " << componentId << " (" << componentParameter << ")" << endl;

  ALICE::HLT::WrapperDevice device(componentLibrary, componentId, componentParameter, runNumber);

#ifdef NANOMSG
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryNN();
#else
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();
#endif

  device.SetTransport(transportFactory);
  device.SetProperty(FairMQDevice::Id, "ID");
  device.SetProperty(FairMQDevice::NumIoThreads, 1);
  device.SetProperty(FairMQDevice::NumInputs, 0);
  device.SetProperty(FairMQDevice::NumOutputs, 1);
  device.ChangeState(FairMQDevice::INIT);
  device.SetProperty(FairMQDevice::OutputSocketType, "push");
  device.SetProperty(FairMQDevice::OutputSndBufSize, 10000, 0);
  device.SetProperty(FairMQDevice::OutputMethod, "connect", 0);
  device.SetProperty(FairMQDevice::OutputAddress, "tcp://localhost:5565", 0);

  device.ChangeState(FairMQDevice::SETOUTPUT);
  device.ChangeState(FairMQDevice::SETINPUT);
  device.ChangeState(FairMQDevice::RUN);

  char ch;
  cin.get(ch);

  device.ChangeState(FairMQDevice::STOP);
  device.ChangeState(FairMQDevice::END);

  return iResult;

  /* !!!! preliminary code below is disabled !!!*/
  AliHLTComponentHandle componentHandle=kEmptyHLTComponentHandle;

  // chop the parameter string in order to provide parameters in the argc/argv format
  vector<const char*> parameters;
  auto_ptr<char> parameterBuffer(new char[strlen(componentParameter)+1]);
  if (strlen(componentParameter)>0 && parameterBuffer.get()!=NULL) {
    strcpy(parameterBuffer.get(), componentParameter);
    char* iterator=parameterBuffer.get();
    parameters.push_back(iterator);
    for (; *iterator!=0; iterator++) {
      if (*iterator!=' ') continue;
      *iterator=0; // separate strings
      if (*(iterator+1)!=' ' && *(iterator+1)!=0)
	parameters.push_back(iterator+1);
    }
  }

  ALICE::HLT::SystemInterface iface;
  if ((iResult=iface.InitSystem(runNumber))<0)
    return iResult;

  if ((iResult=iface.LoadLibrary(componentLibrary))<0)
    return iResult;

  if ((iResult=iface.CreateComponent(componentId, NULL, parameters.size(), &parameters[0], &componentHandle, ""))<0)
    return iResult;

  return 0;
}
