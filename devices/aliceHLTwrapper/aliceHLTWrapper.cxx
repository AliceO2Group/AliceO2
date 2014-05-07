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
#include <iostream>
#include <getopt.h>
#include <vector>
#include <memory>
#include <cstring>

using std::cout;
using std::cerr;

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
  const char* componentLibrary="";
  const char* componentId="";
  const char* componentParameter="";  
  int runNumber=0;

  while ((c=getopt_long(argc, argv, "l:c:p:", programOptions, &iOption)) != -1) {
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
    case '?':
      // TODO: more error handling
      break;
    default:
      cerr << "unknown option: '"<< c << "'" << endl;
    }
  }

  cout << "Library: " << componentLibrary << " - " << componentId << " (" << componentParameter << ")" << endl;

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
