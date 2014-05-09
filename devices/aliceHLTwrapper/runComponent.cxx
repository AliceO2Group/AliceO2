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

//  @file   runComponent.cxx
//  @author Matthias Richter
//  @since  2014-05-07 
//  @brief  run a component encapsulating ALICE HLT code

// TODO: this file is intended to evolve into a unit test

#include "SystemInterface.h"
#include "Component.h"
#include <iostream>
#include <getopt.h>
#include <vector>
#include <memory>
#include <cstring>

using std::cout;
using std::cerr;
using std::stringstream;

int main(int argc, char** argv)
{
  int iResult=0;

  ALICE::HLT::Component c;
  if ((iResult=c.Init(argc, argv))<0) {
    cerr << "error: init failed with " << iResult << endl;
    // the ALICE HLT external interface uses the following error definition
    // 0 success
    // >0 error number
    return -iResult;
  }

  vector<ALICE::HLT::Component::BufferDesc_t> blockData;
  if ((iResult=c.Process(blockData))<0) {
    cerr << "error: init failed with " << iResult << endl;
    return -iResult;
  }
}
