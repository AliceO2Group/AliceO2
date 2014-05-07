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

int main(int argc, char** argv)
{
  ALICE::HLT::SystemInterface iface;
  iface.InitSystem();
  return 0;
}
