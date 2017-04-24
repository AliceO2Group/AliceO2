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


#include "runFairMQDevice.h" // FairMQDevice launcher boiler plate code
#include "aliceHLTwrapper/WrapperDevice.h"
using namespace ALICE::HLT;

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add(WrapperDevice::GetOptionsDescription());
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new ALICE::HLT::WrapperDevice;
}
