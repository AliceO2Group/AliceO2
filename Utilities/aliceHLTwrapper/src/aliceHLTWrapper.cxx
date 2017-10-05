// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   aliceHLTWrapper.cxx
//  @author Matthias Richter
//  @since  2014-05-07
//  @brief  FairRoot/ALFA device running ALICE HLT code


#include "runFairMQDevice.h" // FairMQDevice launcher boiler plate code
#include "aliceHLTwrapper/WrapperDevice.h"
#include "aliceHLTwrapper/Component.h"

namespace bpo = boost::program_options;

using WorkerProcessor = o2::alice_hlt::Component;

void addCustomOptions(bpo::options_description& options)
{
  options.add(o2::alice_hlt::WrapperDevice::GetOptionsDescription());
  WorkerProcessor component;
  options.add(component.getOptionsDescription());
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  auto componentCreatorFct = [](){return new WorkerProcessor;};
  return new o2::alice_hlt::WrapperDevice(componentCreatorFct);
}
