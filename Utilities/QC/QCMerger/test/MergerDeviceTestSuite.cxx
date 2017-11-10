// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE MergerDevice
#define BOOST_TEST_MAIN

#include <memory>

#include <boost/test/unit_test.hpp>

#include "QCMerger/MergerDevice.h"

using namespace std;

namespace
{
const char* INPUT_ADDRESS = "tcp://*:5005";
const char* OUTPUT_ADDREDD = "tcp://login01.pro.cyfronet.pl:5004";
const char* MERGER_DEVICE_ID = "TEST_MERGER";
const int NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA = 2;
int bufferSize = 10;

shared_ptr<MergerDevice> mrgerDevice;
}

BOOST_AUTO_TEST_SUITE(MergerDeviceTestSuite)

BOOST_AUTO_TEST_CASE(createMergerDevice)
{
  unique_ptr<MergerDevice> mrgerDevice(new MergerDevice(
    unique_ptr<Merger>(new Merger(NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA)), MERGER_DEVICE_ID));

  BOOST_CHECK(mrgerDevice->GetId() == MERGER_DEVICE_ID);
}

BOOST_AUTO_TEST_CASE(establishChannelByMergerDevice)
{
  unique_ptr<MergerDevice> mrgerDevice(new MergerDevice(
    unique_ptr<Merger>(new Merger(NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA)), MERGER_DEVICE_ID));

  BOOST_TEST(mrgerDevice->fChannels.size() == 0, "Producer device has a channel connected at startup");

  mrgerDevice->establishChannel("req", "connect", OUTPUT_ADDREDD, "test", bufferSize, bufferSize);
  BOOST_TEST(mrgerDevice->fChannels.size() == 1, "Producer device did not establish channel");
}

BOOST_AUTO_TEST_SUITE_END()
