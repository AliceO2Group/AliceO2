// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TimeFrameMetadata
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include "Headers/DataHeader.h"
#include "Headers/TimeFrameMetadata.h"

using TimeFrameMetadata = o2::header::TimeFrameMetadata;
using TimeFrameMetadataT = o2::header::TimeFrameMetadataT;
using TimeFrameMetadataBuilder = o2::header::TimeFrameMetadataBuilder;

BOOST_AUTO_TEST_CASE(test_timeframemetadata)
{
  flatbuffers::FlatBufferBuilder builder(512);

  TimeFrameMetadataBuilder tfm_builder(builder);
  tfm_builder.add_runNumber(5);
  tfm_builder.add_firstTfOrbit(128);
  tfm_builder.add_tfID(6);
  auto meta_off = tfm_builder.Finish();

  // !!! finish the buffer before accessing the serialized data
  builder.Finish(meta_off);

  // get data
  void *ptr = builder.GetBufferPointer();
  // get size
  size_t size = builder.GetSize();

  // fake a message
  auto buffer = std::make_unique<uint8_t[]>(size);
  std::memcpy(buffer.get(), ptr, size);
  // clear the original data
  builder.Clear();

  std::cout << "Buffer size: " << size << std::endl;


  {
    // recreate the object
    const TimeFrameMetadata *rec_tfm = o2::header::GetTimeFrameMetadata(buffer.get());

    // check the fields
    flatbuffers::Verifier tfm_verifier(buffer.get(), size);

    BOOST_CHECK( o2::header::VerifyTimeFrameMetadataBuffer(tfm_verifier) );
    BOOST_CHECK(rec_tfm->runNumber() == 5);
    BOOST_CHECK(rec_tfm->tfID() == 6);
    BOOST_CHECK(rec_tfm->firstTfOrbit() == 128);
    BOOST_CHECK(rec_tfm->firstTfBc() == 0); // default value
    BOOST_CHECK(rec_tfm->tfSource() == o2::header::TimeFrameSource::TF); // default value
  }
}


BOOST_AUTO_TEST_CASE(test_timeframemetadataT)
{
  TimeFrameMetadataT tfmd;
  tfmd.runNumber = 1;
  tfmd.tfSource = o2::header::TimeFrameSource::CTF;

  flatbuffers::FlatBufferBuilder builder(512);
  auto meta_off = o2::header::CreateTimeFrameMetadata(builder, &tfmd);
  // !!! finish the buffer before accessing the serialized data
  builder.Finish(meta_off);

  // get data
  void *ptr = builder.GetBufferPointer();
  // get size
  size_t size = builder.GetSize();

  // fake a message
  auto buffer = std::make_unique<uint8_t[]>(size);
  std::memcpy(buffer.get(), ptr, size);
  // clear the original data
  builder.Clear();

  std::cout << "Buffer size: " << size << std::endl;

  {
    TimeFrameMetadataT rcvd_tfmd;
    o2::header::GetTimeFrameMetadata(buffer.get())->UnPackTo(&rcvd_tfmd);

    BOOST_CHECK(rcvd_tfmd.runNumber == 1);
    BOOST_CHECK(rcvd_tfmd.firstTfBc == 0); // default value
    BOOST_CHECK(rcvd_tfmd.tfSource == o2::header::TimeFrameSource::CTF);
  }
}
