// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TH1.h>
#include <gsl/span>
#define BOOST_TEST_MODULE Test Utilities MergerObjectStore
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Mergers/ObjectStore.h"
#include "Mergers/CustomMergeableObject.h"
#include "Mergers/CustomMergeableTObject.h"
#include "Headers/DataHeader.h"
#include "Framework/DataRef.h"

#include <TMessage.h>
#include <boost/test/unit_test.hpp>

#include <memory>

using namespace o2::framework;
using namespace o2::mergers;

// Simple test to do root deserialization.
BOOST_AUTO_TEST_SUITE(TestObjectExtraction)

template <typename TypeToDataRef>
DataRef makeDataRef(TypeToDataRef* obj)
{
  DataRef ref;
  TMessage* tm = new TMessage(kMESS_OBJECT);
  tm->WriteObject(obj);

  ref.payload = tm->Buffer();

  auto dh = new o2::header::DataHeader{};
  dh->payloadSerializationMethod = o2::header::gSerializationMethodROOT;
  dh->payloadSize = tm->BufferSize();
  ref.header = reinterpret_cast<char const*>(dh->data());

  return ref;
}

BOOST_AUTO_TEST_CASE(MergeableObject)
{
  auto* obj = new CustomMergeableTObject("obj1", 123);
  DataRef ref = makeDataRef(obj);

  auto objStore = object_store_helpers::extractObjectFrom(ref);
  BOOST_REQUIRE(std::holds_alternative<MergeInterfacePtr>(objStore));

  auto objExtractedCustom = dynamic_cast<CustomMergeableTObject*>(std::get<MergeInterfacePtr>(objStore).get());
  BOOST_REQUIRE(objExtractedCustom != nullptr);
  BOOST_CHECK_EQUAL(objExtractedCustom->getSecret(), 123);

  delete ref.header;
  delete ref.payload;
}

BOOST_AUTO_TEST_CASE(NamedMergeableObject)
{
  auto* obj = new CustomMergeableObject(123);
  DataRef ref = makeDataRef(obj);

  auto objStore = object_store_helpers::extractObjectFrom(ref);
  BOOST_REQUIRE(std::holds_alternative<MergeInterfacePtr>(objStore));

  auto objExtractedCustom = dynamic_cast<CustomMergeableObject*>(std::get<MergeInterfacePtr>(objStore).get());
  BOOST_REQUIRE(objExtractedCustom != nullptr);
  BOOST_CHECK_EQUAL(objExtractedCustom->getSecret(), 123);

  delete ref.header;
  delete ref.payload;
}

BOOST_AUTO_TEST_CASE(Histo1D)
{
  TH1I* obj = new TH1I("histo", "histo", 100, 0, 100);
  obj->Fill(4);

  DataRef ref = makeDataRef(obj);

  auto objStore = object_store_helpers::extractObjectFrom(ref);
  BOOST_CHECK(std::holds_alternative<TObjectPtr>(objStore));

  auto objExtractedHisto = dynamic_cast<TH1I*>(std::get<TObjectPtr>(objStore).get());
  BOOST_REQUIRE(objExtractedHisto != nullptr);
  BOOST_CHECK_EQUAL(objExtractedHisto->GetEntries(), 1);

  delete ref.header;
  delete ref.payload;
  delete obj;
}

BOOST_AUTO_TEST_CASE(TArrayOfHisto1D)
{
  TObjArray* array = new TObjArray();
  array->SetOwner(true);

  TH1I* histo = new TH1I("histo 1d", "histo 1d", 100, 0, 100);
  histo->Fill(5);
  array->Add(histo);

  DataRef ref = makeDataRef(array);
  auto objStore = object_store_helpers::extractObjectFrom(ref);
  BOOST_CHECK(std::holds_alternative<TObjectPtr>(objStore));

  auto objExtractedArray = dynamic_cast<TObjArray*>(std::get<TObjectPtr>(objStore).get());
  BOOST_REQUIRE(objExtractedArray != nullptr);
  BOOST_CHECK_EQUAL(objExtractedArray->GetEntries(), 1);

  delete ref.header;
  delete ref.payload;
  delete array;
}

BOOST_AUTO_TEST_CASE(VectorOfHistos1D)
{
  auto histo = std::make_shared<TH1F>("histo 1d", "histo 1d", 100, 0, 100);
  histo->Fill(5);
  VectorOfTObjectPtrs vectorWithData{histo};

  auto vectorToDataRef = object_store_helpers::toRawObserverPointers(vectorWithData);

  DataRef ref = makeDataRef(&vectorToDataRef);
  auto objStore = object_store_helpers::extractObjectFrom(ref);
  BOOST_CHECK(std::holds_alternative<VectorOfTObjectPtrs>(objStore));
  auto extractedVector = std::get<VectorOfTObjectPtrs>(objStore);
  BOOST_CHECK(extractedVector.size() == 1);
  auto* extractedHisto = dynamic_cast<TH1F*>(extractedVector[0].get());
  BOOST_CHECK(gsl::span(histo->GetArray(), histo->GetSize()) == gsl::span(extractedHisto->GetArray(), extractedHisto->GetSize()));
}

BOOST_AUTO_TEST_SUITE_END()
