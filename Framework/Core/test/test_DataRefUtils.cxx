// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework DataRefUtils
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <TObject.h>
#include <TObjString.h>
#include <TObjArray.h>
#include <TMessage.h>
#include "Framework/RootSerializationSupport.h"
#include "Framework/DataRefUtils.h"
#include <boost/test/unit_test.hpp>

#include <memory>

using namespace o2::framework;

// Simple test to do root deserialization.
BOOST_AUTO_TEST_CASE(TestRootSerialization)
{
  DataRef ref;
  TMessage* tm = new TMessage(kMESS_OBJECT);
  auto sOrig = std::make_unique<TObjString>("test");
  tm->WriteObject(sOrig.get());
  o2::header::DataHeader dh;
  dh.payloadSerializationMethod = o2::header::gSerializationMethodROOT;
  ref.payload = tm->Buffer();
  dh.payloadSize = tm->BufferSize();
  ref.header = reinterpret_cast<char const*>(&dh);

  // Check by using the same type
  auto s = DataRefUtils::as<TObjString>(ref);
  BOOST_REQUIRE(s.get() != nullptr);
  BOOST_CHECK_EQUAL(std::string(s->GetString().Data()), "test");
  BOOST_CHECK_EQUAL(std::string(s->GetName()), "test");

  // Check by using the base type.
  auto o = DataRefUtils::as<TObject>(ref);
  BOOST_REQUIRE(o.get() != nullptr);
  BOOST_CHECK_EQUAL(std::string(o->GetName()), "test");
}

// Simple test for ROOT container deserialization.
BOOST_AUTO_TEST_CASE(TestRootContainerSerialization)
{
  DataRef ref;
  TMessage* tm = new TMessage(kMESS_OBJECT);
  TObjArray container;
  // the original container is explicitly non-owning, its owning
  // flag is preserved during serialization
  container.SetOwner(false);
  auto object = std::make_unique<TObjString>("test");
  container.Add(object.get());
  tm->WriteObject(&container);
  o2::header::DataHeader dh;
  dh.payloadSerializationMethod = o2::header::gSerializationMethodROOT;
  ref.payload = tm->Buffer();
  dh.payloadSize = tm->BufferSize();
  ref.header = reinterpret_cast<char const*>(&dh);

  auto s = DataRefUtils::as<TObjArray>(ref);
  BOOST_REQUIRE(s.get() != nullptr);
  BOOST_REQUIRE(s->GetEntries() == 1);
  BOOST_CHECK_EQUAL(std::string(s->At(0)->GetName()), "test");
  // the extracted object must be owning to avoid memory leaks
  // the get method takes care of this
  BOOST_CHECK(s->IsOwner());
}
