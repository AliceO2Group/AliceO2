// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <catch2/catch.h>

#include <gpucf/common/DataSet.h>

using namespace gpucf;

class SerializeTester
{

 public:
  int x;
  float y;

  Object serialize() const
  {
    Object obj("SerializeTester");

    SET_FIELD(obj, x);
    SET_FIELD(obj, y);

    return obj;
  }

  void deserialize(const Object& obj)
  {
    GET_INT(obj, x);
    GET_FLOAT(obj, y);
  }
};

TEST_CASE("Test DataSet serialization of vectors", "[DataSet]")
{
  std::vector<SerializeTester> in(2);
  in[0].x = 0;
  in[0].y = 2.5;
  in[1].x = 1;
  in[2].y = 4.5;

  DataSet d;

  d.serialize(in);

  auto out = d.deserialize<SerializeTester>();

  REQUIRE(out.size() == 2);

  REQUIRE(out[0].x == in[0].x);
  REQUIRE(out[0].y == in[0].y);
  REQUIRE(out[1].x == in[1].x);
  REQUIRE(out[1].y == in[1].y);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
