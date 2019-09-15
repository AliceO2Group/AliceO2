// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <catch2/catch.hpp>

#include <gpucf/common/Digit.h>


using namespace gpucf;


TEST_CASE("Test Digit serialization", "[Digit]")
{
    Digit d(0, 2, 3, 4);  
    Object obj = d.serialize();

    REQUIRE(obj.getType() == "Digit");

    REQUIRE(obj.hasKey("charge"));
    REQUIRE(obj.hasKey("row"));
    REQUIRE(obj.hasKey("pad"));
    REQUIRE(obj.hasKey("time"));

    REQUIRE(obj.getFloat("charge") == 0);
    REQUIRE(obj.getInt("row") == 2);
    REQUIRE(obj.getInt("pad") == 3);
    REQUIRE(obj.getInt("time") == 4);
}

TEST_CASE("Test Digit deserialization", "[Digit]")
{
    Object obj("Digit");

    obj.set("charge", 0.5f);
    obj.set("row", 2);
    obj.set("pad", 3);
    obj.set("time", 4);

    Digit d;
    d.deserialize(obj);

    REQUIRE(d.charge == 0.5);
    REQUIRE(d.row == 2);
    REQUIRE(d.pad == 3);
    REQUIRE(d.time == 4);
}
// vim: set ts=4 sw=4 sts=4 expandtab:

