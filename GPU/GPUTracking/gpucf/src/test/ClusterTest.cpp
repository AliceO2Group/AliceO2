#include <catch2/catch.hpp>

#include <gpucf/common/Cluster.h>

#include <cmath>


using namespace gpucf;


TEST_CASE("Test for NaNs in Cluster", "[Cluster]")
{
    Cluster c;

    SECTION("Detects no false positive")
    {
        REQUIRE_FALSE(c.hasNaN());
    }

    SECTION("Detects NaN in Q")
    {
        c.Q = NAN;
        REQUIRE(c.hasNaN());
    }

    SECTION("Detects NaN in QMax")
    {
        c.QMax = NAN;
        REQUIRE(c.hasNaN());
    }

    SECTION("Detects NaN in padMean")
    {
        c.padMean = NAN;
        REQUIRE(c.hasNaN());
    }

    SECTION("Detects NaN in timeMean")
    {
        c.timeMean = NAN;
        REQUIRE(c.hasNaN());
    }
    
    SECTION("Detects NaN in padSigma")
    {
        c.padSigma = NAN;
        REQUIRE(c.hasNaN());
    }

    SECTION("Detects NaN in timeSigma")
    {
        c.timeSigma = NAN;
        REQUIRE(c.hasNaN());
    }
    
}

TEST_CASE("Test for negative entries in Cluster", "[Cluster]")
{
    Cluster c;

    SECTION("Detects no false positive")
    {
        REQUIRE_FALSE(c.hasNegativeEntries());
    }

    SECTION("Detects negative Q")
    {
        c.Q = -1;
        REQUIRE(c.hasNegativeEntries());
    }

    SECTION("Detects negative QMax")
    {
        c.QMax = -1;
        REQUIRE(c.hasNegativeEntries());
    }

    SECTION("Detects negative padMean")
    {
        c.padMean = -1;
        REQUIRE(c.hasNegativeEntries()); 
    }

    SECTION("Detects negative timeMean")
    {
        c.timeMean = -1;
        REQUIRE(c.hasNegativeEntries());
    }

    SECTION("Detects negative padSigma")
    {
        c.padSigma = -1;
        REQUIRE(c.hasNegativeEntries());
    }

    SECTION("Detects negative timeSigma")
    {
        c.timeSigma = -1;
        REQUIRE(c.hasNegativeEntries());
    }
}

TEST_CASE("Test equality of Clusters", "[Cluster]")
{
    Cluster c;
    Cluster d = c;

    SECTION("Detects equal Clusters")
    {
        REQUIRE(c == d);
    }

    SECTION("Detects approx. equal Clusters")
    {
        d.padMean += 0.00005;
        REQUIRE(c == d);
    }

    SECTION("Detects unequal cru")
    {
        d.cru = 1;
        REQUIRE_FALSE(c == d);
    }

    SECTION("Detects unequal row")
    {
        d.row = 1;
        REQUIRE_FALSE(c == d);
    }

    SECTION("Detects unequal Q")
    {
        d.Q = 1;
        REQUIRE_FALSE(c == d);
    }

    SECTION("Detects unequal QMax")
    {
        d.QMax = 1;
        REQUIRE_FALSE(c == d);
    }

    SECTION("Detects unequal padMean")
    {
        d.padMean = 1;
        REQUIRE_FALSE(c == d);
    }

    SECTION("Detects unequal timeMean")
    {
        d.timeMean = 1;
        REQUIRE_FALSE(c == d);
    }

    SECTION("Detects unequal padSigma")
    {
        d.padSigma = 1;
        REQUIRE_FALSE(c == d);
    }

    SECTION("Detects unequal timeSigma")
    {
        d.timeSigma = 1;
        REQUIRE_FALSE(c == d);
    }
}

TEST_CASE("Test Cluster serialization", "[Cluster]")
{    
    Cluster c(0,1,2,3,4,5,6,7);
    Object obj = c.serialize();

    REQUIRE(obj.getType() == "Cluster");

    REQUIRE(obj.hasKey("cru"));
    REQUIRE(obj.hasKey("row"));
    REQUIRE(obj.hasKey("Q"));
    REQUIRE(obj.hasKey("QMax"));
    REQUIRE(obj.hasKey("padMean"));
    REQUIRE(obj.hasKey("timeMean"));
    REQUIRE(obj.hasKey("padSigma"));
    REQUIRE(obj.hasKey("timeSigma"));

    REQUIRE(obj.getInt("cru") == 0);
    REQUIRE(obj.getInt("row") == 1);
    REQUIRE(obj.getFloat("Q") == 2);
    REQUIRE(obj.getFloat("QMax") == 3);
    REQUIRE(obj.getFloat("padMean") == 4);
    REQUIRE(obj.getFloat("timeMean") == 5);
    REQUIRE(obj.getFloat("padSigma") == 6);
    REQUIRE(obj.getFloat("timeSigma") == 7);
}

TEST_CASE("Test Cluster deserialization", "[Cluster]")
{
    Object obj("Cluster"); 

    obj.set("cru", 8);
    obj.set("row", 1);
    obj.set("Q", 2);
    obj.set("QMax", 3);
    obj.set("padMean", 4);
    obj.set("timeMean", 5);
    obj.set("padSigma", 6);
    obj.set("timeSigma", 7);

    Cluster c;
    c.deserialize(obj);

    REQUIRE(c.cru == 8);
    REQUIRE(c.row == 1);
    REQUIRE(c.Q == 2);
    REQUIRE(c.QMax == 3);
    REQUIRE(c.padMean == 4);
    REQUIRE(c.timeMean == 5);
    REQUIRE(c.padSigma == 6);
    REQUIRE(c.timeSigma == 7);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
