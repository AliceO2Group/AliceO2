#include <catch2/catch.hpp>

#include <gpucf/Object.h>


using namespace gpucf;


TEST_CASE("Test Object::tryParse", "[Object]")
{
    SECTION("Parses correct string")
    {
        auto optObj = Object::tryParse("Foo: x = 9, y = 1");

        REQUIRE(optObj.has_value());

        Object obj = *optObj;

        REQUIRE(obj.getType() == "Foo");

        REQUIRE(obj.hasKey("x"));
        REQUIRE(obj.getInt("x") == 9);

        REQUIRE(obj.hasKey("y"));
        REQUIRE(obj.getInt("y") == 1);
    }

    SECTION("Parses correct string without spaces")
    {
        auto optObj = Object::tryParse("Foo:x=9,y=1");

        REQUIRE(optObj.has_value());

        Object obj = *optObj;

        REQUIRE(obj.getType() == "Foo");

        REQUIRE(obj.hasKey("x"));
        REQUIRE(obj.getInt("x") == 9);

        REQUIRE(obj.hasKey("y"));
        REQUIRE(obj.getInt("y") == 1);
    }

    SECTION("Parses correct string with many spaces")
    {
        auto optObj = Object::tryParse("  Foo  :  x =   9  ,    y =  1    ");

        REQUIRE(optObj.has_value());

        Object obj = *optObj;

        REQUIRE(obj.getType() == "Foo");

        REQUIRE(obj.hasKey("x"));
        REQUIRE(obj.getInt("x") == 9);

        REQUIRE(obj.hasKey("y"));
        REQUIRE(obj.getInt("y") == 1);
    }

    SECTION("Fails on missing type name")
    {
        auto optObj = Object::tryParse("x = 9, y = 1");
   
        REQUIRE(!optObj.has_value());
    }

    SECTION("Fails on missing key")
    {
        auto optObj = Object::tryParse("Foo: = 0, y = 1");

        REQUIRE(!optObj.has_value());
    }

    SECTION("Fails on missing value")
    {
        auto optObj = Object::tryParse("Foo: x = , y = 2");  

        REQUIRE(!optObj.has_value());
    }
}


// vim: set ts=4 sw=4 sts=4 expandtab:
