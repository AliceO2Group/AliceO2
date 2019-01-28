#include <catch2/catch.hpp>

#include <gpucf/common/Object.h>


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
   
        REQUIRE_FALSE(optObj.has_value());
    }

    SECTION("Fails on missing key")
    {
        auto optObj = Object::tryParse("Foo: = 0, y = 1");

        REQUIRE_FALSE(optObj.has_value());
    }

    SECTION("Fails on missing value")
    {
        auto optObj = Object::tryParse("Foo: x = , y = 2");  

        REQUIRE_FALSE(optObj.has_value());
    }
}


TEST_CASE("Test Object construction", "[Object]")
{
    Object obj("Foo"); 
    obj.set("x", 12);
    obj.set("y", 20.5f);

    SECTION("Sets type info correctly")
    {
        REQUIRE(obj.getType() == "Foo");
    }

    SECTION("Does not find missing key")
    {
        REQUIRE_FALSE(obj.hasKey("z"));
    }

    SECTION("Finds integer key")
    {
        REQUIRE(obj.hasKey("x"));
        REQUIRE(obj.getInt("x") == 12);
    }

    SECTION("Finds float key")
    {
        REQUIRE(obj.hasKey("y"));
        REQUIRE(obj.getFloat("y") == 20.5);
    }

    SECTION("Stringifies correctly")
    {
        nonstd::optional<Object> optObj2 = Object::tryParse(obj.str());

        REQUIRE(optObj2.has_value());

        Object obj2 = *optObj2;

        REQUIRE(obj2.hasKey("x"));
        REQUIRE(obj2.hasKey("y"));

        REQUIRE(obj2.getInt("x") == obj.getInt("x"));
        REQUIRE(obj2.getInt("y") == obj.getInt("y"));
    }
}



// vim: set ts=4 sw=4 sts=4 expandtab:
