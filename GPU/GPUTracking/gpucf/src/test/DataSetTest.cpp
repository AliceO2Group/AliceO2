#include <catch2/catch.hpp>

#include <gpucf/DataSet.h>


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

    void deserialize(const Object &obj)
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
