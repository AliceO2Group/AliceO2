#pragma once

#include <array>


namespace gpucf
{

class ClEnv;

class VectorAdd 
{

public:
    VectorAdd();

    bool run(ClEnv &);

private:
    static constexpr size_t N = 2048;

    std::array<int, N> a;
    std::array<int, N> b;
    std::array<int, N> c;
};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
