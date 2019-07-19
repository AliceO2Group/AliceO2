#pragma once

#include <shared/tpc.h>

#include <array>


namespace gpucf
{

    template<typename T>
    using RowMap = std::array<T, TPC_NUM_OF_ROWS>;
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
