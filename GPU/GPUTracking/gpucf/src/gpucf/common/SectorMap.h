#pragma once

#include <shared/tpc.h>

#include <array>


namespace gpucf
{

    template<typename T>
    using SectorMap = std::array<T, TPC_SECTORS>;

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
