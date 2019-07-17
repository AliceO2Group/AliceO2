#pragma once

#include <nonstd/span.hpp>


namespace gpucf
{
    
    template<typename T>
    using View = nonstd::span<const T>;

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
