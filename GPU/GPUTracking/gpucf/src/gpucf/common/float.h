#pragma once

#include <cmath>


static constexpr float FEQ_EPSILON_SMALL = 0.5;
static constexpr float FEQ_EPSILON_BIG   = 1.0;

static inline bool floatEq(float f1, float f2, float epsilon=FEQ_EPSILON_SMALL)
{
    return std::abs(f1 - f2) <= epsilon;
}


bool almostEqual(float, float);

// vim: set ts=4 sw=4 sts=4 expandtab:
