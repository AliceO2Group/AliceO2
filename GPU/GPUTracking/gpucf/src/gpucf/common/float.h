#pragma once

static constexpr float FEQ_EPSILON = 0.5;

static inline bool floatEq(float f1, float f2, float epsilon=FEQ_EPSILON)
{
    return std::abs(f1 - f2) <= epsilon;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
