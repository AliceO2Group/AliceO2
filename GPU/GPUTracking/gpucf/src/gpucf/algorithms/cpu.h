#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/common/Map.h>
#include <gpucf/common/RowMap.h>
#include <gpucf/common/View.h>


namespace gpucf
{

RowMap<std::vector<Digit>> findPeaksByRow(View<Digit>, const Map<float> &);

bool isPeak(const Digit &, const Map<float> &, float cutoff=0.f);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
