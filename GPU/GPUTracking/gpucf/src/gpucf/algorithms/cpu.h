#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/common/Map.h>
#include <gpucf/common/RowMap.h>
#include <gpucf/common/View.h>


namespace gpucf
{

std::vector<Digit> findPeaks(View<Digit>, const Map<float> &);

RowMap<std::vector<Digit>> findPeaksByRow(View<Digit>, const Map<float> &);

RowMap<Map<bool>> makePeakMapByRow(const RowMap<std::vector<Digit>> &);

bool isPeak(const Digit &, const Map<float> &, float cutoff=0.f);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
