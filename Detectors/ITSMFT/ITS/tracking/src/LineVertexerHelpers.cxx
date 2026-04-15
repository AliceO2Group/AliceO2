// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Math/SMatrix.h>
#include <Math/SVector.h>

#include "ITStracking/Constants.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/LineVertexerHelpers.h"

namespace o2::its::line_vertexer
{
namespace
{
using SymMatrix3 = ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepSym<float, 3>>;
using SVector3 = ROOT::Math::SVector<float, 3>;

constexpr float TukeyC = 4.685f;
constexpr float TukeyC2 = TukeyC * TukeyC;
constexpr float InitialScale2 = 5.f;
constexpr float MinScale2 = 1.f;
constexpr float MedianToSigma = 1.4826f;
constexpr float VertexShiftZTol = 0.01f;
constexpr float VertexShiftR2Tol = 1.e-4f;
constexpr int MaxFitIterations = 10;
constexpr int MaxSeedsPerCluster = 32;
constexpr float MinRelativePeakSupport = 0.1f;
constexpr int MaxHistogramBins = 0x7fff;
constexpr float TieTolerance = 1e-5f;

struct LineRef {
  LineRef(const Line& line, const int index, const float beamX, const float beamY, const float maxZ) : lineIndex(index)
  {
    const auto symTime = line.mTime.makeSymmetrical();
    tCenter = symTime.getTimeStamp();
    tHalfWidth = symTime.getTimeStampError();
    const auto dx = line.originPoint(0) - beamX;
    const auto dy = line.originPoint(1) - beamY;
    const auto ux = line.cosinesDirector(0);
    const auto uy = line.cosinesDirector(1);
    const auto uz = line.cosinesDirector(2);
    const auto den = math_utils::SqSum(ux, uy);
    if (den <= constants::Tolerance) {
      lineIndex = constants::UnusedIndex;
      return;
    }
    const auto s0 = -((dx * ux) + (dy * uy)) / den;
    const auto xb = dx + (s0 * ux);
    const auto yb = dy + (s0 * uy);
    zBeam = line.originPoint(2) + s0 * uz;
    if (!std::isfinite(zBeam) || o2::gpu::CAMath::Abs(zBeam) > maxZ) {
      lineIndex = constants::UnusedIndex;
    }
  }
  bool isDead() const noexcept { return lineIndex == constants::UnusedIndex; }

  int lineIndex = constants::UnusedIndex;
  float zBeam = 0.f;
  float tCenter = 0.f;
  float tHalfWidth = 0.f;
};

struct VertexSeed {
  explicit VertexSeed(const std::shared_ptr<BoundedMemoryResource>& mr) : contributors(mr.get()), assigned(mr.get()) {}

  std::array<float, 3> vertex = {};
  TimeEstBC time;
  float scale2 = InitialScale2;
  bounded_vector<int> contributors;
  bounded_vector<int> assigned;
  bool valid = false;
  bool isUsableSeed() const noexcept
  {
    return valid && contributors.size() >= 2;
  }
};

void compactSeeds(bounded_vector<VertexSeed>& seeds)
{
  seeds.erase(std::remove_if(seeds.begin(), seeds.end(), [](const VertexSeed& seed) {
                return !seed.isUsableSeed();
              }),
              seeds.end());
}

struct Histogram2D {
  explicit Histogram2D(const std::shared_ptr<BoundedMemoryResource>& mr) : bins(mr.get()) {}

  int nTimeBins = 0;
  int nZBins = 0;
  float timeMin = 0.f;
  float zMin = 0.f;
  float timeBinSize = 1.f;
  float zBinSize = 1.f;
  bounded_vector<float> bins;

  int getIndex(const int tBin, const int zBin) const noexcept
  {
    return (tBin * nZBins) + zBin;
  }

  std::pair<int, int> decodeIndex(const int index) const noexcept
  {
    return {index / nZBins, index % nZBins};
  }

  int getTimeBin(const float time) const noexcept
  {
    if (time < timeMin) {
      return -1;
    }
    const auto bin = static_cast<int>((time - timeMin) / timeBinSize);
    return (bin >= 0 && bin < nTimeBins) ? bin : -1;
  }

  int getZBin(const float z) const noexcept
  {
    if (z < zMin) {
      return -1;
    }
    const auto bin = static_cast<int>((z - zMin) / zBinSize);
    return (bin >= 0 && bin < nZBins) ? bin : -1;
  }

  void fill(const float time, const float z, const float weight) noexcept
  {
    const auto tBin = getTimeBin(time);
    const auto zBin = getZBin(z);
    if (tBin < 0 || zBin < 0) {
      return;
    }
    bins[getIndex(tBin, zBin)] += weight;
  }

  int findPeakBin() const noexcept
  {
    float bestWeight = 0.f;
    int bestIndex = -1;
    for (int index = 0; index < static_cast<int>(bins.size()); ++index) {
      if (bins[index] > bestWeight) {
        bestWeight = bins[index];
        bestIndex = index;
      }
    }
    return bestIndex;
  }

  void suppressBin(const int index) noexcept
  {
    if (index >= 0 && index < static_cast<int>(bins.size())) {
      bins[index] = -1.f;
    }
  }

  void suppressNeighborhood(const int index, const int radiusTime, const int radiusZ) noexcept
  {
    if (index < 0) {
      return;
    }
    const auto [tBin, zBin] = decodeIndex(index);
    for (int dt = -radiusTime; dt <= radiusTime; ++dt) {
      const auto tt = tBin + dt;
      if (tt < 0 || tt >= nTimeBins) {
        continue;
      }
      for (int dz = -radiusZ; dz <= radiusZ; ++dz) {
        const auto zz = zBin + dz;
        if (zz < 0 || zz >= nZBins) {
          continue;
        }
        bins[getIndex(tt, zz)] = -1.f;
      }
    }
  }

  float getNeighborhoodSum(const int index, const int radiusTime, const int radiusZ) const noexcept
  {
    if (index < 0) {
      return 0.f;
    }
    const auto [tBin, zBin] = decodeIndex(index);
    float sum = 0.f;
    for (int dt = -radiusTime; dt <= radiusTime; ++dt) {
      const auto tt = tBin + dt;
      if (tt < 0 || tt >= nTimeBins) {
        continue;
      }
      for (int dz = -radiusZ; dz <= radiusZ; ++dz) {
        const auto zz = zBin + dz;
        if (zz < 0 || zz >= nZBins) {
          continue;
        }
        const auto value = bins[getIndex(tt, zz)];
        if (value > 0.f) {
          sum += value;
        }
      }
    }
    return sum;
  }

  float getTimeBinCenter(const int tBin) const noexcept
  {
    return timeMin + ((static_cast<float>(tBin) + 0.5f) * timeBinSize);
  }

  float getZBinCenter(const int zBin) const noexcept
  {
    return zMin + ((static_cast<float>(zBin) + 0.5f) * zBinSize);
  }

  TimeEstBC getTimeInterval(const int tBin) const noexcept
  {
    const auto lowFloat = timeMin + (static_cast<float>(tBin) * timeBinSize);
    const auto highFloat = lowFloat + timeBinSize;
    const auto low = std::max<double>(0., std::floor(lowFloat));
    const auto high = std::max(low + 1., (double)std::ceil(highFloat));
    constexpr auto maxTS = std::numeric_limits<TimeStampType>::max();
    const auto clampedLow = std::min<double>(low, maxTS - 1.);
    const auto width = std::min<double>(high - clampedLow, std::numeric_limits<TimeStampErrorType>::max());
    return {static_cast<TimeStampType>(clampedLow), static_cast<TimeStampErrorType>(std::max<double>(1., width))};
  }

  TimeEstBC getTimeNeighborhoodInterval(const int tBin, const int radius) const noexcept
  {
    const auto lowBin = std::max(0, tBin - radius);
    const auto highBin = std::min(nTimeBins - 1, tBin + radius);
    const auto lowFloat = timeMin + (static_cast<float>(lowBin) * timeBinSize);
    const auto highFloat = timeMin + (static_cast<float>(highBin + 1) * timeBinSize);
    const auto low = std::max<double>(0., std::floor(lowFloat));
    const auto high = std::max(low + 1., (double)std::ceil(highFloat));
    constexpr auto maxTS = std::numeric_limits<TimeStampType>::max();
    const auto clampedLow = std::min<double>(low, maxTS - 1.);
    const auto width = std::min<double>(high - clampedLow, std::numeric_limits<TimeStampErrorType>::max());
    return {static_cast<TimeStampType>(clampedLow), static_cast<TimeStampErrorType>(std::max<double>(1., width))};
  }
};

class SeedHistogram
{
 public:
  SeedHistogram(std::span<const int> members,
                std::span<const LineRef> lineRefs,
                std::span<const Line> lines,
                const Settings& settings)
    : mMembers(members), mLineRefs(lineRefs), mSeedMemberRadiusTime(settings.seedMemberRadiusTime), mSeedMemberRadiusZ(settings.seedMemberRadiusZ), mMemoryPool(settings.memoryPool), mHistogram(mMemoryPool)
  {
    const auto zBinSize = 0.25f * settings.clusterCut;
    const auto timeBinSize = medianTimeError(lines);

    float minZ = std::numeric_limits<float>::max();
    float maxZ = std::numeric_limits<float>::lowest();
    float minTime = std::numeric_limits<float>::max();
    float maxTime = std::numeric_limits<float>::lowest();
    for (const auto lineRefIdx : mMembers) {
      minZ = std::min(minZ, mLineRefs[lineRefIdx].zBeam);
      maxZ = std::max(maxZ, mLineRefs[lineRefIdx].zBeam);
      minTime = std::min(minTime, mLineRefs[lineRefIdx].tCenter);
      maxTime = std::max(maxTime, mLineRefs[lineRefIdx].tCenter);
    }

    const auto dz = std::max(0.f, maxZ - minZ);
    const auto dt = std::max(0.f, maxTime - minTime);
    mHistogram.nZBins = 1 + static_cast<int>(dz / zBinSize);
    mHistogram.nTimeBins = 1 + static_cast<int>(dt / timeBinSize);
    if (mHistogram.nTimeBins * mHistogram.nZBins > MaxHistogramBins) {
      if (mHistogram.nTimeBins > mHistogram.nZBins) {
        mHistogram.nTimeBins = std::max(1, (MaxHistogramBins - 1) / std::max(1, mHistogram.nZBins));
      } else {
        mHistogram.nZBins = std::max(1, (MaxHistogramBins - 1) / std::max(1, mHistogram.nTimeBins));
      }
    }

    mHistogram.timeBinSize = std::max(timeBinSize, dt / (float)std::max(1, mHistogram.nTimeBins));
    mHistogram.zBinSize = std::max(zBinSize, dz / (float)std::max(1, mHistogram.nZBins));
    const auto paddedTime = 0.5f * ((float)mHistogram.nTimeBins * mHistogram.timeBinSize - dt);
    const auto paddedZ = 0.5f * ((float)mHistogram.nZBins * mHistogram.zBinSize - dz);
    mHistogram.timeMin = minTime - paddedTime;
    mHistogram.zMin = minZ - paddedZ;
    mHistogram.bins.assign((size_t)mHistogram.nTimeBins * (size_t)mHistogram.nZBins, 0.f);

    for (const auto lineRefIdx : mMembers) {
      mHistogram.fill(mLineRefs[lineRefIdx].tCenter, mLineRefs[lineRefIdx].zBeam, 1.f);
    }
  }

  int findPeakBin() const noexcept
  {
    return mHistogram.findPeakBin();
  }

  float getPeakSupport(const int peakIndex) const noexcept
  {
    return mHistogram.getNeighborhoodSum(peakIndex, mSeedMemberRadiusTime, mSeedMemberRadiusZ);
  }

  bounded_vector<int> collectLocalMembers(const int peakIndex, const int radiusTime, const int radiusZ) const
  {
    bounded_vector<int> localMembers(mMemoryPool.get());
    localMembers.reserve(mMembers.size());
    const auto [timeBin, zBin] = mHistogram.decodeIndex(peakIndex);
    for (const auto lineRefIdx : mMembers) {
      const auto memberTimeBin = mHistogram.getTimeBin(mLineRefs[lineRefIdx].tCenter);
      const auto memberZBin = mHistogram.getZBin(mLineRefs[lineRefIdx].zBeam);
      if (memberTimeBin < 0 || memberZBin < 0) {
        continue;
      }
      if (o2::gpu::GPUCommonMath::Abs(memberTimeBin - timeBin) > radiusTime) {
        continue;
      }
      if (o2::gpu::GPUCommonMath::Abs(memberZBin - zBin) > radiusZ) {
        continue;
      }
      localMembers.push_back(lineRefIdx);
    }
    return localMembers;
  }

  TimeEstBC getPeakTimeInterval(const int peakIndex, const int radius = 0) const noexcept
  {
    return mHistogram.getTimeNeighborhoodInterval(mHistogram.decodeIndex(peakIndex).first, radius);
  }

  float getPeakZCenter(const int peakIndex) const noexcept
  {
    return mHistogram.getZBinCenter(mHistogram.decodeIndex(peakIndex).second);
  }

  void suppressPeak(const int peakIndex) noexcept
  {
    mHistogram.suppressBin(peakIndex);
  }

  void suppressPeakNeighborhood(const int peakIndex) noexcept
  {
    mHistogram.suppressNeighborhood(peakIndex, mSeedMemberRadiusTime, mSeedMemberRadiusZ);
  }

 private:
  float medianTimeError(std::span<const Line> lines) const
  {
    bounded_vector<float> errors(mMemoryPool.get());
    errors.reserve(mMembers.size());
    for (const auto lineRefIdx : mMembers) {
      errors.push_back(static_cast<float>(lines[mLineRefs[lineRefIdx].lineIndex].mTime.getTimeStampError()));
    }
    std::sort(errors.begin(), errors.end());
    return errors.empty() ? 1.f : std::max(1.f, errors[errors.size() / 2]);
  }

  std::span<const int> mMembers;
  std::span<const LineRef> mLineRefs;
  int mSeedMemberRadiusTime = 1;
  int mSeedMemberRadiusZ = 2;
  std::shared_ptr<BoundedMemoryResource> mMemoryPool;
  Histogram2D mHistogram;
};

float updateScale2(const std::span<const float> chi2s, const std::shared_ptr<BoundedMemoryResource>& mr) noexcept
{
  if (chi2s.empty()) {
    return MinScale2;
  }

  bounded_vector<float> sorted(chi2s.begin(), chi2s.end(), mr.get());
  std::sort(sorted.begin(), sorted.end());
  const auto median = sorted[sorted.size() / 2];

  for (auto& value : sorted) {
    value = o2::gpu::GPUCommonMath::Abs(value - median);
  }
  std::sort(sorted.begin(), sorted.end());
  const auto mad = sorted[sorted.size() / 2];
  if (!std::isfinite(mad) || mad <= constants::Tolerance) {
    return MinScale2;
  }
  return std::max(MinScale2, MedianToSigma * mad);
}

class VertexFit
{
 public:
  void add(const Line& line, const float weight) noexcept
  {
    const auto& direction = line.cosinesDirector;
    const auto& origin = line.originPoint;
    const auto det = ROOT::Math::Dot(direction, direction);
    if (det <= constants::Tolerance) {
      return;
    }

    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        mMatrix(i, j) += weight * (((i == j ? det : 0.f) - direction(i) * direction(j)) / det);
      }
    }

    const auto dDotO = ROOT::Math::Dot(direction, origin);
    for (int i = 0; i < 3; ++i) {
      mRhs(i) += weight * ((direction(i) * dDotO - det * origin(i)) / det);
    }
  }

  bool solve(std::array<float, 3>& vertexOut) const noexcept
  {
    SymMatrix3 inv{mMatrix};
    if (!inv.InvertFast()) {
      return false;
    }
    const auto solution = inv * mRhs;
    vertexOut[0] = static_cast<float>(-solution(0));
    vertexOut[1] = static_cast<float>(-solution(1));
    vertexOut[2] = static_cast<float>(-solution(2));
    return std::isfinite(vertexOut[0]) && std::isfinite(vertexOut[1]) && std::isfinite(vertexOut[2]);
  }

 private:
  SymMatrix3 mMatrix;
  SVector3 mRhs;
};

VertexSeed fitSeed(const VertexSeed& initialSeed,
                   std::span<const int> members,
                   std::span<const LineRef> lineRefs,
                   std::span<const Line> lines,
                   const std::shared_ptr<BoundedMemoryResource>& mr,
                   const float pairCut2)
{
  VertexSeed seed{mr};
  seed.vertex = initialSeed.vertex;
  seed.time = initialSeed.time;
  seed.scale2 = initialSeed.scale2;
  seed.valid = false;
  seed.contributors.clear();
  seed.assigned.clear();
  if (members.size() < 2) {
    return seed;
  }

  for (int iteration = 0; iteration < MaxFitIterations; ++iteration) {
    VertexFit vertexFit;
    TimeEstBC commonTime{};
    bool hasCommonTime = false;
    bounded_vector<int> contributors{mr.get()};
    const auto scale2 = std::max(seed.scale2, MinScale2);
    const auto tukeyFactor = 1.f / (scale2 * TukeyC2);

    for (const auto lineRefIdx : members) {
      const auto lineIdx = lineRefs[lineRefIdx].lineIndex;
      const auto& line = lines[lineIdx];
      if (!line.mTime.isCompatible(seed.time)) {
        continue;
      }
      if (hasCommonTime && !line.mTime.isCompatible(commonTime)) {
        continue;
      }

      const auto chi2 = Line::getDistance2FromPoint(line, seed.vertex) / pairCut2;
      auto weight = 1.f - (chi2 * tukeyFactor);
      if (weight <= 0.f) {
        continue;
      }
      weight *= weight;

      if (!hasCommonTime) {
        commonTime = line.mTime;
        hasCommonTime = true;
      } else {
        commonTime += line.mTime;
      }

      contributors.push_back(lineRefIdx);
      vertexFit.add(line, weight);
    }

    if (!hasCommonTime || contributors.size() < 2) {
      return seed;
    }

    std::sort(contributors.begin(), contributors.end());

    std::array<float, 3> updatedVertex{};
    if (!vertexFit.solve(updatedVertex)) {
      return seed;
    }

    const auto sameContributors = contributors == seed.contributors;
    const auto dz = o2::gpu::GPUCommonMath::Abs(updatedVertex[2] - seed.vertex[2]);
    const auto oldR2 = (seed.vertex[0] * seed.vertex[0]) + (seed.vertex[1] * seed.vertex[1]);
    const auto newR2 = (updatedVertex[0] * updatedVertex[0]) + (updatedVertex[1] * updatedVertex[1]);
    const auto dr2 = o2::gpu::GPUCommonMath::Abs(newR2 - oldR2);

    seed.vertex = updatedVertex;
    seed.time = commonTime;
    bounded_vector<float> updatedChi2s{mr.get()};
    updatedChi2s.reserve(contributors.size());
    for (const auto lineRefIx : contributors) {
      updatedChi2s.push_back(Line::getDistance2FromPoint(lines[lineRefs[lineRefIx].lineIndex], seed.vertex) / pairCut2);
    }
    seed.scale2 = updateScale2(updatedChi2s, mr);
    seed.contributors = std::move(contributors);
    seed.valid = true;

    if (sameContributors && dz < VertexShiftZTol && dr2 < VertexShiftR2Tol) {
      break;
    }
  }

  return seed;
}

size_t countSharedContributors(std::span<const int> lhs, std::span<const int> rhs) noexcept
{
  size_t shared = 0;
  auto lhsIt = lhs.begin();
  auto rhsIt = rhs.begin();
  while (lhsIt != lhs.end() && rhsIt != rhs.end()) {
    if (*lhsIt == *rhsIt) {
      ++shared;
      ++lhsIt;
      ++rhsIt;
    } else if (*lhsIt < *rhsIt) {
      ++lhsIt;
    } else {
      ++rhsIt;
    }
  }
  return shared;
}

bounded_vector<int> collectCompatibleContributors(const VertexSeed& seed,
                                                  std::span<const int> members,
                                                  std::span<const LineRef> lineRefs,
                                                  std::span<const Line> lines,
                                                  const std::shared_ptr<BoundedMemoryResource>& mr,
                                                  const float pairCut2)
{
  bounded_vector<int> contributors{mr.get()};
  contributors.reserve(members.size());
  for (const auto lineRefIdx : members) {
    const auto lineIdx = lineRefs[lineRefIdx].lineIndex;
    const auto& line = lines[lineIdx];
    if (!line.mTime.isCompatible(seed.time)) {
      continue;
    }
    if (Line::getDistance2FromPoint(line, seed.vertex) >= pairCut2) {
      continue;
    }
    contributors.push_back(lineRefIdx);
  }
  std::sort(contributors.begin(), contributors.end());
  return contributors;
}

void deduplicateSeeds(bounded_vector<VertexSeed>& seeds, const Settings& settings)
{
  if (seeds.size() < 2) {
    return;
  }

  std::sort(seeds.begin(), seeds.end(), [](const VertexSeed& lhs, const VertexSeed& rhs) {
    if (lhs.contributors.size() != rhs.contributors.size()) {
      return lhs.contributors.size() > rhs.contributors.size();
    }
    if (o2::gpu::GPUCommonMath::Abs(lhs.scale2 - rhs.scale2) > constants::Tolerance) {
      return lhs.scale2 < rhs.scale2;
    }
    return lhs.vertex[2] < rhs.vertex[2];
  });

  const auto dedupZCut = settings.seedDedupZCut > 0.f ? settings.seedDedupZCut : 0.25f * settings.clusterCut;
  for (size_t i = 0; i < seeds.size(); ++i) {
    auto& candidate = seeds[i];
    if (!candidate.isUsableSeed()) {
      candidate.valid = false;
      continue;
    }
    bool duplicate = false;
    for (size_t j = 0; j < i; ++j) {
      const auto& kept = seeds[j];
      if (!kept.isUsableSeed()) {
        continue;
      }
      if (!candidate.time.isCompatible(kept.time)) {
        continue;
      }
      const auto shared = countSharedContributors(candidate.contributors, kept.contributors);
      const auto minSize = std::min(candidate.contributors.size(), kept.contributors.size());
      const auto zDelta = o2::gpu::GPUCommonMath::Abs(candidate.vertex[2] - kept.vertex[2]);
      const bool clearlyWorse = kept.contributors.size() > candidate.contributors.size() ||
                                kept.scale2 + constants::Tolerance < 0.9f * candidate.scale2;
      const bool overlapDuplicate = shared > 0 && shared * 2 >= minSize;
      const bool nearbyDuplicate = zDelta < dedupZCut && (shared > 0 || clearlyWorse);
      if (overlapDuplicate || nearbyDuplicate) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      candidate.valid = false;
    }
  }
  compactSeeds(seeds);
}

void deduplicateRefittedSeeds(bounded_vector<VertexSeed>& seeds, const Settings& settings)
{
  if (seeds.size() < 2) {
    return;
  }

  std::sort(seeds.begin(), seeds.end(), [](const VertexSeed& lhs, const VertexSeed& rhs) {
    if (lhs.contributors.size() != rhs.contributors.size()) {
      return lhs.contributors.size() > rhs.contributors.size();
    }
    if (o2::gpu::GPUCommonMath::Abs(lhs.scale2 - rhs.scale2) > constants::Tolerance) {
      return lhs.scale2 < rhs.scale2;
    }
    return lhs.vertex[2] < rhs.vertex[2];
  });

  const auto zCut = settings.refitDedupZCut > 0.f ? settings.refitDedupZCut : 0.25f * settings.clusterCut;
  for (size_t i = 0; i < seeds.size(); ++i) {
    auto& candidate = seeds[i];
    if (!candidate.isUsableSeed()) {
      candidate.valid = false;
      continue;
    }
    bool duplicate = false;
    for (size_t j = 0; j < i; ++j) {
      const auto& kept = seeds[j];
      if (!kept.isUsableSeed()) {
        continue;
      }
      if (!candidate.time.isCompatible(kept.time)) {
        continue;
      }
      const auto shared = countSharedContributors(candidate.contributors, kept.contributors);
      const auto minSize = std::min(candidate.contributors.size(), kept.contributors.size());
      const auto zDelta = o2::gpu::GPUCommonMath::Abs(candidate.vertex[2] - kept.vertex[2]);
      const bool overlapDuplicate = shared > 0 && shared * 2 >= minSize;
      const bool lowSupportPair = std::min(candidate.contributors.size(), kept.contributors.size()) < 4;
      const bool clearlyWorse = kept.contributors.size() > candidate.contributors.size() ||
                                kept.scale2 + constants::Tolerance < 0.9f * candidate.scale2;
      const bool geometricDuplicate = zDelta < zCut && (lowSupportPair || clearlyWorse);
      if (overlapDuplicate || geometricDuplicate) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      candidate.valid = false;
    }
  }
  compactSeeds(seeds);
}

struct OrderedComponent {
  explicit OrderedComponent(const std::shared_ptr<BoundedMemoryResource>& mr) : members(mr.get()) {}
  float center = 0.f;
  bounded_vector<int> members;
};

bounded_vector<bounded_vector<int>> buildCoarseClusters(std::span<const LineRef> lineRefs,
                                                        std::span<const Line> lines,
                                                        const Settings& settings)
{
  bounded_vector<bounded_vector<int>> clusters(settings.memoryPool.get());
  if (lineRefs.size() < 2) {
    return clusters;
  }

  bounded_vector<int> sortedByLower(lineRefs.size(), settings.memoryPool.get());
  std::iota(sortedByLower.begin(), sortedByLower.end(), 0);
  std::sort(sortedByLower.begin(), sortedByLower.end(), [&](const int lhs, const int rhs) {
    const auto lhsLower = lines[lineRefs[lhs].lineIndex].mTime.lower();
    const auto rhsLower = lines[lineRefs[rhs].lineIndex].mTime.lower();
    if (lhsLower != rhsLower) {
      return lhsLower < rhsLower;
    }
    return lineRefs[lhs].lineIndex < lineRefs[rhs].lineIndex;
  });

  const auto coarseZWindow = settings.coarseZWindow > 0.f ? settings.coarseZWindow : settings.clusterCut;
  bounded_vector<int> parent(lineRefs.size(), settings.memoryPool.get());
  bounded_vector<int> componentSize(lineRefs.size(), 1, settings.memoryPool.get());
  std::iota(parent.begin(), parent.end(), 0);
  float minZ = std::numeric_limits<float>::max();
  float maxZ = std::numeric_limits<float>::lowest();
  for (const auto& lineRef : lineRefs) {
    minZ = std::min(minZ, lineRef.zBeam);
    maxZ = std::max(maxZ, lineRef.zBeam);
  }
  const auto nZBins = std::max(1, 1 + static_cast<int>((maxZ - minZ) / coarseZWindow));
  auto getZBin = [&](const float z) {
    return std::clamp(static_cast<int>((z - minZ) / coarseZWindow), 0, nZBins - 1);
  };

  auto findRoot = [&](int idx) {
    int root = idx;
    while (parent[root] != root) {
      root = parent[root];
    }
    while (parent[idx] != idx) {
      const auto next = parent[idx];
      parent[idx] = root;
      idx = next;
    }
    return root;
  };

  auto unite = [&](const int lhs, const int rhs) {
    auto lhsRoot = findRoot(lhs);
    auto rhsRoot = findRoot(rhs);
    if (lhsRoot == rhsRoot) {
      return;
    }
    if (componentSize[lhsRoot] < componentSize[rhsRoot]) {
      std::swap(lhsRoot, rhsRoot);
    }
    parent[rhsRoot] = lhsRoot;
    componentSize[lhsRoot] += componentSize[rhsRoot];
  };

  using ActiveEntry = std::pair<TimeStampType, int>;
  bounded_vector<ActiveEntry> activeEntries(settings.memoryPool.get());
  std::priority_queue<ActiveEntry, bounded_vector<ActiveEntry>, std::greater<ActiveEntry>> activeByUpper(std::greater<ActiveEntry>{}, std::move(activeEntries));
  bounded_vector<uint8_t> activeMask(lineRefs.size(), 0, settings.memoryPool.get());
  bounded_vector<bounded_vector<int>> activeByZBin(settings.memoryPool.get());
  activeByZBin.reserve(nZBins);
  for (int iBin = 0; iBin < nZBins; ++iBin) {
    activeByZBin.emplace_back();
  }
  for (const auto lineRefIdx : sortedByLower) {
    const auto& lineRef = lineRefs[lineRefIdx];
    const auto& line = lines[lineRef.lineIndex];
    const auto currentLower = line.mTime.lower();

    while (!activeByUpper.empty() && activeByUpper.top().first < currentLower) {
      activeMask[activeByUpper.top().second] = 0;
      activeByUpper.pop();
    }

    const auto zBin = getZBin(lineRef.zBeam);
    for (int neighborBin = std::max(0, zBin - 1); neighborBin <= std::min(nZBins - 1, zBin + 1); ++neighborBin) {
      auto& bucket = activeByZBin[neighborBin];
      size_t writePos = 0;
      for (size_t readPos = 0; readPos < bucket.size(); ++readPos) {
        const auto oLineRefIdx = bucket[readPos];
        if (!activeMask[oLineRefIdx]) {
          continue;
        }
        bucket[writePos++] = oLineRefIdx;
        const auto& oLineRef = lineRefs[oLineRefIdx];
        if (o2::gpu::GPUCommonMath::Abs(lineRef.zBeam - oLineRef.zBeam) >= coarseZWindow) {
          continue;
        }
        const auto& otherLine = lines[oLineRef.lineIndex];
        if (line.mTime.isCompatible(otherLine.mTime)) {
          unite(lineRefIdx, oLineRefIdx);
        }
      }
      bucket.resize(writePos);
    }

    activeMask[lineRefIdx] = 1;
    activeByUpper.emplace(line.mTime.upper(), lineRefIdx);
    activeByZBin[zBin].push_back(lineRefIdx);
  }

  std::unordered_map<int, bounded_vector<int>> components;
  components.reserve(lineRefs.size());
  for (int lineRefIdx = 0; lineRefIdx < static_cast<int>(lineRefs.size()); ++lineRefIdx) {
    const auto root = findRoot(lineRefIdx);
    auto [it, inserted] = components.try_emplace(root, std::pmr::polymorphic_allocator<int>{settings.memoryPool.get()});
    (void)inserted;
    it->second.push_back(lineRefIdx);
  }

  bounded_vector<OrderedComponent> orderedComponents(settings.memoryPool.get());
  orderedComponents.reserve(components.size());
  for (auto& [root, members] : components) {
    (void)root;
    if (members.size() < 2) {
      continue;
    }
    std::sort(members.begin(), members.end(), [&](const int lhs, const int rhs) {
      const auto lhsLower = lines[lineRefs[lhs].lineIndex].mTime.lower();
      const auto rhsLower = lines[lineRefs[rhs].lineIndex].mTime.lower();
      if (lhsLower != rhsLower) {
        return lhsLower < rhsLower;
      }
      return lineRefs[lhs].lineIndex < lineRefs[rhs].lineIndex;
    });
    orderedComponents.emplace_back(settings.memoryPool);
    orderedComponents.back().center = lineRefs[members.front()].tCenter;
    orderedComponents.back().members = std::move(members);
  }

  std::sort(orderedComponents.begin(), orderedComponents.end(), [](const auto& lhs, const auto& rhs) {
    if (o2::gpu::GPUCommonMath::Abs(lhs.center - rhs.center) > TieTolerance) {
      return lhs.center < rhs.center;
    }
    return lhs.members.front() < rhs.members.front();
  });
  clusters.reserve(orderedComponents.size());
  for (auto& component : orderedComponents) {
    clusters.push_back(std::move(component.members));
  }
  return clusters;
}

bounded_vector<VertexSeed> buildSeeds(std::span<const int> members,
                                      std::span<const LineRef> lineRefs,
                                      std::span<const Line> lines,
                                      const Settings& settings)
{
  SeedHistogram histogram(members, lineRefs, lines, settings);
  bounded_vector<VertexSeed> seeds(settings.memoryPool.get());
  seeds.reserve(MaxSeedsPerCluster);
  float leadingPeakSupport = 0.f;

  while (static_cast<int>(seeds.size()) < MaxSeedsPerCluster) {
    const auto peak = histogram.findPeakBin();
    if (peak < 0) {
      break;
    }
    const auto peakSupport = histogram.getPeakSupport(peak);
    if (peakSupport < 2.f) {
      break;
    }
    if (leadingPeakSupport <= 0.f) {
      leadingPeakSupport = peakSupport;
    } else if (peakSupport < std::max(2.f, MinRelativePeakSupport * leadingPeakSupport)) {
      break;
    }
    auto localMembers = histogram.collectLocalMembers(peak, 0, 0);
    if (localMembers.size() < 2) {
      localMembers = histogram.collectLocalMembers(peak, settings.seedMemberRadiusTime, settings.seedMemberRadiusZ);
    }
    if (localMembers.size() < 2) {
      histogram.suppressPeak(peak);
      continue;
    }

    VertexSeed seed(settings.memoryPool);
    seed.vertex = {settings.beamX, settings.beamY, histogram.getPeakZCenter(peak)};
    seed.time = histogram.getPeakTimeInterval(peak);
    seed.scale2 = InitialScale2;

    auto fitted = fitSeed(seed, localMembers, lineRefs, lines, settings.memoryPool, settings.pairCut2);
    if (fitted.valid && fitted.contributors.size() >= 2) {
      seeds.push_back(std::move(fitted));
      histogram.suppressPeakNeighborhood(peak);
    } else {
      histogram.suppressPeak(peak);
    }
  }

  return seeds;
}

void assignLinesToSeeds(bounded_vector<VertexSeed>& seeds,
                        std::span<const int> members,
                        std::span<const LineRef> lineRefs,
                        std::span<const Line> lines,
                        const float pairCut2)
{
  for (auto& seed : seeds) {
    seed.assigned.clear();
  }

  for (const auto lineRefIdx : members) {
    const auto lineIdx = lineRefs[lineRefIdx].lineIndex;
    const auto& line = lines[lineIdx];

    int bestSeed = -1;
    float bestScore = std::numeric_limits<float>::max();
    size_t bestMult = 0;
    float bestZResidual = std::numeric_limits<float>::max();

    for (int seedIdx = 0; seedIdx < static_cast<int>(seeds.size()); ++seedIdx) {
      const auto& seed = seeds[seedIdx];
      if (!seed.valid || seed.contributors.size() < 2) {
        continue;
      }
      if (!line.mTime.isCompatible(seed.time)) {
        continue;
      }

      const auto distance2 = Line::getDistance2FromPoint(line, seed.vertex);
      if (distance2 >= pairCut2) {
        continue;
      }

      const auto score = distance2 / std::max(seed.scale2, MinScale2);
      const auto zResidual = o2::gpu::GPUCommonMath::Abs(lineRefs[lineRefIdx].zBeam - seed.vertex[2]);
      const auto multiplicity = seed.contributors.size();

      const auto betterScore = score + TieTolerance < bestScore;
      const auto betterMultiplicity = o2::gpu::GPUCommonMath::Abs(score - bestScore) <= TieTolerance && multiplicity > bestMult;
      const auto betterZ = o2::gpu::GPUCommonMath::Abs(score - bestScore) <= TieTolerance &&
                           multiplicity == bestMult && zResidual + constants::Tolerance < bestZResidual;
      if (betterScore || betterMultiplicity || betterZ) {
        bestSeed = seedIdx;
        bestScore = score;
        bestMult = multiplicity;
        bestZResidual = zResidual;
      }
    }

    if (bestSeed >= 0) {
      seeds[bestSeed].assigned.push_back(lineRefIdx);
    }
  }
}

ClusterLines materializeCluster(const VertexSeed& seed,
                                std::span<const LineRef> lineRefs,
                                std::span<const Line> lines,
                                const std::shared_ptr<BoundedMemoryResource>& mr)
{
  bounded_vector<int> lineIndices{mr.get()};
  lineIndices.reserve(seed.contributors.size());
  for (const auto lineRefIdx : seed.contributors) {
    lineIndices.push_back(lineRefs[lineRefIdx].lineIndex);
  }
  std::sort(lineIndices.begin(), lineIndices.end());
  lineIndices.erase(std::unique(lineIndices.begin(), lineIndices.end()), lineIndices.end());

  if (lineIndices.size() < 2) {
    return {};
  }

  return {std::span<const int>{lineIndices.data(), lineIndices.size()}, lines};
}

} // namespace

bounded_vector<ClusterLines> buildClusters(std::span<const Line> lines, const Settings& settings)
{
  bounded_vector<ClusterLines> clusters(settings.memoryPool.get());
  if (lines.size() < 2) {
    return clusters;
  }

  bounded_vector<LineRef> refs(settings.memoryPool.get());
  refs.reserve(lines.size());
  for (int lineIdx = 0; lineIdx < static_cast<int>(lines.size()); ++lineIdx) {
    LineRef ref(lines[lineIdx], lineIdx, settings.beamX, settings.beamY, settings.maxZ);
    if (!ref.isDead()) {
      refs.push_back(ref);
    }
  }

  if (refs.size() < 2) {
    return clusters;
  }

  const auto coarseClusters = buildCoarseClusters(refs, lines, settings);

  for (const auto& members : coarseClusters) {
    auto seeds = buildSeeds(members, refs, lines, settings);
    if (seeds.empty()) {
      continue;
    }

    for (auto& seed : seeds) {
      if (!seed.isUsableSeed()) {
        seed.valid = false;
        continue;
      }
      auto contributors = collectCompatibleContributors(seed, members, refs, lines, settings.memoryPool, settings.pairCut2);
      if (contributors.size() < 2) {
        seed.valid = false;
        continue;
      }
      seed.contributors = std::move(contributors);
    }
    compactSeeds(seeds);
    if (seeds.empty()) {
      continue;
    }
    deduplicateSeeds(seeds, settings);
    if (seeds.empty()) {
      continue;
    }
    assignLinesToSeeds(seeds, members, refs, lines, settings.pairCut2);
    for (auto& seed : seeds) {
      if (seed.assigned.size() < 2) {
        seed.valid = false;
        continue;
      }
      seed = fitSeed(seed, seed.assigned, refs, lines, settings.memoryPool, settings.pairCut2);
      if (!seed.isUsableSeed()) {
        seed.valid = false;
        continue;
      }
    }
    compactSeeds(seeds);
    deduplicateRefittedSeeds(seeds, settings);
    for (auto& refit : seeds) {
      auto cluster = materializeCluster(refit, refs, lines, settings.memoryPool);
      if (cluster.getSize() < 2) {
        continue;
      }
      if (!cluster.isValid()) {
        continue;
      }
      clusters.push_back(std::move(cluster));
    }
  }

  return clusters;
}

} // namespace o2::its::line_vertexer
