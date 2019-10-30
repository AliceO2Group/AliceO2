// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <gpucf/algorithms/cpu.h>
#include <gpucf/common/LabelContainer.h>
#include <gpucf/common/log.h>
#include <gpucf/common/RowMap.h>
#include <gpucf/common/SectorMap.h>
#include <gpucf/common/serialization.h>
#include <gpucf/common/TpcHitPos.h>
#include <gpucf/noisesuppression/NoiseSuppression.h>
#include <gpucf/noisesuppression/NoiseSuppressionOverArea.h>
#include <gpucf/noisesuppression/NoNoiseSuppression.h>
#include <gpucf/noisesuppression/plot.h>
#include <gpucf/noisesuppression/utils.h>

#include <args/args.h>

#include <memory>
#include <vector>

using namespace gpucf;

class QmaxCutoff : public NoiseSuppression
{

 public:
  QmaxCutoff(int cutoff)
    : NoiseSuppression("Filtere qmax < " + std::to_string(cutoff)), cutoff(cutoff)
  {
  }

 protected:
  std::vector<Digit> runImpl(
    View<Digit> digits,
    const Map<bool>&,
    const Map<float>&)
  {
    std::vector<Digit> filtered;
    for (const Digit& d : digits) {
      if (d.charge > cutoff) {
        filtered.push_back(d);
      }
    }

    return filtered;
  }

 private:
  int cutoff;
};

size_t countTracks(const SectorMap<LabelContainer>& labels)
{
  size_t tracks = 0;
  for (const LabelContainer& container : labels) {
    tracks += container.countTracks();
  }
  return tracks;
}

std::vector<int> countPeaksPerTrack(
  const SectorMap<RowMap<std::vector<Digit>>>& peaks,
  const SectorMap<LabelContainer>& labels)
{
  std::unordered_map<TpcHitPos, int> trackToPeaknum;

  for (short sector = 0; sector < TPC_SECTORS; sector++) {
    for (short row = 0; row < TPC_NUM_OF_ROWS; row++) {
      for (const Digit& p : peaks[sector][row]) {
        for (const MCLabel& label : labels[sector][p]) {
          trackToPeaknum[{sector, row, label}]++;
        }
      }
    }
  }

  log::Debug() << "total hits = " << trackToPeaknum.size();

  int maxPeaks = 0;
  for (auto& p : trackToPeaknum) {
    maxPeaks = std::max(maxPeaks, p.second);
  }

  log::Debug() << "maxPeaks = " << maxPeaks;

  std::vector<int> peaknumToTracknum(maxPeaks + 1);
  for (auto& p : trackToPeaknum) {
    peaknumToTracknum[p.second]++;
  }

  log::Debug() << "# hits with one peak: " << peaknumToTracknum[1];

  return peaknumToTracknum;
}

std::unordered_map<TpcHitPos, std::vector<Digit>> sortPeaksByHit(
  const SectorMap<LabelContainer>& labels,
  const SectorMap<RowMap<std::vector<Digit>>>& peaks)
{
  std::unordered_map<TpcHitPos, std::vector<Digit>> hits;
  for (short sector = 0; sector < TPC_SECTORS; sector++) {
    for (short row = 0; row < TPC_NUM_OF_ROWS; row++) {
      for (const Digit& peak : peaks[sector][row]) {
        for (const MCLabel& label : labels[sector][peak]) {
          hits[{sector, row, label}].push_back(peak);
        }
      }
    }
  }

  return hits;
}

void countLostHits(
  const SectorMap<LabelContainer>& labels,
  const std::vector<std::string>& names,
  const std::vector<SectorMap<RowMap<std::vector<Digit>>>>& peaks,
  size_t baseline)
{
  ASSERT(names.size() == peaks.size());
  ASSERT(baseline <= names.size());

  std::vector<std::unordered_map<TpcHitPos, std::vector<Digit>>> hits(names.size());
  for (size_t i = 0; i < names.size(); i++) {
    hits[i] = sortPeaksByHit(labels, peaks[i]);
  }

  for (size_t i = 0; i < names.size(); i++) {
    size_t lostHits = 0;
    size_t hitsWithOnePeak = 0;
    size_t hitsWithTwoPeaksOverlap = 0;
    size_t hitsWithTwoPeaksNoOverlap = 0;
    size_t hitsWithTenPeaks = 0;
    size_t hitsWithMoreThanPeaks = 0;

    for (const auto& hit : hits[baseline]) {
      auto hitWithPeaks = hits[i].find(hit.first);

      if (hitWithPeaks == hits[i].end()) {
        lostHits++;
      } else {
        const std::vector<Digit>& peaks = hitWithPeaks->second;
        size_t peaknum = peaks.size();

        if (peaknum == 2) {
          bool overlap = peaksOverlap(
            peaks[0],
            peaks[1],
            labels[hitWithPeaks->first.sector]);

          hitsWithTwoPeaksOverlap += overlap;
          hitsWithTwoPeaksNoOverlap += !overlap;
        } else {
          hitsWithOnePeak += (peaknum == 1);
          hitsWithTenPeaks += (peaknum > 2 && peaknum <= 10);
          hitsWithMoreThanPeaks += (peaknum > 10);
        }
      }
    }

    float totalHits = hits[baseline].size();

    log::Info() << names[i] << ":\n"
                << "  lost hits              : "
                << lostHits / totalHits << "\n"
                << "  1 peak            / hit: "
                << hitsWithOnePeak / totalHits << "\n"
                << "  2 peaks (overlap) / hit: "
                << hitsWithTwoPeaksOverlap / totalHits << "\n"
                << "  2 peaks           / hit: "
                << hitsWithTwoPeaksNoOverlap / totalHits << "\n"
                << "  3-10 peaks        / hit: "
                << hitsWithTenPeaks / totalHits << "\n"
                << "  > 10 peaks        / hit: "
                << hitsWithMoreThanPeaks / totalHits;
  }
}

int main(int argc, const char* argv[])
{
  args::ArgumentParser parser("");

  args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});

  args::ValueFlag<std::string> digitfile(parser, "D", "Digit file", {'d', "digits"});
  args::ValueFlag<std::string> labelfile(parser, "L", "Label file", {'l', "labels"});

  try {
    parser.ParseCLI(argc, argv);
  } catch (const args::Help&) {
    std::cerr << parser;
    std::exit(1);
  }

  float epsilon = 10;

  std::vector<std::unique_ptr<NoiseSuppression>> noiseSuppressionAlgos;
  noiseSuppressionAlgos.emplace_back(new NoNoiseSuppression);
  /* noiseSuppressionAlgos.emplace_back(new QmaxCutoff(2)); */
  noiseSuppressionAlgos.emplace_back(new QmaxCutoff(3));
  /* noiseSuppressionAlgos.emplace_back(new QmaxCutoff(9)); */
  /* noiseSuppressionAlgos.emplace_back(new NoiseSuppressionOverArea(2, 2, 3, 1025)); */
  /* noiseSuppressionAlgos.emplace_back(new NoiseSuppressionOverArea(2, 3, 3, 1025)); */
  /* noiseSuppressionAlgos.emplace_back(new NoiseSuppressionOverArea(3, 3, 3, 1025)); */
  /* noiseSuppressionAlgos.emplace_back(new NoiseSuppressionOverArea(3, 4, 3, 1025)); */
  noiseSuppressionAlgos.emplace_back(new NoiseSuppressionOverArea(2, 2, 3, epsilon));
  noiseSuppressionAlgos.emplace_back(new NoiseSuppressionOverArea(2, 3, 3, epsilon));
  noiseSuppressionAlgos.emplace_back(new NoiseSuppressionOverArea(3, 3, 3, epsilon));
  noiseSuppressionAlgos.emplace_back(new NoiseSuppressionOverArea(3, 4, 3, epsilon));
  /* noiseSuppressionAlgos.emplace_back(new NoiseSuppressionOverArea(2, 4, 3)); */
  /* noiseSuppressionAlgos.emplace_back(new NoiseSuppressionOverArea(1, 4, 3)); */
  /* noiseSuppressionAlgos.emplace_back(new NoiseSuppressionOverArea(0, 4, 3)); */

  size_t baseline = 0; // Index of algorithm thats used as baseline when looking for lost hits

  // map algorithm id -> result of algorithm
  std::vector<SectorMap<RowMap<std::vector<Digit>>>> filteredPeaks(
    noiseSuppressionAlgos.size());

  SectorMap<std::vector<Digit>> digits;
  {
    log::Info() << "Reading digit file " << args::get(digitfile);
    SectorMap<std::vector<RawDigit>> rawdigits =
      gpucf::read<RawDigit>(args::get(digitfile));
    digits = Digit::bySector(rawdigits);
  }

  size_t totalDigits = 0;
  for (size_t sector = 0; sector < TPC_SECTORS; sector++) {
    totalDigits += digits[sector].size();
  }

  log::Debug() << "num of digits = " << totalDigits;

  SectorMap<LabelContainer> labels;
  {
    log::Info() << "Reading label file " << args::get(labelfile);
    SectorMap<std::vector<RawLabel>> rawlabels =
      gpucf::read<RawLabel>(args::get(labelfile));
    labels = LabelContainer::bySector(rawlabels, digits);
  }

  {
    log::Info() << "Found " << countTracks(labels) << " tracks in label data.";
    /* log::Info() << "... generating " << countHits(labels) << " hits."; */

    log::Info() << "Creating chargemap";
    SectorMap<Map<float>> chargemaps;
    for (size_t sector = 0; sector < TPC_SECTORS; sector++) {
      chargemaps[sector] = Map<float>(
        digits[sector],
        [](const Digit& d) { return d.charge; },
        0.f);
    }

    for (size_t sector = 0; sector < TPC_SECTORS; sector++) {
      log::Info() << "Processing sector " << sector;
      RowMap<std::vector<Digit>> peaks =
        findPeaksByRow(digits[sector], chargemaps[sector]);

      RowMap<Map<bool>> peakmap = makePeakMapByRow(peaks);

      for (size_t id = 0; id < noiseSuppressionAlgos.size(); id++) {
        auto& algo = noiseSuppressionAlgos[id];
        filteredPeaks[id][sector] =
          algo->run(peaks, peakmap, chargemaps[sector]);
      }
    }
  }

  // map algorithm id, N -> num of tracks with N peaks (in a row)
  std::vector<std::vector<int>> peaknumToTracknum(
    noiseSuppressionAlgos.size());
  for (size_t id = 0; id < noiseSuppressionAlgos.size(); id++) {
    peaknumToTracknum[id] = countPeaksPerTrack(filteredPeaks[id], labels);
  }

  std::vector<std::string> names;
  for (auto& algo : noiseSuppressionAlgos) {
    names.push_back(algo->getName());
  }

  PlotConfig cnf;
  /* cnf.logYAxis = true; */
  cnf.logXAxis = true;
  plot(names, peaknumToTracknum, "peaknumToHits_" + std::to_string(epsilon) + ".pdf",
       "# peaks", "# hits", cnf);

  countLostHits(
    labels,
    names,
    filteredPeaks,
    baseline);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
