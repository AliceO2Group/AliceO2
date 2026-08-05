// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"

#include "TSystem.h"
#include "Algorithm/RangeTokenizer.h"
#include "Framework/Logger.h"
#include "CommonConstants/LHCConstants.h"
#include "SpacePoints/SpacePointsCalibConfParam.h"
#include "SpacePoints/TrackResiduals.h"
#include "SpacePoints/TrackInterpolation.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "DataFormatsTPC/Defs.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "DetectorsBase/Propagator.h"
#include "TPCBase/Mapper.h"
#include "ReconstructionDataFormats/TrackUtils.h"

#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include "TTreePerfStats.h"
#include <TChain.h>
#include <TGeoManager.h>
#include <TGrid.h>
#include <TH2.h>
#include <TF1.h>

#include <fmt/format.h>
#include <filesystem>
#include <fstream>
#include <sys/resource.h>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>

#include <ctime>
#include <iostream>
#include <chrono>
#include <sstream>
#include <cstdio>
#include <unistd.h>

// For multiple threads
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <deque>
#include "TROOT.h"

// Compile-time detection of TrackData::filterFlag / UnbinnedResid::rejected, for back-compat with O2
// builds that predate them.
template <typename T, typename = void>
struct HasFilterFlagMember : std::false_type {
};
template <typename T>
struct HasFilterFlagMember<T, std::void_t<decltype(std::declval<T>().filterFlag)>> : std::true_type {
};

template <typename T, typename = void>
struct HasRejectedMember : std::false_type {
};
template <typename T>
struct HasRejectedMember<T, std::void_t<decltype(std::declval<T>().rejected)>> : std::true_type {
};

template <typename T>
bool hasPositiveFilterFlag(const T& t)
{
  if constexpr (HasFilterFlagMember<T>::value) {
    return t.filterFlag > 0;
  } else {
    return false;
  }
}

template <typename T>
bool isRejectedResidual(const T& t)
{
  if constexpr (HasRejectedMember<T>::value) {
    return t.rejected;
  } else {
    return false;
  }
}

#else

#error This macro must run in compiled mode

#endif

using namespace o2::tpc;
using GID = o2::dataformats::GlobalTrackID;
namespace fs = std::filesystem;

constexpr int NSectors = SECTORSPERSIDE * SIDES;
constexpr int NRows = Mapper::PADROWS;

// Portable peak-RSS report via getrusage(), which works on both Linux and macOS (unlike parsing
// /proc/self/status, which is Linux-only procfs). ru_maxrss's UNIT differs by platform though (bytes on
// macOS, KB on Linux) -- handled below. Only reports peak resident-set size (one number), not the
// separate VmRSS/VmPeak/VmSize that /proc/self/status exposes on Linux.
void printMemoryUsage(const std::string& label = "")
{
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) != 0) {
    return;
  }
#ifdef __APPLE__
  const double maxRssGB = ru.ru_maxrss / 1024.0 / 1024.0 / 1024.0; // macOS: ru_maxrss in bytes
#else
  const double maxRssGB = ru.ru_maxrss / 1024.0 / 1024.0; // Linux: ru_maxrss in KB
#endif
  if (label.empty()) {
    LOGP(info, "VmPeakRSS {:.2f} GB", maxRssGB);
  } else {
    LOGP(info, "[{}] VmPeakRSS {:.2f} GB", label, maxRssGB);
  }
}

template <typename T>
double calculateMean(const std::vector<T>& vec)
{
  if (vec.empty()) {
    LOGP(error, "vector is empty");
    return 0.;
  }

  return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

template <typename T>
double calculateMedian(std::vector<T> vec)
{
  if (vec.empty()) {
    LOGP(error, "vector is empty");
    return 0.;
  }

  const size_t size = vec.size();
  const size_t midIndex = size / 2;

  std::nth_element(vec.begin(), vec.begin() + midIndex, vec.end());

  if (size % 2 != 0) {
    return vec[midIndex];
  }

  return (vec[midIndex - 1] + vec[midIndex]) / 2.0;
}

// Lightweight double-precision 3-vector for the hot per-residual/per-voxel-flush loop, ~10x faster
// than TVector3 here (0.550s -> 0.054s over 20M calls) for numerically identical math -- matters
// because getIntCircles() runs while the per-voxel mutex is held. Double (not float) because this is
// live geometric computation (sqrt- and division-heavy) rather than storage -- contrast Vec3f below,
// which is storage-only and fine in single precision.
struct Vec3d {
  double x = 0.0, y = 0.0, z = 0.0;
  double X() const { return x; }
  double Y() const { return y; }
  double Z() const { return z; }
  void SetXYZ(double xx, double yy, double zz)
  {
    x = xx;
    y = yy;
    z = zz;
  }
  double Perp() const { return std::sqrt(x * x + y * y); }
  double Mag() const { return std::sqrt(x * x + y * y + z * z); }
  void RotateZ(double angle)
  {
    const double c = std::cos(angle), s = std::sin(angle);
    const double xn = x * c - y * s, yn = x * s + y * c;
    x = xn;
    y = yn;
  }
  Vec3d& operator-=(const Vec3d& o)
  {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    return *this;
  }
  Vec3d& operator+=(const Vec3d& o)
  {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
  }
  Vec3d& operator*=(double a)
  {
    x *= a;
    y *= a;
    z *= a;
    return *this;
  }
};
inline Vec3d operator-(const Vec3d& a, const Vec3d& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3d operator*(const Vec3d& a, double s) { return {a.x * s, a.y * s, a.z * s}; }

//------------------------------------------------------------------------------------------------------------
// Crossing point of two circles in the transverse plane.
//
// Two sentinel returns, both of which callers reject via their own transverse-radius band cut:
//   (9999, 9999, 9999) -- the circles do not intersect at all (Perp() far above any accepted radius)
//   (0, 0, 0)          -- they intersect, but neither solution falls in the valid TPC radial band below
// A third, implicit outcome is NaN: when d lands within a few ulps of a tangency bound (d == r1+r2 or
// d == |r1-r2|) the guards below still pass, but rounding can drive the sqrt argument marginally
// negative, since a == r1 exactly at both bounds in exact arithmetic. NaN then fails the callers'
// band cut too (every comparison against it is false), so such a degenerate pair is simply dropped --
// which is the wanted behaviour, hence no clamping here.
Vec3d getIntCircles(double r1, double r2, Vec3d circleCenter1, Vec3d circleCenter2, double voxX, double voxY)
{
  Vec3d vecSp; // default-constructed to (0, 0, 0), which is the "no solution accepted" sentinel

  const Vec3d dVec = (circleCenter1 - circleCenter2);
  const double d = dVec.Perp(); // dist. circle center 1 & 2

  if (d > r1 + r2) {
    // no solutions, the circles are separate
    vecSp.SetXYZ(9999, 9999, 9999);
    return vecSp;
  }
  if (d < std::fabs(r1 - r2)) {
    // no solutions, one circle is contained in the other
    vecSp.SetXYZ(9999, 9999, 9999);
    return vecSp;
  }
  if (d < 0.0001) {
    // no solutions, same circle center
    vecSp.SetXYZ(9999, 9999, 9999);
    return vecSp;
  }

  const double dx = circleCenter2.X() - circleCenter1.X();
  const double dy = circleCenter2.Y() - circleCenter1.Y();

  const double a = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
  const double h = std::sqrt(r1 * r1 - a * a);

  // Midpoint of the chord joining the two intersection points. Deliberately written as a reciprocal
  // multiply rather than `a * dx / d`: floating-point reciprocal-then-multiply and multiply-then-divide
  // are not guaranteed to give the same last-bit result. Keep this exact form.
  const double Mx = circleCenter1.X() + (1 / d) * a * dx;
  const double My = circleCenter1.Y() + (1 / d) * a * dy;

  const double sp1x = Mx + h * dy / d;
  const double sp2x = Mx - h * dy / d;

  const double sp1y = My - h * dx / d;
  const double sp2y = My + h * dx / d;

  // The two solutions take the z of the circle they came from, not a common z.
  const double sp1z = circleCenter1.Z();
  const double sp2z = circleCenter2.Z();

  const double sp1Perp = std::sqrt(sp1x * sp1x + sp1y * sp1y);
  const double sp2Perp = std::sqrt(sp2x * sp2x + sp2y * sp2y);

  const bool sp1In = (sp1Perp > 60.0 && sp1Perp < 280.0);
  const bool sp2In = (sp2Perp > 60.0 && sp2Perp < 280.0);

  if (sp1In && sp2In) {
    // Both solutions are physically plausible -- pick whichever is closer to the voxel center rather
    // than an arbitrary, data-independent choice that could just as easily discard the better of two
    // valid solutions.
    const double d1 = std::sqrt((sp1x - voxX) * (sp1x - voxX) + (sp1y - voxY) * (sp1y - voxY));
    const double d2 = std::sqrt((sp2x - voxX) * (sp2x - voxX) + (sp2y - voxY) * (sp2y - voxY));
    if (d1 <= d2) {
      vecSp.SetXYZ(sp1x, sp1y, sp1z);
    } else {
      vecSp.SetXYZ(sp2x, sp2y, sp2z);
    }
  } else if (sp1In) {
    vecSp.SetXYZ(sp1x, sp1y, sp1z);
  } else if (sp2In) {
    vecSp.SetXYZ(sp2x, sp2y, sp2z);
  }
  // else: neither in band, vecSp stays at its default (0,0,0) sentinel

  return vecSp;
}
//------------------------------------------------------------------------------------------------------------

struct range {
  long from{-1};
  long to{-1};

  bool operator<(const range& other)
  {
    return from < other.from;
  }

  void sort()
  {
    if (from > to) {
      std::swap(from, to);
    }
  }
};

// ---- Fastest-replica selection for alien:// residual files (used by getInputFileList below) ----
// Different Storage Elements serving the same LFN can have very different real-world throughput
// depending on where this job actually lands on the network (verified: ALICE::FZK::SE faster than
// ALICE::CERN::EOS from one machine, the reverse from another) -- a name-based heuristic can't predict
// that, so probe with a real timed read instead. Residual files are O(10GB), so getting this wrong once
// costs far more than the probe itself.
struct SEReplica {
  std::string se;
  std::string pfn;
};

std::vector<SEReplica> getAlienReplicas(const std::string& plainLFN)
{
  std::vector<SEReplica> result;
  std::string cmd = "alien.py whereis -r " + plainLFN + " 2>/dev/null";
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
  if (!pipe) {
    return result;
  }
  std::array<char, 512> buffer;
  std::string output;
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    output += buffer.data();
  }

  auto trim = [](std::string& s) {
    s.erase(0, s.find_first_not_of(" \t"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
  };
  std::istringstream iss(output);
  std::string line;
  while (std::getline(iss, line)) {
    auto sePos = line.find("SE =>");
    auto pfnPos = line.find("pfn =>");
    if (sePos == std::string::npos || pfnPos == std::string::npos) {
      continue;
    }
    std::string se = line.substr(sePos + 5, pfnPos - sePos - 5);
    std::string pfn = line.substr(pfnPos + 6);
    trim(se);
    trim(pfn);
    result.push_back({se, pfn});
  }
  return result;
}

// Generates a tiny standalone probe macro on disk (ROOT requires the file stem to match the top-level
// function name) that opens one alien:// replica and reads a few real entries from the 'unbinnedResid'
// tree (the same tree/branch doFileProcessing itself reads, with the same two dominant unused
// sub-branches disabled -- see there), timing the real transfer via TFile::GetBytesRead(). Prints
// "PROBE_OK <bytesRead> <seconds>" or "PROBE_FAIL" to stdout -- this is invoked as a subprocess by
// probeOneReplicaSubprocess below, never called in-process.
std::string writeProbeChildMacro(const std::string& funcName)
{
  static const char* templateSrc =
    "#include <TFile.h>\n"
    "#include <TGrid.h>\n"
    "#include <TTreeReader.h>\n"
    "#include <TTreeReaderValue.h>\n"
    "#include \"SpacePoints/TrackInterpolation.h\"\n"
    "#include <vector>\n"
    "#include <chrono>\n"
    "#include <cstdio>\n"
    "#include <memory>\n"
    "#include <string>\n"
    "\n"
    "void @FUNC@(const char* plainLFN, const char* se, Long64_t probeEntries = 3)\n"
    "{\n"
    "  // This runs in a fresh, separate ROOT process (see probeOneReplicaSubprocess) -- unlike the parent\n"
    "  // process, gGrid is never already set up here, so it must be connected explicitly.\n"
    "  if (!gGrid && !TGrid::Connect(\"alien://\")) {\n"
    "    printf(\"PROBE_FAIL\\n\");\n"
    "    return;\n"
    "  }\n"
    "  std::string url = std::string(\"alien://\") + plainLFN + \"?se=\" + se;\n"
    "  std::unique_ptr<TFile> f(TFile::Open(url.c_str(), \"READ\"));\n"
    "  if (!f || f->IsZombie()) { printf(\"PROBE_FAIL\\n\"); return; }\n"
    "  TTreeReader reader(\"unbinnedResid\", f.get());\n"
    "  TTree* residTree = reader.GetTree();\n"
    "  if (!residTree) { printf(\"PROBE_FAIL\\n\"); return; }\n"
    "  TTreeReaderValue<std::vector<o2::tpc::UnbinnedResid>> res(reader, \"res\");\n"
    "  residTree->SetBranchStatus(\"res.tgSlp\", 0);\n"
    "  residTree->SetBranchStatus(\"res.channel\", 0);\n"
    "\n"
    "  Long64_t bytesBefore = f->GetBytesRead();\n"
    "  auto t0 = std::chrono::steady_clock::now();\n"
    "  Long64_t nRead = 0;\n"
    "  for (Long64_t i = 0; i < probeEntries && reader.Next(); ++i) {\n"
    "    if (static_cast<int>(res.GetSetupStatus()) < 0) break;\n"
    "    (void)res->size();\n"
    "    ++nRead;\n"
    "  }\n"
    "  auto t1 = std::chrono::steady_clock::now();\n"
    "  Long64_t bytesRead = f->GetBytesRead() - bytesBefore;\n"
    "  if (nRead == 0 || bytesRead <= 0) { printf(\"PROBE_FAIL\\n\"); return; }\n"
    "  double sec = std::chrono::duration<double>(t1 - t0).count();\n"
    "  printf(\"PROBE_OK %lld %f\\n\", (long long)bytesRead, sec);\n"
    "}\n";

  std::string src(templateSrc);
  const std::string placeholder = "@FUNC@";
  auto pos = src.find(placeholder);
  if (pos != std::string::npos) {
    src.replace(pos, placeholder.size(), funcName);
  }

  const std::string path = "/tmp/" + funcName + ".C";
  std::ofstream out(path);
  out << src;
  out.close();
  return path;
}

// Probes one candidate replica in a SEPARATE OS PROCESS, bounded by the `timeout` command, so a genuine
// hang (not just a slow-but-working transfer) can be killed without touching this process's own
// TGrid/JAlien connection. An in-process std::async-based timeout was tried and rejected:
// std::future's destructor from std::launch::async BLOCKS until the task finishes regardless of what
// wait_for() returned, so it doesn't actually bound wall-clock time -- and leaking the future to dodge
// that reopens a real concurrent-JAlien-access crash. A real OS process boundary is the only mechanism
// that is both a genuine timeout AND safe against that crash.
//
// Confirmed necessary by a real GRID failure: a job hung completely (~1% CPU for 15 minutes, no
// progress) inside an in-process probe's first candidate until AliEn's idle-CPU watchdog killed the
// whole job. A bounded probe lets it fall through to the next candidate instead of losing the slot.
bool probeOneReplicaSubprocess(const std::string& plainLFN, const std::string& se, double& bytesPerSec,
                               int timeoutSec = 30, Long64_t probeEntries = 3)
{
  bytesPerSec = -1.0;
  const std::string funcName = fmt::format("probeSEChild{}", static_cast<long>(getpid()));
  const std::string macroPath = writeProbeChildMacro(funcName);

  const std::string cmd = fmt::format(
    "timeout {}s root.exe -b -q -l -x '{}(\"{}\", \"{}\", {})' 2>&1",
    timeoutSec, macroPath, plainLFN, se, probeEntries);

  std::string output;
  {
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (pipe) {
      std::array<char, 512> buffer;
      while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        output += buffer.data();
      }
    }
  }
  std::remove(macroPath.c_str());

  // Parse line-by-line rather than a single sscanf over the whole output -- ROOT's own startup banner
  // and "Info in <TFile::Open>" lines precede the actual PROBE_OK/PROBE_FAIL line.
  std::istringstream iss(output);
  std::string line;
  while (std::getline(iss, line)) {
    long long bytesRead = 0;
    double sec = 0.0;
    if (sscanf(line.c_str(), "PROBE_OK %lld %lf", &bytesRead, &sec) == 2) {
      if (sec > 0 && bytesRead > 0) {
        bytesPerSec = static_cast<double>(bytesRead) / sec;
        return true;
      }
      return false;
    }
    if (line.rfind("PROBE_FAIL", 0) == 0) {
      return false;
    }
  }
  return false; // empty/no matching line -- timed out (killed by `timeout`) or crashed
}

// Probes each candidate replica in turn (via probeOneReplicaSubprocess above) and returns the SE with
// the highest measured throughput, or "" if every candidate failed or timed out.
std::string probeFastestSE(const std::string& plainLFN, const std::vector<SEReplica>& replicas,
                           Long64_t probeEntries = 3, int timeoutSec = 30)
{
  std::string bestSE;
  double bestBytesPerSec = -1.0;
  for (const auto& r : replicas) {
    double bps = -1.0;
    if (!probeOneReplicaSubprocess(plainLFN, r.se, bps, timeoutSec, probeEntries)) {
      LOGP(warning, "SE probe: {} failed or timed out after {}s", r.se, timeoutSec);
      continue;
    }
    LOGP(info, "SE probe: {} -> {:.1f} MB/s", r.se, bps / (1024.0 * 1024.0));
    if (bps > bestBytesPerSec) {
      bestBytesPerSec = bps;
      bestSE = r.se;
    }
  }
  return bestSE;
}

std::vector<range> loadRunTimeSpans(const std::string& flname, int onlyRun, const std::string& selection);
std::vector<std::string> getInputFileList(const std::string& fileInput)
{
  std::vector<std::string> fileList;
  std::vector<std::string> fileListVerified;
  // check if only one input file (a txt file contaning a list of files is provided)
  if (fileInput.length() > 3 && fileInput.substr(fileInput.length() - 3, 3) == "txt") {
    LOGP(info, "Reading files from input file list {}", fileInput);
    std::ifstream is(fileInput);
    std::istream_iterator<std::string> start(is);
    std::istream_iterator<std::string> end;
    fileList.insert(fileList.begin(), start, end);
  } else {
    fileList.push_back(fileInput);
  }

  // fastestSE is probed once, for the first alien:// file, and reused for the rest of this slot's files
  // (same production run -> same SE set, in practice), avoiding a ~10GB-file probe cost per file. If a
  // later file isn't actually hosted on the cached SE, doFileProcessing's own open has a fallback to
  // unforced resolution -- so a stale cache can't silently drop a file, only cost a little speed for
  // that one file.
  std::string fastestSE;
  for (auto file : fileList) {
    if ((file.find("alien://") == 0) && !gGrid && !TGrid::Connect("alien://")) {
      LOGP(fatal, "Failed to open alien connection");
    }
    if (gSystem->Getenv("FORCESE") && !TString(file.data()).EndsWith(gSystem->Getenv("FORCESE"))) {
      file += "?se=";
      file += gSystem->Getenv("FORCESE");
    } else if (!gSystem->Getenv("FORCESE") && file.rfind("alien://", 0) == 0) {
      if (fastestSE.empty()) {
        std::string plainLFN = file.substr(std::string("alien://").size());
        auto slash = plainLFN.find_first_not_of('/');
        plainLFN = (slash == std::string::npos) ? "/" : "/" + plainLFN.substr(slash);
        auto replicas = getAlienReplicas(plainLFN);
        if (!replicas.empty()) {
          fastestSE = (replicas.size() == 1) ? replicas.front().se : probeFastestSE(plainLFN, replicas);
          if (!fastestSE.empty()) {
            LOGP(info, "Auto-selected fastest SE {} for this slot's alien:// files", fastestSE);
          }
        }
      }
      if (!fastestSE.empty()) {
        file += "?se=";
        file += fastestSE;
      }
    }
    fileListVerified.push_back(file);
  }

  if (fileListVerified.size() == 0) {
    LOGP(error, "No input files to process");
  }
  return fileListVerified;
}

bool revalidateTrack(const TrackData& trk, const SpacePointsCalibConfParam& params)
{

  if (hasPositiveFilterFlag(trk)) {
    return false;
  }

  if (fabs(trk.par.getTgl()) > params.maxZ2X) {
    return false;
  }
  if (trk.nClsITS < params.minITSNCls) {
    return false;
  }
  if (trk.nClsTPC < params.minTPCNCls) {
    return false;
  }
  // No TRD-based cuts here (neither on the tracklet count nor on chi2TRD): this macro does not use TRD.
  // track quality cuts
  if (trk.chi2ITS / trk.nClsITS > params.maxITSChi2) {
    return false;
  }
  if (trk.chi2TPC / trk.nClsTPC > params.maxTPCChi2) {
    return false;
  }

  if (params.cutOnDCA) {
    auto propagator = o2::base::Propagator::Instance();
    // o2::track::TrackPar trkPar(trk.x, trk.alpha, trk.p); // use this line, in case ClassDef version of TrackData < 4
    o2::track::TrackPar trkPar = trk.par;
    if (!propagator->propagateToX(trkPar, 0, propagator->getNominalBz())) {
      return false;
    }
    if (trkPar.getX() * trkPar.getX() + trkPar.getY() * trkPar.getY() > params.maxDCA * params.maxDCA) {
      return false;
    }
  }
  return true;
}

// Check that a TTreeReaderValue was actually bound to a branch of the expected type.
// Unlike SetBranchAddress (which silently tolerated a missing/mismatching branch), dereferencing a
// TTreeReaderValue that failed to set up returns a null proxy and segfaults, so this has to be
// checked once per file before the data is used. The setup status is only final after the first
// entry has been loaded, so call this after SetEntry().
template <typename T>
bool checkReaderValue(const TTreeReaderValue<T>& value, const int iThread, const std::string& fileName)
{
  // all failure codes of ESetupStatus are negative (kSetupMatch is 0, the other success codes are positive)
  const auto status = value.GetSetupStatus();
  if (static_cast<int>(status) < 0) {
    LOGP(warning, "[Thread{}] Branch '{}' could not be set up (setup status {}) in file {}", iThread, value.GetBranchName(), static_cast<int>(status), fileName);
    return false;
  }
  return true;
}

// Pool size per voxel/charge. Compile-time rather than a runtime parameter so that circleCenters and
// circleRadii below can be fixed-size std::array instead of heap-allocated std::vector: with ~3M
// voxels in a realistic bin configuration, per-voxel heap allocations add up to roughly 4.2 GB of
// resident memory and 12M+ small mallocs for these two members alone. Changing the pool size means
// editing this line and recompiling.
constexpr int NPool = 15;

// Warm-up threshold for the no-input-map DZ rolling-average correction: minimum cumulative NP/PP/NN
// sample count (see VoxelData below) required before a voxel's residualsAll[0] (dX) is trusted enough
// to shift the Z propagation. See the call site in doFileProcessing for the full reasoning.
constexpr int MinDxSamplesForZCorr = 20;

// Storage for circleCenters: pure scratch coordinates (never serialized/drawn), populated from
// float-precision inputs (xycircle.xC/yC, track/voxel positions) in the first place, so float is
// enough -- consistent with circleRadii already being float. Converts implicitly to Vec3d at the point
// of use (getIntCircles computes in double internally regardless of the float input).
struct Vec3f {
  float x = 0.f, y = 0.f, z = 0.f;
  operator Vec3d() const { return Vec3d{x, y, z}; }
};

// shared circle pool for one voxel; one instance per voxel (not per thread), guarded by its own mutex
struct VoxelData {
  std::mutex mtx;                                        //! per-voxel mutex, shared across threads
  std::array<std::array<Vec3f, NPool>, 2> circleCenters; // [charge] was vec_TV3_circle_center_thread
  std::array<std::array<float, NPool>, 2> circleRadii;   // [charge] was vec_TV3_circle_radius_thread
  std::array<int, 2> poolCounter{};                      // [charge] was vec_counter_thread

  std::array<double, 3> residualsAll{}; // was vec_residualsAll_thread; [2] (Z) is a running sum, see counterZAll
  std::array<double, 3> residualsNP{};  // was vec_residualsNP_thread
  std::array<double, 3> residualsPP{};  // was vec_residualsPP_thread
  std::array<double, 3> residualsNN{};  // was vec_residualsNN_thread
  int counterNP = 0;                    // was vec_residuals_counterNP_thread
  int counterPP = 0;                    // was vec_residuals_counterPP_thread
  int counterNN = 0;                    // was vec_residuals_counterNN_thread
  int counterZAll = 0;                  // was vec_residuals_counterZAll_thread
};

// One TimeFrame's data, fully OWNED (copied out of the TTreeReaderValues rather than referencing them).
// TTreeReaderValue::operator* reuses the same underlying storage on every SetEntry -- a background
// producer thread reading TF N+1 into that storage while the consumer is still processing TF N's tracks
// would race and corrupt data, the same class of hazard already found once in this file (a lazy-
// deserialization race across concurrent track-worker threads). Copying the three vectors out per TF
// avoids that; the copy itself is small next to the per-TF track-processing time.
struct TFPackage {
  int iEntry = 0;
  std::vector<TrackDataCompact> trackRefsVec;
  std::vector<TrackData> trackDataVec;
  std::vector<UnbinnedResid> unbinnedResidualsVec;
};

// Bounded single-producer/single-consumer queue of TFPackages. Lets one background thread stay a few
// TimeFrames ahead of the (I/O-free -- see doFileProcessing's own track-loop-parallelism notes) track
// processing, so a TTreeCache refill on some later TF can overlap with the current TF's track-worker
// compute instead of blocking it. This is meant to hide the periodic TTreeCache-refill spikes this
// pipeline's I/O shows in practice, and only pays off paired with a moderate TTreeCache size -- too
// large a cache makes individual refills bigger than any reasonable queue depth can absorb.
class BoundedTFQueue
{
 public:
  explicit BoundedTFQueue(size_t maxDepth) : mMaxDepth(maxDepth) {}

  // Producer side. Blocks while the queue is already at capacity.
  void push(std::unique_ptr<TFPackage> pkg)
  {
    std::unique_lock<std::mutex> lock(mMtx);
    mNotFull.wait(lock, [this] { return mQueue.size() < mMaxDepth; });
    mQueue.push_back(std::move(pkg));
    lock.unlock();
    mNotEmpty.notify_one();
  }

  // Producer side, called once (file fully read, or the maxTracks quota was reached).
  void setDone()
  {
    {
      std::lock_guard<std::mutex> lock(mMtx);
      mDone = true;
    }
    mNotEmpty.notify_one();
  }

  // Consumer side. Returns nullptr once the producer is done AND the queue has been fully drained.
  std::unique_ptr<TFPackage> pop()
  {
    std::unique_lock<std::mutex> lock(mMtx);
    mNotEmpty.wait(lock, [this] { return !mQueue.empty() || mDone; });
    if (mQueue.empty()) {
      return nullptr;
    }
    auto pkg = std::move(mQueue.front());
    mQueue.pop_front();
    lock.unlock();
    mNotFull.notify_one();
    return pkg;
  }

  // Diagnostic only: how many packages are sitting ready right now. Lets the consumer log whether the
  // producer is comfortably ahead (queue usually near mMaxDepth) or struggling to keep up (queue usually
  // near empty) -- distinguishes "the buffer is the wrong depth" from "the producer itself is too slow
  // to ever fill it, no matter how deep it is".
  size_t size() const
  {
    std::lock_guard<std::mutex> lock(mMtx);
    return mQueue.size();
  }

 private:
  const size_t mMaxDepth;
  mutable std::mutex mMtx;
  std::condition_variable mNotEmpty;
  std::condition_variable mNotFull;
  std::deque<std::unique_ptr<TFPackage>> mQueue;
  bool mDone = false;
};

// How many TimeFrames the background producer may stay ahead of track-processing. Absorbs some of the
// periodic TTreeCache-refill spikes this pipeline's I/O shows in practice, though 10 (the default
// below) isn't enough to fully hide the biggest ones -- a much larger depth would be needed for that,
// at a real memory cost (a single TF can carry tens of thousands of tracks). Overridable via
// SCDCALIB_TF_QUEUE_DEPTH (no recompile) to tune against a real workload's spike size.
constexpr size_t TFQueueDepthDefault = 10;

void doFileProcessing(const int iThread,
                      const int nFileThreads,
                      const int maxTrackWorkers,
                      const long firstTFTime,
                      const long lastTFTime,
                      const bool invertBadRange,
                      const float maxdEdx,
                      const float maxdEdxExp,
                      const float maxDevdEdxOverExp,
                      const float skipEdgePads,
                      std::vector<long int>& nEdgeClustersSkipped_thread,
                      std::vector<long int>& nTracksSkippedByBadRangeList_thread,
                      std::vector<long int>& nTFs_thread,
                      std::vector<long int>& nTFsSkippedByBadRangeList_thread,
                      std::vector<long int>& nTFsSkippedByTimeWindow_thread,
                      const std::string voxMapInput,
                      const GID::mask_t sources,
                      const int64_t orbitResetTimeMS,
                      const float magfieldvalue,
                      const std::vector<std::string> fileList,
                      const int maxTracksPerSlice,
                      const Long64_t maxTracks,
                      std::atomic<Long64_t>& nTracksProcessed,
                      const std::array<std::vector<TrackResiduals::VoxRes>, NSectors>& voxelResults, // read-only input correction map, shared across threads (no per-thread copy)
                      std::vector<std::vector<range>>& badRanges_thread,
                      const TrackResiduals& trackResiduals, // read-only: findVoxelBin/getVoxelCoordinates/getGlbVoxBin are all const, safe to share across threads
                      const float maxDistIntCls,
                      const int nY2XBins,
                      const int nZ2XBins,
                      std::vector<VoxelData>& voxels, // flat [sec*152*nY2XBins*nZ2XBins + ix*nY2XBins*nZ2XBins + iy*nZ2XBins + iz] -- shared across threads, one instance per voxel
                      std::vector<Long64_t>& totalBytesReadPerf_thread,
                      std::vector<size_t>& lumiEntriesCTP_thread,
                      std::vector<double>& lumiSumCTP_thread,
                      std::vector<std::vector<uint32_t>>& orbitsSel_thread,
                      std::vector<std::vector<float>>& ctpLumiSel_thread,
                      std::vector<std::vector<long>>& timeMSsel_thread)
{
  // Get Mapper
  const Mapper& mapper = Mapper::instance();

  // yMaxCentrePadByRow only depends on the pad row (152 values) -- precomputed once per file-thread
  // here, gated on skipEdgePads since that's its only use, rather than via a Mapper lookup per residual.
  std::array<float, NRows> yMaxCentrePadByRow{};
  if (skipEdgePads) {
    for (int irow = 0; irow < NRows; ++irow) {
      yMaxCentrePadByRow[irow] = mapper.getPadCentre(o2::tpc::PadPos(irow, 0)).Y() - mapper.getPadRegionInfo(o2::tpc::Mapper::REGION[irow]).getPadWidth() / 2;
    }
  }

  // Obtain configuration per thread
  const SpacePointsCalibConfParam& params_thread = SpacePointsCalibConfParam::Instance();

  // Per-thread input handles and I/O-monitoring state. These are plain locals: each file-thread only ever
  // touches its own, so there is nothing to share with the other threads or hand back to the caller.
  //
  // DECLARATION ORDER IS LOAD-BEARING. Locals are destroyed in reverse order of declaration, and these
  // objects reference each other: a TTreeReaderValue refers to its TTreeReader, which refers to a TTree
  // owned by the TFile, and TTreePerfStats refers to that tree too. Declaring the file first and the
  // reader values last therefore tears them down in the only safe order -- values, then perf stats, then
  // readers, then the file. Do not reorder these.
  std::unique_ptr<TFile> inputFile;
  std::unique_ptr<TTreeReader> treeUnbinnedResiduals;
  std::unique_ptr<TTreeReader> treeTrackData;
  std::unique_ptr<TTreeReader> treeRecords;
  std::unique_ptr<TTreePerfStats> perfStats;
  std::unique_ptr<TTreeReaderValue<std::vector<UnbinnedResid>>> unbinnedResiduals; // unbinned residuals input
  std::unique_ptr<TTreeReaderValue<std::vector<TrackDataCompact>>> trackRefs;      // track references for the unbinned residuals
  std::unique_ptr<TTreeReaderValue<std::vector<TrackData>>> trackData;             // additional track info (chi2, nClusters, track parameters)
  std::unique_ptr<TTreeReaderValue<std::vector<uint32_t>>> orbits;                 // first orbit of each TF in the input data
  std::unique_ptr<TTreeReaderValue<o2::ctp::LumiInfo>> lumiTF;                     // lumi info

  // Previous I/O sample, for the instantaneous-rate log inside the TF loop.
  Long64_t perfLastBytes = 0;
  std::chrono::steady_clock::time_point perfLastSample;

  int trackCounter_local{0};

  // Track-loop parallelism, WITHIN this one file-thread only -- active when nFileThreads==1 (GRID/
  // alien:// mode, forced by getInputFileList() for TGrid safety) or when there's only one input file
  // (otherwise every other core would sit idle). Safe because the track/cluster loop body touches no
  // TGrid/CCDB/file I/O (SetEntry() already happened before this point), only in-memory TF data plus the
  // same per-voxel mutex (vox.mtx) multi-file-thread mode already relies on. Each worker gets its own
  // copy of every piece of mutated state (RNG, scratch coordinates, counters) -- sharing any of it
  // across workers would be a silent data race. Otherwise nTrackWorkers is forced to 1 (serial, via
  // worker index 0) to avoid oversubscribing on top of the file-threads.
  int nTrackWorkers = 1;
  if (nFileThreads == 1 || fileList.size() == 1) {
    if (maxTrackWorkers > 0) {
      // Explicit override (e.g. the number of cores actually allocated to a batch/GRID job).
      // hardware_concurrency() reports the machine's core count, not the job's allocation -- an 8-core
      // GRID job auto-detects 32 and oversubscribes 4x -- so when the caller knows, trust it instead.
      nTrackWorkers = maxTrackWorkers;
    } else {
      unsigned hc = std::thread::hardware_concurrency();
      nTrackWorkers = (hc == 0) ? 8 : static_cast<int>(hc);
      if (nTrackWorkers > 32) {
        nTrackWorkers = 32;
      }
    }
  }
  LOGP(info, "[Thread_{}] Using {} track-worker thread(s) for the track loop", iThread, nTrackWorkers);
  std::vector<Vec3d> clsPosWorker(nTrackWorkers);
  // Per-worker scratch: this track's position at the current row.
  std::vector<Vec3d> trackPosAtRow_worker(nTrackWorkers);
  std::vector<long int> nEdgeClustersSkipped_worker(nTrackWorkers, 0);
  std::vector<Long64_t> trackCounter_local_worker(nTrackWorkers, 0);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // ===| FILE LOOP |===================================================================================================================
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // This thread's share of the input files, as an explicit work list: this is what lets a file that
  // looks unhealthy be pushed to the BACK and retried later instead of being processed now or dropped --
  // see the health checks below for why deferring beats both. Appending while iterating by index is
  // safe: the bound is re-read every iteration and no iterators are held.
  std::vector<int> workList;
  for (int i = iThread; i < int(fileList.size()); i += nFileThreads) {
    workList.push_back(i);
  }
  // How many times each file has already been deferred, so a persistently sick one cannot cycle forever.
  std::vector<int> deferCount(fileList.size(), 0);
  int maxFileDeferrals = 1;
  if (const char* envMaxDeferrals = gSystem->Getenv("SCDCALIB_MAX_FILE_DEFERRALS")) {
    maxFileDeferrals = std::atoi(envMaxDeferrals);
  }
  // Health-probe timings (seconds) of the files accepted so far, used as this job's own baseline. An
  // absolute threshold cannot work here: measured healthy timings differ by ~6x between GRID regimes,
  // so what matters is how a file compares to the others in ITS OWN slot, not to a constant.
  std::vector<double> probeTimes;
  double probeSlowFactor = 5.0;
  if (const char* envSlowFactor = gSystem->Getenv("SCDCALIB_PROBE_SLOW_FACTOR")) {
    probeSlowFactor = std::atof(envSlowFactor);
  }
  // Fallback used until enough samples exist for a median to mean anything (and if the very first file
  // of a slot is the sick one, this is the only thing standing between us and the stall).
  double probeAbsMaxSec = 15.0;
  if (const char* envProbeAbsMax = gSystem->Getenv("SCDCALIB_PROBE_ABS_MAX_SEC")) {
    probeAbsMaxSec = std::atof(envProbeAbsMax);
  }
  // Floor under the median-relative threshold once >=3 samples exist. Without it, a fast-regime median
  // (tens of ms) makes ordinary network jitter look "5x slower than usual" and trip the probe -- and a
  // file unlucky enough to trip it twice is dropped for good (see deferFile below), losing real data to
  // noise. The probe's actual target is stalls an order of magnitude bigger (~20s against a ~1.5s
  // baseline, in the real case this was built around), so a small floor can't mask that while still
  // absorbing sub-second jitter in a fast regime.
  double probeMinThresholdSec = 2.0;
  if (const char* envProbeMinThreshold = gSystem->Getenv("SCDCALIB_PROBE_MIN_THRESHOLD_SEC")) {
    probeMinThresholdSec = std::atof(envProbeMinThreshold);
  }

  for (size_t iWork = 0; iWork < workList.size(); ++iWork) {
    const int iFile = workList[iWork];
    // Get filename from fileList
    auto fileName = fileList[iFile];

    // Check if enough tracks are processed
    if ((maxTracks > 0) && (nTracksProcessed.load(std::memory_order_relaxed) > maxTracks)) {
      LOGP(info, "[Thread_{}] Maximum number of requested tracks processed {} > {} ({}), will not process further files", iThread, nTracksProcessed.load(std::memory_order_relaxed), maxTracks, maxTracksPerSlice);
      break;
    }

    if (gSystem->Getenv("FORCESE") && !TString(fileName.data()).EndsWith(gSystem->Getenv("FORCESE"))) {
      fileName += "?se=";
      fileName += gSystem->Getenv("FORCESE");
    }

    // Open tree and set branches
    LOGP(info, "[Thread{}] Processing input file {}", iThread, fileName);
    unbinnedResiduals.reset(nullptr);
    trackRefs.reset(nullptr);
    lumiTF.reset(nullptr);
    trackData.reset(nullptr);
    orbits.reset(nullptr);
    // Must be destroyed here, before the readers/file below: TTreePerfStats registers itself as the
    // process-wide gPerfStats global and keeps raw TFile*/TTree* pointers to the file it watches.
    // Leaving it alive past this point means gPerfStats still points at this (about to be destroyed)
    // file while the NEXT file's setup does real reads -- TFile::ReadBuffer() unconditionally calls
    // gPerfStats->FileReadEvent(thatFile, ...), which compares thatFile against the dangling fFile
    // pointer; if the allocator hands the new TFile the same address the old one was just freed from
    // (a real, common allocator pattern for same-sized immediate reuse), the comparison spuriously
    // matches and it dereferences the also-dangling fTree -- a real, reproduced segfault on the 2nd+
    // file of a multi-file run, verified against the installed ROOT's TTreePerfStats.cxx/TFile.cxx
    // source. perfStats's own destructor is safe to call here (only clears gPerfStats if it's still the
    // registered one; never dereferences fTree/fFile), so this alone fixes it.
    perfStats.reset(nullptr);
    treeUnbinnedResiduals.reset(nullptr);
    treeTrackData.reset(nullptr);
    treeRecords.reset(nullptr);

    const auto openStart = std::chrono::steady_clock::now();
    inputFile.reset(TFile::Open(fileName.c_str()));
    if ((!inputFile || inputFile->IsZombie())) {
      // A forced-SE URL (FORCESE or the auto-picked/cached fastest SE, see getInputFileList) can fail if
      // this particular file isn't actually hosted there -- fall back to unforced alien:// resolution
      // rather than skipping the file outright, since a stale SE cache would otherwise silently drop data.
      auto sePos = fileName.find("?se=");
      if (sePos != std::string::npos) {
        std::string fallbackName = fileName.substr(0, sePos);
        LOGP(warning, "[Thread_{}] Forced-SE open failed for {}, retrying without SE override: {}", iThread, fileName, fallbackName);
        inputFile.reset(TFile::Open(fallbackName.c_str()));
      }
    }
    // Gates the "[prefetch diag]" per-TF/per-file I/O diagnostics further below: real signal for
    // diagnosing GRID stalls/CPU-idle-watchdog kills (the reason this machinery exists at all), but
    // pure noise on a local/Lustre run with many file-threads where none of that risk applies -- a
    // 36-file-thread local run was producing an unreadable flood of per-TF consumer lines otherwise.
    const bool isAlienFile = fileName.find("alien://") != std::string::npos;
    if (isAlienFile) {
      if (inputFile && !inputFile->IsZombie()) {
        inputFile->SetBufferSize(4000000);
        LOGP(info, "[Thread_{}] Set buffer size to {}", iThread, inputFile->GetBufferSize());
      } else {
        LOGP(info, "[Thread_{}] TFile {} is empty", iThread, fileName);
      }
    }

    if (!inputFile || inputFile->IsZombie()) {
      LOGP(warning, "[Thread{}] Skipping file {}", iThread, fileName);
      continue;
    }

    // Deferring an unhealthy-looking file, rather than skipping it. Observed for real: a file that was
    // reproducibly slow in one job read at full speed a short time later -- the slowness is transient
    // storage-server state, not a property of the file. So dropping it outright throws away data that
    // would very likely have been fine. Pushing it to the back of this thread's work list costs exactly
    // what skipping costs right now, but gives the server time to recover, and if maxTracks stops the
    // job first the file is never touched again at all. The caller always moves on to the next file.
    auto deferFile = [&](const char* reason, double measured, double threshold) {
      if (deferCount[iFile] < maxFileDeferrals) {
        ++deferCount[iFile];
        workList.push_back(iFile);
        LOGP(warning, "[Thread{}] {} for {} ({:.1f} s vs {:.1f} s threshold) -- deferring it to the end of this thread's file list (attempt {} of {}); storage slowness has been seen to be transient, so it may read fine later, and maxTracks may stop the job before we return to it",
             iThread, reason, fileName, measured, threshold, deferCount[iFile], maxFileDeferrals);
        return;
      }
      LOGP(warning, "[Thread{}] {} for {} ({:.1f} s vs {:.1f} s threshold) and it has already been deferred {} time(s) -- skipping this file for good",
           iThread, reason, fileName, measured, threshold, deferCount[iFile]);
    };

    // --- Unhealthy-replica gate -------------------------------------------------------------------
    // An abnormally slow TFile::Open is the earliest sign a replica's storage server is struggling,
    // and the last point we can walk away cheaply: the warm-up read right after this pulls a whole
    // TTreeCache-sized chunk in one uninterruptible call, so if the server stalls there the job hangs
    // until AliEn's ~15-minute idle-CPU watchdog kills it, discarding all work already done. Skipping
    // one file here costs a fraction of a slot's statistics, so the trade is heavily one-sided. 15 s
    // threshold, calibrated against real GRID opens (absolute, not relative to this job's own timings
    // -- may need revisiting on very different links). Tune via SCDCALIB_MAX_FILE_OPEN_SEC; <= 0
    // disables it.
    double maxFileOpenSec = 15.0;
    if (const char* envMaxOpenSec = gSystem->Getenv("SCDCALIB_MAX_FILE_OPEN_SEC")) {
      maxFileOpenSec = std::atof(envMaxOpenSec);
    }
    const double openSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - openStart).count();
    if (maxFileOpenSec > 0 && openSec > maxFileOpenSec) {
      deferFile("Slow file open (SCDCALIB_MAX_FILE_OPEN_SEC)", openSec, maxFileOpenSec);
      continue;
    }

    treeUnbinnedResiduals = std::make_unique<TTreeReader>("unbinnedResid", inputFile.get());
    if (!treeUnbinnedResiduals->GetTree()) {
      LOGP(warning, "[Thread{}] Could not get tree 'unbinnedResid' from file {}. Skipping file!", iThread, fileName);
      continue;
    }

    // GetEntries() only reads TTree header metadata, not branch data -- cheap even on alien://, unlike
    // the TTreeCache/branch setup further below. Fetched here (still before that setup) so the fail-fast
    // time-window check below can run before anything expensive touches the network.
    const auto nTFEntries = treeUnbinnedResiduals->GetEntries();
    if (nTFEntries <= 0) {
      LOGP(warning, "[Thread{}] Tree 'unbinnedResid' in file {} has no entries. Skipping file!", iThread, fileName);
      continue;
    }

    treeRecords = std::make_unique<TTreeReader>("records", inputFile.get());
    if (!treeRecords->GetTree()) {
      LOGP(warning, "[Thread{}] Could not get tree 'records' from file {}. Skipping file!", iThread, fileName);
      continue;
    }
    orbits = std::make_unique<TTreeReaderValue<std::vector<uint32_t>>>(*treeRecords, "firstTForbit");
    // Real data has exactly one entry in 'records', but MC input can have several (e.g. one per
    // simulation chunk merged into this file) -- each entry's own 'firstTForbit' only covers that
    // chunk's TFs, so reading just entry 0 silently truncated the orbit list on MC, tripping the
    // length check below and skipping the whole file. Concatenate every entry's vector instead, in
    // entry order, to rebuild the same flat, TF-index-ordered list this file's real-data path already
    // produced from its single entry (verified for real: MC input observed with 5 'records' entries).
    std::vector<uint32_t> combinedOrbits;
    {
      const Long64_t nRecordsEntries = treeRecords->GetEntries();
      bool recordsOk = true;
      for (Long64_t ie = 0; ie < nRecordsEntries; ++ie) {
        if (treeRecords->SetEntry(ie) != TTreeReader::kEntryValid ||
            !checkReaderValue(*orbits, iThread, fileName)) {
          recordsOk = false;
          break;
        }
        const auto& thisEntryOrbits = **orbits;
        combinedOrbits.insert(combinedOrbits.end(), thisEntryOrbits.begin(), thisEntryOrbits.end());
      }
      if (!recordsOk) {
        LOGP(warning, "[Thread{}] Could not load the orbits from tree 'records' in file {}. Skipping file!", iThread, fileName);
        continue;
      }
    }
    // the orbit list is indexed with the TF index of the unbinnedResid tree below, so it has to be at least as long
    if (static_cast<Long64_t>(combinedOrbits.size()) < nTFEntries) {
      LOGP(error, "[Thread{}] 'firstTForbit' has fewer entries than the residual tree has TFs ({} vs {}) in file {}. Skipping file!", iThread,
           combinedOrbits.size(), nTFEntries, fileName);
      continue;
    }

    // Set timeStamp for processing, get this file's [min,max] orbit-derived time range, and find the
    // first TF entry actually inside the requested window -- all in one pass over the (already
    // downloaded) 'firstTForbit' array. Bounded to the first nTFEntries entries: orbits can have extra
    // trailing entries beyond what the residual tree actually has (see the size check above) that don't
    // correspond to any real TF and must not feed either the fail-fast check below or the warm-up entry.
    uint32_t minFirstOrbit = -1;
    uint32_t maxFirstOrbit = 0;
    Long64_t warmupEntry = 0;
    bool foundWarmupEntry = (firstTFTime <= 0); // no time filter set: entry 0 is always fine to warm up on
    {
      const auto& orbitsVec = combinedOrbits;
      for (Long64_t i = 0; i < nTFEntries; ++i) {
        const uint32_t orbit = orbitsVec[i];
        if (orbit < minFirstOrbit) {
          minFirstOrbit = orbit;
        }
        if (orbit > maxFirstOrbit) {
          maxFirstOrbit = orbit;
        }
        if (!foundWarmupEntry) {
          const int64_t t = orbitResetTimeMS + orbit * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
          if (t >= firstTFTime && t <= lastTFTime) {
            warmupEntry = i;
            foundWarmupEntry = true;
          }
        }
      }
    }
    // ---| Fail fast on files with zero overlap with the requested time window |---
    if (firstTFTime > 0) {
      const int64_t fileMinTimeMS = orbitResetTimeMS + minFirstOrbit * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
      const int64_t fileMaxTimeMS = orbitResetTimeMS + maxFirstOrbit * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
      if (fileMaxTimeMS < firstTFTime || fileMinTimeMS > lastTFTime) {
        LOGP(warning, "[Thread{}] File {} has no TF inside the requested time window [{}, {}] ms (file spans [{}, {}] ms) -- skipping the whole file without downloading its residual tree",
             iThread, fileName, firstTFTime, lastTFTime, fileMinTimeMS, fileMaxTimeMS);
        // Counted as if the per-TF loop below had visited and skipped each entry, so nTFs stays a
        // consistent denominator for the skip-fraction summary at the end.
        nTFs_thread[iThread] += nTFEntries;
        nTFsSkippedByTimeWindow_thread[iThread] += nTFEntries;
        continue;
      }
    }

    // --- Read-health probe, BEFORE the TTreeCache is enabled --------------------------------------
    // A slow open alone can miss a replica that opens fine but stalls on the first real read, so this
    // probes a small read directly, before SetCacheSize/AddBranchToCache below turn the first read into
    // a whole cache-sized fetch that can hang with nothing able to interrupt it. Probes 'trackData' (a
    // small member-split branch) rather than the already-read 'records' tree, which sits next to the
    // file header and reads fast regardless of whether the replica then stalls on real data. Uses
    // TBranch::GetEntry directly rather than a TTreeReader, to avoid creating a second reader on a tree
    // that gets its real one further below. A missing branch skips the probe rather than failing it --
    // this is a health check, not a validity check.
    {
      TTree* probeTree = dynamic_cast<TTree*>(inputFile->Get("trackData"));
      TBranch* probeBranch = probeTree ? probeTree->GetBranch("trk.nClsTPC") : nullptr;
      if (probeBranch && probeTree->GetEntries() > 0) {
        const Long64_t probeEntry = std::min<Long64_t>(warmupEntry, probeTree->GetEntries() - 1);
        const auto probeStart = std::chrono::steady_clock::now();
        const Int_t probeBytes = probeBranch->GetEntry(probeEntry);
        const double probeSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - probeStart).count();

        // If the read returned nothing, the measurement says nothing about the replica's health -- fall
        // through to the normal path rather than judging the file on it. Deliberately fail-open: a probe
        // that cannot measure must not be able to reject files, or an unexpected branch layout would
        // quietly defer every file in the slot.
        if (probeBytes <= 0) {
          LOGP(info, "[Thread{}] Read-health probe returned no data for {} (entry {}) -- skipping the health check for this file", iThread, fileName, probeEntry);
        } else {
          // Baseline: the median probe time of files already accepted in this slot. Below 3 samples a
          // median is meaningless, so fall back to the absolute guard.
          double probeThreshold = probeAbsMaxSec;
          if (probeTimes.size() >= 3) {
            std::vector<double> sorted(probeTimes);
            std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
            const double median = sorted[sorted.size() / 2];
            probeThreshold = std::max(probeSlowFactor * median, probeMinThresholdSec);
          }
          if (probeThreshold > 0 && probeSec > probeThreshold) {
            deferFile("Slow read probe (SCDCALIB_PROBE_SLOW_FACTOR/SCDCALIB_PROBE_ABS_MAX_SEC)", probeSec, probeThreshold);
            continue;
          }
          // Only healthy files feed the baseline, so one sick file cannot raise the bar for the next.
          probeTimes.push_back(probeSec);
        }
      }
    }

    // I/O throughput monitor -- large cache so async prefetch has room to read many baskets ahead; only
    // branches actually accessed get cached/prefetched. TTreePerfStats records raw bytes read from the
    // remote file, so its rate is the actual download speed.
    //
    // 256 MB, not 512 MB: a real GRID job hit an 11+ minute stall refilling a single 512MB chunk (~0.7
    // MB/s vs. 45-100 MB/s for every other chunk on the same SE), then a second stall that never
    // recovered, losing the whole job to AliEn's idle-CPU watchdog. A smaller chunk halves the
    // worst-case single-chunk wait and lets a persistently slow file be abandoned sooner -- a
    // reliability trade-off against 512MB's better throughput (~17% vs ~13% wall-clock reduction) when
    // nothing is stalling. Overridable via SCDCALIB_CACHE_SIZE_MB.
    {
      TTree* residTree = treeUnbinnedResiduals->GetTree();
      int dbgCacheSizeMB = 256;
      if (const char* envCacheSizeMB = gSystem->Getenv("SCDCALIB_CACHE_SIZE_MB")) {
        dbgCacheSizeMB = std::atoi(envCacheSizeMB);
      }
      if (isAlienFile) {
        LOGP(info, "[Thread{}] [prefetch diag] TTreeCache size = {} MB (SCDCALIB_CACHE_SIZE_MB, default 256)", iThread, dbgCacheSizeMB);
      }
      residTree->SetCacheSize(static_cast<Long64_t>(dbgCacheSizeMB) * 1024 * 1024);
      residTree->AddBranchToCache("*", true);
      perfStats = std::make_unique<TTreePerfStats>(fmt::format("ioperf_{}_{}", iThread, iFile).data(), residTree);
    }
    perfLastBytes = 0;
    perfLastSample = std::chrono::steady_clock::now();

    unbinnedResiduals = std::make_unique<TTreeReaderValue<std::vector<UnbinnedResid>>>(*treeUnbinnedResiduals, "res");
    trackRefs = std::make_unique<TTreeReaderValue<std::vector<TrackDataCompact>>>(*treeUnbinnedResiduals, "trackInfo");
    lumiTF = std::make_unique<TTreeReaderValue<o2::ctp::LumiInfo>>(*treeUnbinnedResiduals, "CTPLumi");

    // Skip reading UnbinnedResid/TrackDataCompact members that are never used below. 'res' and
    // 'trackInfo' are member-split branches, one sub-branch per struct field, and the unused ones
    // dominate the file (res.tgSlp alone can be >1 GB in a single input file), so disabling them means
    // they are never transferred at all -- measured ~17% fewer bytes read. Must come AFTER the
    // TTreeReaderValues above so that it is the final word on these sub-branches' status.
    // Do not disable res.dy/dz/y/z/row/sec/rejected or trackInfo.idxFirstResidual/nResiduals/sourceId --
    // all of those are read below. Note trackInfo.filterFlag (on TrackDataCompact) is unused, while the
    // separate trk.filterFlag (on TrackData, read further down) is used; they are different fields.
    {
      TTree* residTree = treeUnbinnedResiduals->GetTree();
      for (const char* br : {"res.tgSlp", "res.channel", "trackInfo.multStack*",
                             "trackInfo.nExtDetResid", "trackInfo.filterFlag"}) {
        residTree->SetBranchStatus(br, 0);
      }
    }

    // Load one entry once so the reader values get bound, then verify all branches are there before
    // anything below dereferences them. Warms up on warmupEntry (computed above, the first entry
    // actually inside the requested time window) rather than always entry 0 -- for a file only partially
    // overlapping the window, entry 0 is often outside it, and fetching its residual data would be
    // exactly the kind of wasted network read the whole-file fail-fast check above targets, just at the
    // scale of one TF.
    if (treeUnbinnedResiduals->SetEntry(warmupEntry) != TTreeReader::kEntryValid) {
      LOGP(warning, "[Thread{}] Could not load entry {} of 'unbinnedResid' from file {}. Skipping file!", iThread, warmupEntry, fileName);
      continue;
    }
    if (!checkReaderValue(*unbinnedResiduals, iThread, fileName) ||
        !checkReaderValue(*trackRefs, iThread, fileName) ||
        !checkReaderValue(*lumiTF, iThread, fileName)) {
      LOGP(warning, "[Thread{}] Skipping file {}", iThread, fileName);
      continue;
    }

    // Re-prime the reader values after the pre-scan (which leaves the reader past the last entry),
    // so a stale-read (e.g. the bad-range-skip stats below) at warmupEntry dereferences valid data.
    if (treeUnbinnedResiduals->SetEntry(warmupEntry) != TTreeReader::kEntryValid) {
      LOGP(warning, "[Thread{}] Could not re-load entry {} of 'unbinnedResid' from file {}. Skipping file!", iThread, warmupEntry, fileName);
      continue;
    }

    {
      treeTrackData = std::make_unique<TTreeReader>("trackData", inputFile.get());
      if (!treeTrackData->GetTree()) {
        LOGP(warning, "[Thread{}] Could not get tree 'trackData' from file {}. Skipping file!", iThread, fileName);
        continue;
      }
      {
        TTree* trackDataTree = treeTrackData->GetTree();
        for (const char* br : {"trk.gid*", "trk.chi2TRD", "trk.deltaTOF", "trk.nTrkltsTRD",
                               "trk.clAvailTOF", "trk.TRDTrkltSlope*", "trk.nExtDetResid",
                               "trk.clIdx.*", "trk.multStack*"}) {
          trackDataTree->SetBranchStatus(br, 0);
        }
      }
      trackData = std::make_unique<TTreeReaderValue<std::vector<TrackData>>>(*treeTrackData, "trk");
      if (treeTrackData->GetEntries() != nTFEntries) {
        LOGP(error, "[Thread{}] The input trees with unbinned residuals and track information have a different number of entries ({} vs {}). Skipping file!", iThread,
             nTFEntries, treeTrackData->GetEntries());
        continue;
      }
      // Same TF indexing as 'unbinnedResid' -- warm up on the same entry for the same reason (see above).
      if (treeTrackData->SetEntry(warmupEntry) != TTreeReader::kEntryValid ||
          !checkReaderValue(*trackData, iThread, fileName)) {
        LOGP(warning, "[Thread{}] Skipping file {}", iThread, fileName);
        continue;
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ===| TIME FRAME LOOP |=============================================================================================================
    // Split into a background PRODUCER thread (skip-checks, SetEntry/warm-up, sync check, lumi/orbit
    // bookkeeping, and reading each TF's data) pushing TFPackages into a bounded queue, and this thread
    // as CONSUMER, popping a package and dispatching the track workers on it -- see BoundedTFQueue/
    // TFPackage above for why. The producer is the only thread that ever touches
    // treeUnbinnedResiduals/treeTrackData/the TTreeReaderValues (exactly one thread doing TGrid/file I/O
    // at a time); the consumer never touches them at all, only the packages it pops.
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    size_t tfQueueDepth = TFQueueDepthDefault;
    if (const char* envQueueDepth = gSystem->Getenv("SCDCALIB_TF_QUEUE_DEPTH")) {
      tfQueueDepth = static_cast<size_t>(std::atoi(envQueueDepth));
    }
    if (isAlienFile) {
      LOGP(info, "[Thread{}] [prefetch diag] TFQueueDepth = {} (SCDCALIB_TF_QUEUE_DEPTH, default {})", iThread, tfQueueDepth, TFQueueDepthDefault);
    }
    BoundedTFQueue tfQueue(tfQueueDepth);

    // Set by the consumer below when a single tfQueue.pop() waited unreasonably long, checked by the
    // producer between TF reads (never inside one -- see the long comment at the consumer's check site
    // for why). Assumes the fetch has become persistently slow rather than truly dead: a producer stuck
    // forever never reaches this check at all and would need a separate, not-yet-built process-level
    // watchdog -- this only shortens the "slow but eventually returns" case, not a genuine hang.
    std::atomic<bool> abandonFile{false};
    double popWaitAbandonMs = 60000.0; // 60s -- matches the scale of the real stalls that motivated this
    if (const char* envAbandonMs = gSystem->Getenv("SCDCALIB_POP_WAIT_ABANDON_MS")) {
      popWaitAbandonMs = std::atof(envAbandonMs);
    }

    std::thread producerThread([&]() {
      // A failed SetEntry() (checked below) only means the tree could not be repositioned -- it says
      // nothing about whether a given lazily-read branch's basket actually arrived. TTreeReaderValue
      // fetches each branch on its own first dereference for the current entry, and a mid-stream storage
      // hiccup there doesn't throw: it prints a ROOT error and leaves the underlying object in an
      // unspecified state, which then segfaults wherever it's first used with no indication of why
      // (observed for real: an xrootd "Operation expired" mid basket-read, immediately followed by a
      // SIGSEGV with only the ROOT error line as a clue). Must be called right after the value's first
      // dereference for this entry -- GetReadStatus() reports the status of that specific read.
      auto checkReadOk = [&](ROOT::Internal::TTreeReaderValueBase& val, const char* branchName, int iEntry) {
        if (val.GetReadStatus() != ROOT::Internal::TTreeReaderValueBase::kReadSuccess) {
          LOGP(warning, "[Thread{}] Storage read error on branch '{}' at entry {} of file {} (GetReadStatus={}) -- skipping TF!",
               iThread, branchName, iEntry, fileName, static_cast<int>(val.GetReadStatus()));
          return false;
        }
        return true;
      };
      for (int iEntry = 0; iEntry < nTFEntries; ++iEntry) {
        // Checked here, between TF reads, never inside one -- this point is only ever reached right
        // after the previous push() succeeded, so the producer is never mid-call when this fires. See
        // the consumer's check site for the full reasoning.
        if (abandonFile.load(std::memory_order_relaxed)) {
          LOGP(warning, "[Thread{}] Abandoning the rest of file {} ({}/{} TFs read) after a persistently slow fetch", iThread, fileName, iEntry, nTFEntries);
          break;
        }
        ++nTFs_thread[iThread];

        // Periodic I/O throughput progress log. Now reports the producer's own progress through the
        // file, which can run ahead of what the consumer has actually finished processing.
        if (iEntry % 50 == 0) {
          double instMBps = 0.0, totMB = 0.0;
          if (perfStats) {
            const auto nowSample = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(nowSample - perfLastSample).count();
            const Long64_t br = perfStats->GetBytesRead();
            instMBps = (dt > 0) ? (br - perfLastBytes) / (1024.0 * 1024.0) / dt : 0.0;
            totMB = br / (1024.0 * 1024.0);
            perfLastBytes = br;
            perfLastSample = nowSample;
          }
          const Long64_t nProcessedNow = nTracksProcessed.load(std::memory_order_relaxed);
          if (maxTracks > 0) {
            LOGP(info, "[Thread{}] TF entry {}/{} | read {:.1f} MB, inst. {:.1f} MB/s | tracks processed {}/{} ({:.1f}%)",
                 iThread, iEntry, nTFEntries, totMB, instMBps, nProcessedNow, maxTracks, 100.0 * nProcessedNow / maxTracks);
          } else {
            LOGP(info, "[Thread{}] TF entry {}/{} | read {:.1f} MB, inst. {:.1f} MB/s | tracks processed {} (no limit set)",
                 iThread, iEntry, nTFEntries, totMB, instMBps, nProcessedNow);
          }
        }

        // Check if enough tracks are processed
        if ((maxTracks > 0) && (nTracksProcessed.load(std::memory_order_relaxed) > maxTracks)) {
          LOGP(info, "[Thread{}] Maximum number of requested tracks processed {} > {} ({}), will not process further TFs", iThread, nTracksProcessed.load(std::memory_order_relaxed), maxTracks, maxTracksPerSlice);
          break;
        }

        // ---| check for TF time acceptance |---
        const int64_t tfTimeInMS = orbitResetTimeMS + combinedOrbits[iEntry] * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
        if ((firstTFTime > 0) && (tfTimeInMS < firstTFTime || tfTimeInMS > lastTFTime)) {
          if (nTFsSkippedByTimeWindow_thread[iThread] == 0) {
            // Log once per thread rather than per TF: this can legitimately fire for every TF of every
            // file, e.g. when the requested [firstTFTime,lastTFTime] window contains no data at all
            // because a time-slice boundary landed past the end of the run. A per-TF log would then
            // produce one line per TF for the entire job.
            LOGP(warning, "[Thread{}] TF at index {} (time {} ms, orbit {}) outside requested window [{}, {}] ms -- skipping (will keep happening silently for further TFs outside the window, see final summary for the total count)",
                 iThread, iEntry, tfTimeInMS, combinedOrbits[iEntry], firstTFTime, lastTFTime);
          }
          ++nTFsSkippedByTimeWindow_thread[iThread];
          continue;
        }
        // ---| check for time exclusion list |---
        if (badRanges_thread[iThread].size() > 0) {
          bool skip = false;
          for (const auto& range : badRanges_thread[iThread]) {
            if ((combinedOrbits[iEntry] >= range.from) && (combinedOrbits[iEntry] <= range.to)) {
              skip = true;
              break;
            }
          }
          if (invertBadRange) {
            skip = !skip;
          }
          if (skip) {
            nTracksSkippedByBadRangeList_thread[iThread] += (*trackRefs)->size();
            ++nTFsSkippedByBadRangeList_thread[iThread];
            continue;
          }
        }
        if (params_thread.timeFilter) {
          if (tfTimeInMS < params_thread.startTimeMS || tfTimeInMS > params_thread.endTimeMS) {
            continue;
          }
        }

        // --- [prefetch diag]: producer-side read/deserialize timing, kept to confirm the periodic-spike
        // I/O pattern still looks the same underneath the producer/consumer split -- expected to be
        // mostly hidden from the consumer by TFQueueDepth, not eliminated. ---
        const auto dbgTIoStart = std::chrono::steady_clock::now();

        // ---| Read entries |---
        if (treeUnbinnedResiduals->SetEntry(iEntry) != TTreeReader::kEntryValid) {
          LOGP(warning, "[Thread{}] Could not load entry {} of 'unbinnedResid' from file {}. Skipping TF!", iThread, iEntry, fileName);
          continue;
        }
        if (treeTrackData->SetEntry(iEntry) != TTreeReader::kEntryValid) {
          LOGP(warning, "[Thread{}] Could not load entry {} of 'trackData' from file {}. Skipping TF!", iThread, iEntry, fileName);
          continue;
        }

        // First dereference of 'trackInfo' for this entry -- triggers the actual branch read; must be
        // checked before .size() (or anything else) trusts the result, see checkReadOk above.
        (void)(**trackRefs);
        if (!checkReadOk(*trackRefs, "trackInfo", iEntry)) {
          continue;
        }
        const auto nTracks = (*trackRefs)->size();

        // Materialize this TF's 'res' branch here, on the producer thread, before it's copied out below.
        // TTreeReaderValue::operator* deserializes lazily on the first dereference per entry.
        (void)(**unbinnedResiduals).size();
        if (!checkReadOk(*unbinnedResiduals, "res", iEntry)) {
          continue;
        }

        const double dbgIoMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dbgTIoStart).count();
        {
          thread_local double dbgSumIoMs = 0.0;
          thread_local uint64_t dbgNProduced = 0;
          dbgSumIoMs += dbgIoMs;
          ++dbgNProduced;
          if (isAlienFile && dbgNProduced % 10 == 0) {
            LOGP(info, "[Thread{}] [prefetch diag][producer] TF {} nTracks={} ioMs={:.1f} | running: {} TF(s) read, sumIoMs={:.0f}",
                 iThread, iEntry, nTracks, dbgIoMs, dbgNProduced, dbgSumIoMs);
          }
        }

        // First dereference of 'trackData' for this entry -- same lazy-read/checkReadOk requirement as
        // 'trackInfo'/'res' above.
        (void)(**trackData);
        if (!checkReadOk(*trackData, "trackData", iEntry)) {
          continue;
        }

        // the track loop below indexes trackData with the trackRefs index, so both have to be in sync
        if ((**trackData).size() < nTracks) {
          LOGP(warning, "[Thread{}] TF {} of file {} has fewer track data entries than track references ({} vs {}). Skipping TF!", iThread, iEntry, fileName,
               (**trackData).size(), nTracks);
          continue;
        }

        lumiSumCTP_thread[iThread] += (*lumiTF)->getLumi();
        ++lumiEntriesCTP_thread[iThread];

        timeMSsel_thread[iThread].emplace_back(tfTimeInMS);
        orbitsSel_thread[iThread].emplace_back((*lumiTF)->orbit);
        ctpLumiSel_thread[iThread].emplace_back((*lumiTF)->getLumi());

        // Copy the three vectors out (see TFPackage) and hand the package to the consumer. push() blocks
        // here if the queue is already at TFQueueDepth, which is exactly the back-pressure that keeps the
        // producer from running arbitrarily far ahead.
        auto pkg = std::make_unique<TFPackage>();
        pkg->iEntry = iEntry;
        pkg->trackRefsVec = **trackRefs;
        pkg->trackDataVec = **trackData;
        pkg->unbinnedResidualsVec = **unbinnedResiduals;
        tfQueue.push(std::move(pkg));
      }
      tfQueue.setDone();
    });

    while (true) {
      const auto dbgTPopStart = std::chrono::steady_clock::now();
      std::unique_ptr<TFPackage> pkg = tfQueue.pop();
      const double dbgPopWaitMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dbgTPopStart).count();
      const size_t dbgQueueSizeAfterPop = tfQueue.size(); // how many the producer still has ready right now

      // A pop() this slow only ever returns *after* the producer's own push() for this package already
      // succeeded -- i.e. the producer is guaranteed to be between TF reads right now, not blocked
      // inside one, so signalling it here can never race a live TFile/TGrid call. Real motivation: a
      // real GRID job had one file's chunk refill alone take 11+ minutes (~0.7 MB/s vs. 45-100 MB/s for
      // every other chunk on the same SE), then a second chunk on the same file never returned and the
      // whole job was killed by AliEn's idle-CPU watchdog. This won't catch that second, truly-dead case
      // (the producer never reaches this check then -- needs a separate process-level watchdog, not yet
      // built), but it does mean a merely very slow file gets abandoned after one bad chunk instead of
      // risking a permanent stall.
      if (pkg && dbgPopWaitMs > popWaitAbandonMs && !abandonFile.load(std::memory_order_relaxed)) {
        LOGP(warning, "[Thread{}] popWait {:.0f} ms exceeds abandon threshold {:.0f} ms (SCDCALIB_POP_WAIT_ABANDON_MS) -- signalling the producer to give up on the rest of this file, assuming a persistently slow fetch rather than a dead one",
             iThread, dbgPopWaitMs, popWaitAbandonMs);
        abandonFile.store(true, std::memory_order_relaxed);
      }
      if (!pkg) {
        break; // producer is done and the queue is drained
      }
      const int iEntry = pkg->iEntry;
      const auto nTracks = pkg->trackRefsVec.size();

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // ===| TRACK LOOP |==================================================================================================================
      // Body extracted into a per-track lambda so it can be dispatched across nTrackWorkers compute
      // threads (see the setup + rationale above the FILE LOOP). A `return` inside this lambda skips to
      // the next track -- the lambda body IS one track's worth of work. The cluster loop's own
      // `continue`s still mean what they always do: that for-loop lives inside the lambda unchanged.
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      auto processTrack = [&](size_t iTrack, int iWorker) {
        const auto& trkInfo = pkg->trackRefsVec[iTrack];
        if (!GID::includesSource(trkInfo.sourceId, sources)) {
          return;
        }
        const auto& trk = pkg->trackDataVec[iTrack];
        if (!revalidateTrack(trk, params_thread)) {
          return;
        }

        auto propagator = o2::base::Propagator::Instance();
        o2::track::TrackPar trkPar = trk.par;
        int sign = trkPar.getSign();

        int charge = 0;
        if (sign > 0) {
          charge = 1;
        }

        // dE/dx cut
        if (maxdEdx > 0 && trk.dEdxTPC > maxdEdx) {
          return;
        }

        if (maxdEdxExp > 0 || maxDevdEdxOverExp > 0) {
          // propagate to the beginning of the inner containment vessel, to use the momentum for dE/dx expected
          if (!propagator->PropagateToXBxByBz(trkPar, 63.2, 0.99, 2., o2::base::Propagator::MatCorrType::USEMatCorrLUT)) { // USEMatCorrTGeo, USEMatCorrLUT, USEMatCorrNONE
            return;
          }

          const auto dEdxExp = o2::track::BetheBlochSolidOpt(trk.par.getP() / trk.par.getPID().getMass()) * 3e4;

          if (maxdEdxExp > 0 && dEdxExp > maxdEdxExp) {
            return;
          }

          if (maxDevdEdxOverExp > 0 && std::abs(trk.dEdxTPC / dEdxExp - 1) > maxDevdEdxOverExp) {
            return;
          }
        }
        if (!propagator->PropagateToXBxByBz(trkPar, 85., 0.99, 2., o2::base::Propagator::MatCorrType::USEMatCorrLUT)) { // USEMatCorrTGeo, USEMatCorrLUT, USEMatCorrNONE
          return;
        }

        // INCREASE LOCAL TRACK COUNTER
        ++trackCounter_local_worker[iWorker];

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // ===| CLUSTER & RESIDUAL LOOP |=====================================================================================================
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (unsigned int i = trkInfo.idxFirstResidual; i < trkInfo.idxFirstResidual + trkInfo.nResiduals; ++i) {
          const auto& residIn = pkg->unbinnedResidualsVec[i];
          int sec = residIn.sec;
          if (residIn.row >= NRows || sec >= NSectors || sec < 0) { // non TPC residuals have row>=160, though to see them one should loop until i<(trc.clIdx.getFirstEntry() + trc.clIdx.getEntries() + trc.nExtDetResid)
            continue;
          }

          if (isRejectedResidual(residIn)) {
            continue;
          }

          float angleSec = TMath::DegToRad() * (10.0 + 20.0 * sec);
          std::array<unsigned char, TrackResiduals::VoxDim> bvox;
          // cluster position
          float xPos = param::RowX[residIn.row];
          float yPos = residIn.y * param::MaxY / 0x7fff + residIn.dy * param::MaxResid / 0x7fff;
          float zPos = residIn.z * param::MaxZ / 0x7fff + residIn.dz * param::MaxResid / 0x7fff;
          // exclude the edge pads as they are biased!
          // get max y-position of edge pad: pad centre last pad - pad width/2
          if (skipEdgePads && std::abs(yPos) > yMaxCentrePadByRow[residIn.row]) {
            ++nEdgeClustersSkipped_worker[iWorker];
            continue;
          }

          clsPosWorker[iWorker].SetXYZ(xPos, yPos, zPos);
          if (!trackResiduals.findVoxelBin(sec, xPos, yPos, zPos, bvox)) {
            // we are not inside any voxel
            continue;
          }

          // circle pool for this voxel is shared across threads -> lock for the rest of this iteration
          auto& vox = voxels[((sec * NRows + bvox[2]) * nY2XBins + bvox[1]) * nZ2XBins + bvox[0]];
          std::unique_lock<std::mutex> voxLock(vox.mtx);

          // XALEX
          if (vox.poolCounter[charge] == NPool) {
            continue;
          }

          //---------------------------------------------------------
          // Main part of the new code
          float xposvox, yoverxpos, zoverxpos;
          trackResiduals.getVoxelCoordinates(sec, bvox[2], bvox[1], bvox[0], xposvox, yoverxpos, zoverxpos);
          if (fabs(xposvox) < 5.0) {
            continue;
          }
          float yposvox = yoverxpos * xposvox;
          float zposvox = zoverxpos * xposvox;

          float dxclsvoxel = clsPosWorker[iWorker].X() - xposvox;
          float dyclsvoxel = clsPosWorker[iWorker].Y() - yposvox;
          float dzclsvoxel = clsPosWorker[iWorker].Z() - zposvox;

          Vec3d deltaClsVoxel;
          deltaClsVoxel.SetXYZ(dxclsvoxel, dyclsvoxel, dzclsvoxel);

          //-----------------------------------------
          trkPar.rotate(o2::math_utils::sector2Angle(sec));

          propagator->PropagateToXBxByBz(trkPar, xPos, 0.99, 2., o2::base::Propagator::MatCorrType::USEMatCorrLUT); // USEMatCorrTGeo, USEMatCorrLUT, USEMatCorrNONE
          trackPosAtRow_worker[iWorker].SetXYZ(trkPar.getX(), trkPar.getY(), trkPar.getZ());

          float sna, csa;
          o2::math_utils::CircleXY<o2::track::TrackParametrization<float>::value_t> xycircle;
          trkPar.getCircleParams(magfieldvalue, xycircle, sna, csa); // in global coordinates

          Vec3d circleCenterEstimate;
          circleCenterEstimate.SetXYZ(xycircle.xC, xycircle.yC, 0.0);
          circleCenterEstimate.RotateZ(-angleSec);
          float radius_estimate = xycircle.rC;

          if ((trackPosAtRow_worker[iWorker] - clsPosWorker[iWorker]).Perp() > maxDistIntCls) {
            continue;
          }

          //-----------------------------------------
          // DeltaZ corrections, two methods
          if (voxMapInput.size()) // with input map from first itteration, should be more precise than second method
          {
            // we already have a correction map available
            const auto& voxRes = voxelResults[sec][trackResiduals.getGlbVoxBin(bvox)]; // bvox: z,y,x
            float DX_input_map = voxRes.D[TrackResiduals::ResX];

            propagator->PropagateToXBxByBz(trkPar, xPos - DX_input_map, 0.99, 2., o2::base::Propagator::MatCorrType::USEMatCorrLUT); // USEMatCorrTGeo, USEMatCorrLUT, USEMatCorrNONE
            float DZdist = (zPos - trkPar.getZ());                                                                                   // distortion
            // accumulate Z sum (shared across threads for this voxel, guarded by vox.mtx) and increment counter
            vox.residualsAll[2] += DZdist;
            vox.counterZAll++;
          } else {
            // without input map, use rolling average -- gate on the cumulative NP/PP/NN counters (never
            // reset across flushes, unlike poolCounter) so this only fires once residualsAll[0] has
            // actually been computed from a decent number of samples, not merely whenever the in-flight
            // pool for this charge happens to be non-empty (poolCounter resets to 0 on every flush, so
            // checking it here both under- and over-fires relative to residualsAll[0]'s real validity).
            if (vox.counterNP > MinDxSamplesForZCorr || vox.counterPP > MinDxSamplesForZCorr || vox.counterNN > MinDxSamplesForZCorr) {
              float DXrollingaverage = vox.residualsAll[0];
              propagator->PropagateToXBxByBz(trkPar, xPos - DXrollingaverage, 0.99, 2., o2::base::Propagator::MatCorrType::USEMatCorrLUT); // USEMatCorrTGeo, USEMatCorrLUT, USEMatCorrNONE
              float DZdist = (zPos - trkPar.getZ());                                                                                       // distortion
              vox.residualsAll[2] += DZdist;
              vox.counterZAll++;
            }
          }
          //-----------------------------------------

          circleCenterEstimate -= deltaClsVoxel; // shift track to voxel center to avoid smearing within voxel

          int counter = vox.poolCounter[charge];
          vox.circleCenters[charge][counter] = Vec3f{static_cast<float>(circleCenterEstimate.X()), static_cast<float>(circleCenterEstimate.Y()), static_cast<float>(circleCenterEstimate.Z())};
          vox.circleRadii[charge][counter] = radius_estimate;
          vox.poolCounter[charge]++;

          //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
          // ===| PROCESSING VOXEL |============================================================================================================
          //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
          if ((vox.poolCounter[0] >= NPool || vox.poolCounter[1] >= NPool)) {
            ////////////////////////////////
            // Same polarity combinations //
            ////////////////////////////////
            float weightCurvatureInt = 0.0;
            Vec3d averageInt;
            averageInt.SetXYZ(0.0, 0.0, 0.0);

            for (int counterPos = 0; counterPos < (vox.poolCounter[1] - 1); counterPos++) {
              for (int counterPosB = (counterPos + 1); counterPosB < vox.poolCounter[1]; counterPosB++) {
                float radiusA = vox.circleRadii[1][counterPos];
                float radiusB = vox.circleRadii[1][counterPosB];

                if (fabs((radiusA - radiusB) / (radiusA + radiusB)) < 0.2) {
                  continue; // to similar bending radii
                }

                Vec3d intCircles = getIntCircles(radiusA, radiusB, vox.circleCenters[1][counterPos], vox.circleCenters[1][counterPosB], xposvox, yposvox);
                if (intCircles.Perp() < 280.0 && intCircles.Perp() > 60.0) {
                  float distIntCls = (intCircles - clsPosWorker[iWorker]).Perp(); // why?
                  if (distIntCls > maxDistIntCls) {
                    continue;
                  }

                  float weight = fabs((radiusA - radiusB) / (radiusA + radiusB));
                  float weight_curvature = weight;

                  weightCurvatureInt += weight_curvature;
                  averageInt += intCircles * weight_curvature;
                }
              }
            }

            if (weightCurvatureInt > 0) {
              averageInt *= 1.0 / weightCurvatureInt;

              // distortion
              // accumulate PP sums (x,y) and increment counter
              vox.residualsPP[0] += (xposvox - averageInt.X());
              vox.residualsPP[1] += (yposvox - averageInt.Y());
              vox.counterPP++;
            }

            weightCurvatureInt = 0.0;
            averageInt.SetXYZ(0.0, 0.0, 0.0);

            for (int counterNeg = 0; counterNeg < (vox.poolCounter[0] - 1); counterNeg++) {
              for (int counterNegB = (counterNeg + 1); counterNegB < vox.poolCounter[0]; counterNegB++) {
                float radiusA = vox.circleRadii[0][counterNeg];
                float radiusB = vox.circleRadii[0][counterNegB];

                if (fabs((radiusA - radiusB) / (radiusA + radiusB)) < 0.2) {
                  continue; // to similar bending radii
                }

                Vec3d intCircles = getIntCircles(radiusA, radiusB, vox.circleCenters[0][counterNeg], vox.circleCenters[0][counterNegB], xposvox, yposvox);
                if (intCircles.Perp() < 280.0 && intCircles.Perp() > 60.0) {
                  float distIntCls = (intCircles - clsPosWorker[iWorker]).Perp(); // why?
                  if (distIntCls > maxDistIntCls) {
                    continue;
                  }

                  float weight = fabs((radiusA - radiusB) / (radiusA + radiusB));
                  float weight_curvature = weight;

                  weightCurvatureInt += weight_curvature;
                  averageInt += intCircles * weight_curvature;
                }
              }
            }

            if (weightCurvatureInt > 0) {
              averageInt *= 1.0 / weightCurvatureInt;

              // distortion
              // accumulate NN sums (x,y) and increment counter
              vox.residualsNN[0] += (xposvox - averageInt.X());
              vox.residualsNN[1] += (yposvox - averageInt.Y());
              vox.counterNN++;
            }

            ////////////////////////////////////
            // Opposite polarity combinations //
            ////////////////////////////////////
            weightCurvatureInt = 0.0;
            averageInt.SetXYZ(0.0, 0.0, 0.0);

            for (int counterPos = 0; counterPos < vox.poolCounter[1]; counterPos++) {
              for (int counterNeg = 0; counterNeg < vox.poolCounter[0]; counterNeg++) {
                float radiusA = vox.circleRadii[1][counterPos];
                float radiusB = vox.circleRadii[0][counterNeg];
                Vec3d intCircles = getIntCircles(radiusA, radiusB, vox.circleCenters[1][counterPos], vox.circleCenters[0][counterNeg], xposvox, yposvox);
                if (intCircles.Perp() < 280.0 && intCircles.Perp() > 60.0) {
                  float distIntCls = (intCircles - clsPosWorker[iWorker]).Perp(); // why?
                  if (distIntCls > maxDistIntCls) {
                    continue;
                  }

                  // float weight_curvature = (1.0/radiusA)*(1.0/radiusB); // the larger the curvature the more precise the intersection can be calculated
                  // float weight_curvature = (radiusA)*(radiusB); // the larger the curvature the more precise the intersection can be calculated

                  float weight_curvature = 1.0; // (1.0/radiusA)*(1.0/radiusB);

                  weightCurvatureInt += weight_curvature;
                  averageInt += intCircles * weight_curvature;
                }
              }
            }

            if (weightCurvatureInt > 0) {
              averageInt *= 1.0 / weightCurvatureInt;

              // distortion
              // accumulate NP sums (x,y) and increment counter
              vox.residualsNP[0] += (xposvox - averageInt.X());
              vox.residualsNP[1] += (yposvox - averageInt.Y());
              vox.counterNP++;
            }

            float weightNP = vox.counterNP * 1.0;
            float weightPP = vox.counterPP * 0.01;
            float weightNN = vox.counterNN * 0.01;

            float sum_weight = weightNP + weightPP + weightNN;

            if (sum_weight > 0.0f) {
              // convert sums to means before weighting
              float meanNPx = (vox.counterNP > 0) ? (vox.residualsNP[0] / static_cast<float>(vox.counterNP)) : 0.0f;
              float meanNPy = (vox.counterNP > 0) ? (vox.residualsNP[1] / static_cast<float>(vox.counterNP)) : 0.0f;
              float meanPPx = (vox.counterPP > 0) ? (vox.residualsPP[0] / static_cast<float>(vox.counterPP)) : 0.0f;
              float meanPPy = (vox.counterPP > 0) ? (vox.residualsPP[1] / static_cast<float>(vox.counterPP)) : 0.0f;
              float meanNNx = (vox.counterNN > 0) ? (vox.residualsNN[0] / static_cast<float>(vox.counterNN)) : 0.0f;
              float meanNNy = (vox.counterNN > 0) ? (vox.residualsNN[1] / static_cast<float>(vox.counterNN)) : 0.0f;

              vox.residualsAll[0] = (weightNP * meanNPx + weightPP * meanPPx + weightNN * meanNNx) / sum_weight;
              vox.residualsAll[1] = (weightNP * meanNPy + weightPP * meanPPy + weightNN * meanNNy) / sum_weight;
            }

            // reset the pools: only the counter is reset. circleCenters/circleRadii are fixed-size
            // std::array<NPool> now (not std::vector) -- there's no clear()/resize() to even call;
            // stale entries beyond the new poolCounter are simply overwritten as the pool refills.
            for (int ic = 0; ic < 2; ic++) {
              vox.poolCounter[ic] = 0;
            }
          }
          //---------------------------------------------------------

          // // update COG for voxel bvox (update for X only needed in case binning is not per pad row)

        } // end of cluster loop
      }; // end of processTrack lambda

      const auto dbgTCpuStart = std::chrono::steady_clock::now();
      if (nTrackWorkers <= 1) {
        for (size_t iTrack = 0; iTrack < nTracks; ++iTrack) {
          processTrack(iTrack, 0);
        }
      } else {
        std::vector<std::thread> trackThreads;
        trackThreads.reserve(nTrackWorkers);
        for (int iWorker = 0; iWorker < nTrackWorkers; ++iWorker) {
          trackThreads.emplace_back([&, iWorker]() {
            for (size_t iTrack = iWorker; iTrack < nTracks; iTrack += nTrackWorkers) {
              processTrack(iTrack, iWorker);
            }
          });
        }
        for (auto& th : trackThreads) {
          th.join();
        }
      }
      const double dbgCpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dbgTCpuStart).count();
      {
        // [prefetch diag][consumer]: the number that actually matters. dbgPopWaitMs is how long this
        // thread blocked in tfQueue.pop() waiting for the producer -- with the pipeline working, this
        // should be near zero most of the time (the producer stays ahead), spiking only when it can't
        // keep up (e.g. a cache-refill spike bigger than TFQueueDepth's cushion).
        thread_local double dbgSumPopWaitMs = 0.0;
        thread_local double dbgSumCpuMs = 0.0;
        thread_local uint64_t dbgNTFs = 0;
        dbgSumPopWaitMs += dbgPopWaitMs;
        dbgSumCpuMs += dbgCpuMs;
        ++dbgNTFs;
        // Unlike the producer log above, this one had no sampling gate at all -- printed every single
        // TF, on every thread, which is the dominant source of "[prefetch diag]" log volume. Matched to
        // the producer's every-10th cadence (the cumulative sum* fields lose nothing from sampling) and
        // gated on isAlienFile for the same reason as the producer log.
        if (isAlienFile && dbgNTFs % 10 == 0) {
          LOGP(info, "[Thread{}] [prefetch diag][consumer] TF {} nTracks={} popWaitMs={:.1f} cpuMs={:.1f} queueSizeAfterPop={} | running totals over {} TF(s): sumPopWaitMs={:.0f} sumCpuMs={:.0f}",
               iThread, iEntry, nTracks, dbgPopWaitMs, dbgCpuMs, dbgQueueSizeAfterPop, dbgNTFs, dbgSumPopWaitMs, dbgSumCpuMs);
        }
      }
      // Merge this TF's per-worker counters into the file-thread-level counters the rest of
      // doFileProcessing (and the caller's merge loop, for nEdgeClustersSkipped_thread) expect.
      for (int iWorker = 0; iWorker < nTrackWorkers; ++iWorker) {
        trackCounter_local += trackCounter_local_worker[iWorker];
        trackCounter_local_worker[iWorker] = 0;
        nEdgeClustersSkipped_thread[iThread] += nEdgeClustersSkipped_worker[iWorker];
        nEdgeClustersSkipped_worker[iWorker] = 0;
      }
    } // end of TF loop (consumer)

    // producerThread only ever reaches here via tfQueue.setDone() (end of file or maxTracks reached),
    // which the consumer's pop() == nullptr check above already waited for -- this join is therefore
    // just reclaiming the thread, not a real wait. Must happen before perfStats->Finish()/Print() below,
    // since perfStats is only ever touched by the producer thread.
    producerThread.join();

    if (perfStats) {
      perfStats->Finish();
      if (isAlienFile) {
        perfStats->Print(); // full TTreePerfStats I/O report (incl. "Disk IO = ... MBytes/s") -- GRID-diagnostic only, see isAlienFile above
      }
      totalBytesReadPerf_thread[iThread] += perfStats->GetBytesRead();
    }

    // Occasionally flush local count to global
    if (trackCounter_local >= 1000) {
      nTracksProcessed.fetch_add(trackCounter_local, std::memory_order_relaxed);
      trackCounter_local = 0;
    }

  } // end of file loop

  if (trackCounter_local > 0) {
    nTracksProcessed.fetch_add(trackCounter_local, std::memory_order_relaxed);
  }
}

void staticMapCreatorCPM(std::string fileInput = "files.txt",
                         int runNumber = 527976,
                         std::string fileOutput = "voxRes.root",
                         std::string voxMapInput = "",
                         std::string trackSources = static_cast<std::string>(GID::ALL),
                         std::string z2xBinning = "",           // empty: default binning; single number: uniform binning; otherwise bin boundaries, e.g. "0.,0.02, 0.04, 0.06, 1"
                         std::string y2xBinning = "",           // empty: default binning; single number: uniform binning; otherwise bin boundaries, e.g. "-1,-0.998, -0.996, ... 0.996, 0.998, 1"
                         bool useSmoothed = true,               // use smoothed residuals as input
                         bool createSpline = true,              // create the splines
                         int maxTracksPerSlice = -1,            // limit the number of total tracks processed to maxTracksPerSlice * nBinsZ2X * nBinxY2X * 36
                         int minTracksPerSlice = -1,            // request a minimum number of tracks per slice. Otherwise the calibration is not created
                         std::string badRangeList = "",         // list of bad time ranges to be excluded in the calibration
                         long firstTFTime = -1,                 // First TF time to accept
                         long lastTFTime = -1,                  // Last TF time to accept
                         float maxdEdx = -1,                    // dE/dx cut above which tracks will be rejected
                         float maxdEdxExp = -1,                 // dE/dx expected cut above which tracks will be rejected
                         float maxDevdEdxOverExp = -1,          // maximum deviation of dE/dx / expected value, above the track is rejected
                         bool skipEdgePads = 1,                 // skip edge pads in the calibration, by default on
                         std::string badRangeSelection = "ALL", // use bad time ranges only for specific comment e.g. C1
                         float maxZ2XCut = 1.f,                 // overrides scdcalib.maxZ2X (the track tgl cut applied in revalidateTrack).
                                                                // Defaults to the SpacePointsCalibConfParam compiled-in value of 1.0
                         int maxTrackWorkers = -1,              // number of threads used for the track loop within one file. <=0 (default)
                                                                // auto-detects via hardware_concurrency(), which reports the machine's core
                                                                // count rather than the cores actually allocated to a batch job -- pass the
                                                                // real allocation explicitly when running on a batch system or the GRID
                         int nThreads = 8)                      // number of file-level threads, i.e. how many input files are processed
                                                                // concurrently. Forced to 1 for alien:// input regardless, since TGrid is
                                                                // not thread-safe (see below)
{
  LOGP(info, "TrackData::filterFlag {}, UnbinnedResid::rejected {}",
       HasFilterFlagMember<TrackData>::value ? "available" : "NOT available (old O2, cut skipped)",
       HasRejectedMember<UnbinnedResid>::value ? "available" : "NOT available (old O2, cut skipped)");

  // Enable multiple threads
  ROOT::EnableThreadSafety();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // ==== TIME ===============================================================================================================
  auto t_start = std::chrono::high_resolution_clock::now();
  // =========================================================================================================================
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  fair::Logger::SetVerbosity(fair::Verbosity::medium);
  fair::Logger::SetConsoleSeverity(fair::Severity::info);
  fair::Logger::SetFileSeverity(fair::Severity::info);

  const Mapper& mapper = Mapper::instance();

  // Obtain configuration
  const SpacePointsCalibConfParam& params = SpacePointsCalibConfParam::Instance();
  if (!std::filesystem::exists("scdconfig.ini")) {
    LOGP(warning, "Did not find configuration file. Using default parameters and storing them in scdconfig.ini");
    params.writeINI("scdconfig.ini", "scdcalib"); // to write default parameters to a file
  } else {
    params.updateFromFile("scdconfig.ini");
  }
  // Explicit override from the macro's own argument, applied AFTER any scdconfig.ini load so it wins
  // regardless of whether one happens to exist in CWD -- see maxZ2XCut's doc comment above. Using the
  // string-based setValue overload (implemented out-of-line in ConfigurableParam.cxx), not the
  // templated one -- that one needs boost::property_tree fully instantiated at the call site, which
  // isn't included here.
  o2::conf::ConfigurableParam::setValue("scdcalib.maxZ2X", std::to_string(maxZ2XCut));
  LOGP(info, "----- Dumping configuration values START -----");
  params.printKeyValues();
  LOGP(info, "----- Dumping configuration values END -----");

  GID::mask_t allowedSources = GID::getSourcesMask("ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF");
  GID::mask_t sources = allowedSources & GID::getSourcesMask(trackSources);

  // Get CCDB objects
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbmgr.setCaching(true);
  ccdbmgr.setFatalWhenNull(false);
  ccdbmgr.setURL("http://alice-ccdb.cern.ch");
  auto runDuration = ccdbmgr.getRunDuration(runNumber);
  auto tRun = runDuration.first + (runDuration.second - runDuration.first) / 2; // time stamp for the middle of the run duration
  ccdbmgr.setTimestamp(tRun);

  // CTP orbit reset time, does not change during the run
  const auto orbitResetTimeNS = ccdbmgr.get<std::vector<int64_t>>("CTP/Calib/OrbitReset");
  const int64_t orbitResetTimeMS = (*orbitResetTimeNS)[0] * 1e-3;
  LOGP(info, "Orbit reset time in MS is {}", orbitResetTimeMS);

  // Geometry, material budget and B-field
  auto geoAligned = ccdbmgr.get<TGeoManager>("GLO/Config/GeometryAligned");
  auto magField = ccdbmgr.get<o2::parameters::GRPMagField>("GLO/Config/GRPMagField");
  const o2::base::MatLayerCylSet* matLut = o2::base::MatLayerCylSet::rectifyPtrFromFile(ccdbmgr.get<o2::base::MatLayerCylSet>("GLO/Param/MatLUT"));
  o2::base::Propagator::initFieldFromGRP(magField);
  auto prop = o2::base::Propagator::Instance();
  prop->setMatLUT(matLut);
  float magfieldvalue = static_cast<float>(magField->getNominalL3Field());
  LOGP(info, "Nominal L3 field: {:.3f}", magfieldvalue);

  // GRP LHC and beam type
  auto grplhc = ccdbmgr.get<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF");
  const auto beamA = grplhc->getBeamZ(o2::constants::lhc::BeamA);
  const auto beamC = grplhc->getBeamZ(o2::constants::lhc::BeamC);
  const auto eCM = grplhc->getSqrtS();
  bool isPbPb = (beamA == 82 && beamC == 82);
  LOGP(info, "BeamA: {}, BeamC: {}, isPbPb: {}, Ecm: {}", beamA, beamC, isPbPb, eCM);

  // Input
  auto fileList = getInputFileList(fileInput);

  // Single-threaded when reading from AliEn: TGrid/xrootd access is not thread-safe. This overrides the
  // nThreads argument. Detected from the actual input file entries rather than from fileInput itself,
  // which for a GRID job is a local list of alien:// entries, so that local/Lustre input still gets the
  // requested parallelism.
  if (!fileList.empty() && fileList[0].rfind("alien://", 0) == 0) {
    LOGP(info, "AliEn input detected (alien:// prefix) -- TGrid access is not thread-safe, forcing nThreads=1");
    nThreads = 1;
  }
  int nFileThreads = nThreads;
  LOGP(info, "Using {} threads for processing", nFileThreads);

  // Set up binning
  const auto z2xBins = o2::RangeTokenizer::tokenize<float>(z2xBinning);
  const auto y2xBins = o2::RangeTokenizer::tokenize<float>(y2xBinning);

  TrackResiduals trackResiduals;

  trackResiduals.setZ2XBinning(z2xBins);
  trackResiduals.setY2XBinning(y2xBins);
  trackResiduals.init();

  const int nY2XBins = trackResiduals.getNY2XBins();
  const int nZ2XBins = trackResiduals.getNZ2XBins();
  LOGP(info, "nY2XBins: {}, nZ2XBins: {}", nY2XBins, nZ2XBins);

  const float maxDistIntCls = 25.0;

  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> vec_residualsAll; // sector, voxX, voxF, voxZ, xyz
  std::vector<std::vector<std::vector<std::vector<int>>>> vec_residuals_counterAll;        // sector, voxX, voxF, voxZ

  vec_residualsAll.resize(NSectors);
  vec_residuals_counterAll.resize(NSectors);
  for (int isec = 0; isec < NSectors; isec++) {
    vec_residualsAll[isec].resize(NRows);
    vec_residuals_counterAll[isec].resize(NRows);
    for (int ix = 0; ix < NRows; ix++) {
      vec_residualsAll[isec][ix].resize(nY2XBins);
      vec_residuals_counterAll[isec][ix].resize(nY2XBins);
      for (int iy = 0; iy < nY2XBins; iy++) {
        vec_residualsAll[isec][ix][iy].resize(nZ2XBins);
        vec_residuals_counterAll[isec][ix][iy].resize(nZ2XBins);
        for (int iz = 0; iz < nZ2XBins; iz++) {
          vec_residualsAll[isec][ix][iy][iz].resize(3);
          for (int ixyz = 0; ixyz < 3; ixyz++) {
            vec_residualsAll[isec][ix][iy][iz][ixyz] = 0.0;
          }
          vec_residuals_counterAll[isec][ix][iy][iz] = 0;
        }
      }
    }
  }

  const int nSlices = NSectors * trackResiduals.getNY2XBins() * trackResiduals.getNZ2XBins();
  const Long64_t maxTracks = maxTracksPerSlice * nSlices;
  const int minTracks = minTracksPerSlice * nSlices;
  std::atomic<Long64_t> nTracksProcessed{0};
  Long64_t nTracksProcessed_final{0};

  // Do we have a correction map available that we should apply to the clusters before the map extraction?
  std::array<std::vector<TrackResiduals::VoxRes>, NSectors> voxelResults{};
  if (voxMapInput.size()) {
    LOGP(info, "[InputCorrMap]: A correction map has been provided. Will apply the corrections to the cluster residuals");
    LOGP(info, "[InputCorrMap]: Resizing voxelResults to number of voxels");
    for (int iSec = 0; iSec < NSectors; ++iSec) {
      voxelResults[iSec].resize(trackResiduals.getNVoxelsPerSector());
    }
    TrackResiduals::VoxRes* voxResPtr = nullptr;
    std::unique_ptr<TFile> fIn = std::make_unique<TFile>(voxMapInput.c_str());
    if (!fIn->IsOpen() || fIn->IsZombie()) {
      LOGP(fatal, "[InputCorrMap]: Could not open input file {}", voxMapInput);
    }
    LOGP(info, "[InputCorrMap]: Getting TTree of voxels");
    std::unique_ptr<TTree> treeIn;
    treeIn.reset((TTree*)fIn->Get("voxResTree"));
    if (!treeIn) {
      LOGP(fatal, "[InputCorrMap]: Could not extract voxResTree from input file {}", voxMapInput);
    }
    treeIn->SetBranchAddress("voxRes", &voxResPtr);
    LOGP(info, "[InputCorrMap]: Getting voxel results and filling voxelResults");
    for (int iEntry = 0; iEntry < treeIn->GetEntries(); ++iEntry) {
      treeIn->GetEntry(iEntry);
      auto& voxRes = *voxResPtr;
      voxelResults[voxRes.bsec][trackResiduals.getGlbVoxBin(voxRes.bvox)] = voxRes;
    }
  }
  // voxelResults is only read inside doFileProcessing and is not touched between thread start and join,
  // so it is shared by const reference instead of copied once per thread (copying would cost
  // nFileThreads x 36 x nVoxPerSector VoxRes objects).
  LOGP(info, "Sharing the provided input Map with all threads");

  // Check for bad ranges
  std::vector<range> badRanges;
  std::vector<std::vector<range>> badRanges_thread(nFileThreads);
  bool invertBadRange = false;
  if (badRangeList.length() > 0) {
    if (badRangeList[0] == '-') {
      LOGP(info, "Inverting badRange list!");
      invertBadRange = true;
      badRangeList.erase(0, 1);
    }
    badRanges = loadRunTimeSpans(badRangeList, runNumber, badRangeSelection);
  }
  for (int iThread = 0; iThread < nFileThreads; ++iThread) {
    badRanges_thread[iThread] = badRanges;
  }

  // ---| Lumi estimators |---
  size_t lumiEntriesCTP = 0;
  double lumiSumCTP = 0;

  // vector of selected values
  std::vector<uint32_t> orbitsSel;
  // This macro does not read IDC scalers from CCDB (see the "IDC values" note above the OrbitLumiInfo
  // tree below). These two stay empty and are written out only to keep the output format stable for
  // readers that expect the branches; the values are filled in offline from timeMSsel.
  std::vector<float> idcScalerASel;
  std::vector<float> idcScalerCSel;
  std::vector<float> ctpLumiSel;
  std::vector<long> timeMSsel; // time in ms collected over all selected TFs

  //----------------------------
  const std::filesystem::path pFileOutput(fileOutput);
  std::string outPath(pFileOutput.parent_path().c_str());
  if (outPath.empty()) {
    outPath = ".";
  }

  fair::Logger::SetConsoleSeverity(fair::Severity::error);
  ///////////////////////////
  // Create thread vectors //
  ///////////////////////////
  // trackResiduals (declared earlier, already configured) is shared read-only across threads -- no per-thread copy needed

  long int nEdgeClustersSkipped{0};
  std::vector<long int> nEdgeClustersSkipped_thread(nFileThreads, 0); // Does NEED to be merged
  long int nTracksSkippedByBadRangeList{0};
  std::vector<long int> nTracksSkippedByBadRangeList_thread(nFileThreads, 0); // Does NEED to be merged
  long int nTFs{0};
  long int nTFsSkippedByBadRangeList{0};
  long int nTFsSkippedByTimeWindow{0};
  std::vector<long int> nTFs_thread(nFileThreads, 0);                      // Does NEED to be merged
  std::vector<long int> nTFsSkippedByBadRangeList_thread(nFileThreads, 0); // Does NEED to be merged
  std::vector<long int> nTFsSkippedByTimeWindow_thread(nFileThreads, 0);   // Does NEED to be merged

  std::vector<VoxelData> voxels(NSectors * NRows * nY2XBins * nZ2XBins); // flat [sec][ix][iy][iz] -- shared circle pool, ONE instance for all threads (guarded by per-voxel mutex); sized directly since VoxelData (holds a mutex) cannot be resize()'d after construction  // Does NOT need to be merged

  // residualsAll/NP/PP/NN + their counters now live directly in VoxelData (shared, per-voxel mutex), so no per-thread copies or later merge pass are needed for them.

  // The per-file input handles (TFile, TTreeReaders, TTreeReaderValues) and the I/O-rate sampling state
  // are plain locals inside doFileProcessing: each thread only ever used its own slot, so they never
  // needed to be shared vectors here. Only totalBytesReadPerf is still per-thread, because it is summed
  // across threads after the join below.
  std::vector<Long64_t> totalBytesReadPerf_thread(nFileThreads, 0); // Does NEED to be merged

  std::vector<size_t> lumiEntriesCTP_thread(nFileThreads, 0); // Does NEED to be merged
  std::vector<double> lumiSumCTP_thread(nFileThreads, 0);     // Does NEED to be merged

  // vector of selected values
  std::vector<std::vector<uint32_t>> orbitsSel_thread(nFileThreads); // Does NEED to be merged
  std::vector<std::vector<float>> ctpLumiSel_thread(nFileThreads);   // Does NEED to be merged
  std::vector<std::vector<long>> timeMSsel_thread(nFileThreads);     // time in ms collected over all selected TFs                                                                                                        // Does NEED to be merged

  fair::Logger::SetConsoleSeverity(fair::Severity::info);
  printMemoryUsage("Memory usage after init buffers");

  // Start threads
  std::vector<std::thread> threads(nFileThreads);
  for (int i = 0; i < nFileThreads; i++) {
    threads[i] = std::thread(doFileProcessing,
                             i,
                             nFileThreads,
                             maxTrackWorkers,
                             firstTFTime,
                             lastTFTime,
                             invertBadRange,
                             maxdEdx,
                             maxdEdxExp,
                             maxDevdEdxOverExp,
                             skipEdgePads,
                             std::ref(nEdgeClustersSkipped_thread),
                             std::ref(nTracksSkippedByBadRangeList_thread),
                             std::ref(nTFs_thread),
                             std::ref(nTFsSkippedByBadRangeList_thread),
                             std::ref(nTFsSkippedByTimeWindow_thread),
                             voxMapInput,
                             sources,
                             orbitResetTimeMS,
                             magfieldvalue,
                             fileList,
                             maxTracksPerSlice,
                             maxTracks,
                             std::ref(nTracksProcessed),
                             std::cref(voxelResults),
                             std::ref(badRanges_thread),
                             std::cref(trackResiduals),
                             maxDistIntCls,
                             nY2XBins,
                             nZ2XBins,
                             std::ref(voxels),
                             std::ref(totalBytesReadPerf_thread),
                             std::ref(lumiEntriesCTP_thread),
                             std::ref(lumiSumCTP_thread),
                             std::ref(orbitsSel_thread),
                             std::ref(ctpLumiSel_thread),
                             std::ref(timeMSsel_thread));
  }

  // Wait for the threads to finish
  for (auto& th : threads) {
    th.join();
  }

  //////////////////////////////////////////////////////////////////////
  // ===| CODE TO MERGE VECTORS |=======================================
  //////////////////////////////////////////////////////////////////////
  // START OF MERGE
  nTracksProcessed_final = nTracksProcessed.load();
  Long64_t totalBytesReadPerf = 0; // sum of TTreePerfStats bytes read across all threads/files (network volume)
  for (int iThread = 0; iThread < nFileThreads; ++iThread) {
    // Merge counters
    totalBytesReadPerf += totalBytesReadPerf_thread[iThread];
    lumiEntriesCTP += lumiEntriesCTP_thread[iThread];
    lumiSumCTP += lumiSumCTP_thread[iThread];
    nEdgeClustersSkipped += nEdgeClustersSkipped_thread[iThread];
    nTracksSkippedByBadRangeList += nTracksSkippedByBadRangeList_thread[iThread];
    nTFs += nTFs_thread[iThread];
    nTFsSkippedByBadRangeList += nTFsSkippedByBadRangeList_thread[iThread];
    nTFsSkippedByTimeWindow += nTFsSkippedByTimeWindow_thread[iThread];

    // Merge vector data
    orbitsSel.insert(orbitsSel.end(),
                     orbitsSel_thread[iThread].begin(),
                     orbitsSel_thread[iThread].end());

    ctpLumiSel.insert(ctpLumiSel.end(),
                      ctpLumiSel_thread[iThread].begin(),
                      ctpLumiSel_thread[iThread].end());

    timeMSsel.insert(timeMSsel.end(),
                     timeMSsel_thread[iThread].begin(),
                     timeMSsel_thread[iThread].end());
  }

  {
    const double totalMBReadPerf = totalBytesReadPerf / (1024.0 * 1024.0);
    LOGP(info, "All threads done | read {:.1f} MB total (unbinnedResid, across {} thread(s))", totalMBReadPerf, nFileThreads);
  }

  // Compute final binned residuals directly from the shared per-voxel accumulators in `voxels`
  // (each thread already accumulated straight into the single shared VoxelData under vox.mtx,
  // so there is no per-thread data left to sum over here).
  for (int isec = 0; isec < NSectors; isec++) {
    for (int iz = 0; iz < nZ2XBins; iz++) {
      for (int iy = 0; iy < nY2XBins; iy++) {
        for (int ix = 0; ix < NRows; ix++) {
          const auto& vox = voxels[((isec * NRows + ix) * nY2XBins + iy) * nZ2XBins + iz];

          // weights as in original code
          double weightNP = vox.counterNP * 1.0;
          double weightPP = vox.counterPP * 0.01;
          double weightNN = vox.counterNN * 0.01;
          double sum_weight = weightNP + weightPP + weightNN;

          // For ixyz == 0,1: weighted average as before
          for (int ixyz = 0; ixyz < 2; ++ixyz) {
            if (sum_weight > 0.0) {
              double meanNP = (vox.counterNP > 0) ? (vox.residualsNP[ixyz] / vox.counterNP) : 0.0;
              double meanPP = (vox.counterPP > 0) ? (vox.residualsPP[ixyz] / vox.counterPP) : 0.0;
              double meanNN = (vox.counterNN > 0) ? (vox.residualsNN[ixyz] / vox.counterNN) : 0.0;
              vec_residualsAll[isec][ix][iy][iz][ixyz] = static_cast<float>((weightNP * meanNP + weightPP * meanPP + weightNN * meanNN) / sum_weight);
            } else {
              vec_residualsAll[isec][ix][iy][iz][ixyz] = 0.0f;
            }
          }
          // For ixyz == 2: simple mean; vox.residualsAll[2] is the running sum of DZdist, vox.counterZAll the count
          vec_residualsAll[isec][ix][iy][iz][2] = (vox.counterZAll > 0) ? static_cast<float>(vox.residualsAll[2] / vox.counterZAll) : 0.0f;
          vec_residuals_counterAll[isec][ix][iy][iz] = vox.counterNP + vox.counterPP + vox.counterNN;
        }
      }
    }
  }
  // END OF MERGE
  //////////////////////////////////////////////////////////////////////

  bool isBadCalib = false;
  if ((minTracksPerSlice > 0) && (nTracksProcessed_final < minTracks)) {
    LOGP(error, "Processed tracks: {} ({}), max requested tracks: {} ({}), minimum number of tracks not reached {} ({}), calibration will be marked as bad, skipped {} edge clusters, skipped tracks by badRangeList {} ({}), skipped TFs outside requested time window {} of {} ({:.1f}%)", nTracksProcessed_final, nTracksProcessed_final / nSlices, maxTracks, maxTracksPerSlice, minTracks, minTracksPerSlice, nEdgeClustersSkipped, nTracksSkippedByBadRangeList, nTracksSkippedByBadRangeList / nSlices, nTFsSkippedByTimeWindow, nTFs, (nTFs > 0) ? (100.0 * nTFsSkippedByTimeWindow / nTFs) : 0.0);
    LOGP(info, "Processed time: {}ms, skipped time by badRangeList {}ms ({})", nTFs * o2::constants::lhc::LHCOrbitMUS * 1.e-3 * 32, nTFsSkippedByBadRangeList * o2::constants::lhc::LHCOrbitMUS * 1.e-3 * 32, float(nTFsSkippedByBadRangeList) / float(nTFs));
    const std::string stem = fs::path(fileOutput.data()).stem().c_str();
    std::ofstream(fmt::format("badCalib.{}", stem)).close();
    isBadCalib = true;
  } else {
    LOGP(info, "Processed tracks: {} ({}), max requested tracks: {} ({}), skipped {} edge clusters, skipped tracks by badRangeList {} ({}), skipped TFs outside requested time window {} of {} ({:.1f}%)", nTracksProcessed_final, nTracksProcessed_final / nSlices, maxTracks, maxTracksPerSlice, nEdgeClustersSkipped, nTracksSkippedByBadRangeList, nTracksSkippedByBadRangeList / nSlices, nTFsSkippedByTimeWindow, nTFs, (nTFs > 0) ? (100.0 * nTFsSkippedByTimeWindow / nTFs) : 0.0);
    LOGP(info, "Processed time: {}ms, skipped time by badRangeList {}ms ({})", nTFs * o2::constants::lhc::LHCOrbitMUS * 1.e-3 * 32, nTFsSkippedByBadRangeList * o2::constants::lhc::LHCOrbitMUS * 1.e-3 * 32, float(nTFsSkippedByBadRangeList) / float(nTFs));
  }
  //----------------------------------------------------------------

  // IDC values: this macro deliberately does not query the TPC scalers from CCDB. Doing so per TF from
  // inside the processing loop is slow and unreliable (uncached fetch, object reload on every jump in
  // time), so meanIDC/medianIDC are written as 0 placeholders here. The per-TF timestamps needed to
  // recover them are written to the OrbitLumiInfo tree below (timeMSsel), so the real values can be
  // joined in afterwards, offline, against a properly cached CCDB client.
  float meanIDC = 0.f; // not const: written to a TTree branch below, which needs a mutable address

  double meanCTP = 0;
  if (lumiEntriesCTP > 0) {
    meanCTP = lumiSumCTP / lumiEntriesCTP * (isPbPb ? 2.414 : 1); // 2.414 for PbPb
  }

  TFile* outputfile = new TFile(fileOutput.c_str(), "RECREATE");
  LOGP(info, "Output file: {} created", fileOutput);

  o2::tpc::TrackResiduals::VoxRes mVoxelResultsOut{};                      ///< the results from mVoxelResults are copied in here to be able to stream them
  o2::tpc::TrackResiduals::VoxRes* mVoxelResultsOutPtr{&mVoxelResultsOut}; ///< pointer to set the branch address to for the output
  std::unique_ptr<TTree> mTreeOut;

  // Same tree-alias set as the real TrackResiduals::createOutputFile() (SpacePoints/TrackResiduals.cxx),
  // so this tree can be TTree::Draw()'n the same way downstream regardless of which of the two wrote it.
  if (trackResiduals.getNVoxelsPerSector() == 0) {
    LOGP(warning, "For the tree aliases to work you must initialize the binning before calling createOutputFile()");
  }
  mTreeOut = std::make_unique<TTree>("voxResTree", "voxRes map results and statistics");
  mTreeOut->SetAlias("z2xBin", "bvox[0]");
  mTreeOut->SetAlias("y2xBin", "bvox[1]");
  mTreeOut->SetAlias("xBin", "bvox[2]");
  mTreeOut->SetAlias("z2xAV", "stat[0]");
  mTreeOut->SetAlias("y2xAV", "stat[1]");
  mTreeOut->SetAlias("xAV", "stat[2]");
  mTreeOut->SetAlias("fsector", "bsec+0.5+9.*(y2xAV)/pi");
  mTreeOut->SetAlias("phi", "(bsec%18+0.5+9.*(stat[1])/pi)/9*pi");
  mTreeOut->SetAlias("r", "stat[2]");
  mTreeOut->SetAlias("z", "z2xAV*xAV");
  mTreeOut->SetAlias("dX", "D[0]");
  mTreeOut->SetAlias("dY", "D[1]");
  mTreeOut->SetAlias("dZ", "D[2]");
  mTreeOut->SetAlias("dXS", "DS[0]");
  mTreeOut->SetAlias("dYS", "DS[1]");
  mTreeOut->SetAlias("dZS", "DS[2]");
  mTreeOut->SetAlias("dXE", "E[0]");
  mTreeOut->SetAlias("dYE", "E[1]");
  mTreeOut->SetAlias("dZE", "E[2]");
  mTreeOut->SetAlias("voxelIndex", Form("xBin + %i * (y2xBin + %i * z2xBin) + %i * bsec", trackResiduals.getNXBins(), trackResiduals.getNY2XBins(), trackResiduals.getNVoxelsPerSector()));
  mTreeOut->SetAlias("entries", "stat[3]");
  mTreeOut->SetAlias("fitOK", Form("(flags & %u) == %u", TrackResiduals::DistDone, TrackResiduals::DistDone));
  mTreeOut->SetAlias("dispOK", Form("(flags & %u) == %u", TrackResiduals::DispDone, TrackResiduals::DispDone));
  mTreeOut->SetAlias("smtOK", Form("(flags & %u) == %u", TrackResiduals::SmoothDone, TrackResiduals::SmoothDone));
  mTreeOut->SetAlias("masked", Form("(flags & %u) == %u", TrackResiduals::Masked, TrackResiduals::Masked));
  mTreeOut->Branch("voxRes", &mVoxelResultsOutPtr);

  // Placeholder, filled in offline together with meanIDC -- see the note above.
  float medianIDC = 0.f;
  float medianCTP = static_cast<float>(calculateMedian(ctpLumiSel));
  long medianTimeMS = static_cast<long>(calculateMedian(timeMSsel));
  long meanTimeMS = static_cast<long>(calculateMean(timeMSsel));

  auto userInfo = mTreeOut->GetUserInfo();
  userInfo->Add(new TNamed("meanIDC", std::to_string(meanIDC).data()));
  userInfo->Add(new TNamed("meanCTP", std::to_string(meanCTP).data()));
  userInfo->Add(new TNamed("meanTimeMS", std::to_string(meanTimeMS).data()));
  userInfo->Add(new TNamed("medianIDC", std::to_string(medianIDC).data()));
  userInfo->Add(new TNamed("medianCTP", std::to_string(medianCTP).data()));
  userInfo->Add(new TNamed("medianTimeMS", std::to_string(medianTimeMS).data()));
  userInfo->Add(new TNamed("y2xBinning", y2xBinning.data()));
  userInfo->Add(new TNamed("z2xBinning", z2xBinning.data()));
  // TrackResiduals::setZ2XBinning() scales the physical z/x bin boundaries by scdcalib.maxZ2X -- this
  // value is baked into what each z2x voxel index means in THIS tree, not just a cosmetic config knob.
  // Stage 2 has no scdconfig.ini on the GRID and would otherwise silently reconstruct the binning with
  // the O2 code default (1.0) instead of maxZ2XCut (1.4 in production), misaligning its geometry against
  // the one this tree's voxels were actually filled with. Stored here so stage 2 can apply the exact
  // same value it was built with, not a separately-configured guess.
  userInfo->Add(new TNamed("maxZ2X", std::to_string(maxZ2XCut).data()));
  userInfo->Add(new TNamed("nSlicesPhiZ", std::to_string(nSlices)));
  userInfo->Add(new TNamed("maxTracks", std::to_string(maxTracks)));
  userInfo->Add(new TNamed("minTracks", std::to_string(minTracks)));
  userInfo->Add(new TNamed("nTracksProcessed", std::to_string(nTracksProcessed_final)));
  if (isBadCalib) {
    userInfo->Add(new TNamed("badCalib", "1"));
  }

  for (int isec = 0; isec < NSectors; isec++) {
    for (int iz = 0; iz < nZ2XBins; iz++) {
      for (int iy = 0; iy < nY2XBins; iy++) {
        for (int ix = 0; ix < NRows; ix++) {
          for (int ixyz = 0; ixyz < 3; ixyz++) {
            mVoxelResultsOut.D[ixyz] = vec_residualsAll[isec][ix][iy][iz][ixyz];
            mVoxelResultsOut.DS[ixyz] = vec_residualsAll[isec][ix][iy][iz][ixyz];
            mVoxelResultsOut.DC[ixyz] = vec_residualsAll[isec][ix][iy][iz][ixyz];
            mVoxelResultsOut.E[ixyz] = 0.1;
          }

          float xposvox, yoverxpos, zoverxpos;
          trackResiduals.getVoxelCoordinates(isec, ix, iy, iz, xposvox, yoverxpos, zoverxpos);

          mVoxelResultsOut.stat[0] = static_cast<float>(zoverxpos); // z/x, y/x, x, entries
          mVoxelResultsOut.stat[1] = static_cast<float>(yoverxpos);
          mVoxelResultsOut.stat[2] = static_cast<float>(xposvox);
          mVoxelResultsOut.stat[3] = static_cast<float>(vec_residuals_counterAll[isec][ix][iy][iz]); // number of entries used

          mVoxelResultsOut.EXYCorr = 1.0;
          mVoxelResultsOut.dYSigMAD = 1.0;
          mVoxelResultsOut.dZSigLTM = 1.0;

          mVoxelResultsOut.bvox[0] = iz;
          mVoxelResultsOut.bvox[1] = iy;
          mVoxelResultsOut.bvox[2] = ix;
          mVoxelResultsOut.bsec = isec;
          mVoxelResultsOut.flags = 7;

          mTreeOut->Fill();
        }
      }
    }
  }

  // write orbit and lumi info
  outputfile->cd();
  TTree tOrbitLumi("OrbitLumiInfo", "Orbit and Lumi Info");
  int64_t orbitResetMS = orbitResetTimeMS;
  tOrbitLumi.Branch("orbitResetTimeMS", &orbitResetMS);
  tOrbitLumi.Branch("timeMSsel", &timeMSsel);
  tOrbitLumi.Branch("orbitsSel", &orbitsSel);
  tOrbitLumi.Branch("idcScalerASel", &idcScalerASel);
  tOrbitLumi.Branch("idcScalerCSel", &idcScalerCSel);
  tOrbitLumi.Branch("ctpLumiSel", &ctpLumiSel);
  tOrbitLumi.Branch("meanIDC", &meanIDC);
  tOrbitLumi.Branch("meanCTP", &meanCTP);
  tOrbitLumi.Branch("meanTimeMS", &meanTimeMS);
  tOrbitLumi.Branch("medianIDC", &medianIDC);
  tOrbitLumi.Branch("medianCTP", &medianCTP);
  tOrbitLumi.Branch("medianTimeMS", &medianTimeMS);
  tOrbitLumi.Fill();
  tOrbitLumi.Write();

  {
    TTree tMetaData("MetaData", "Meta data information");

    int nSlicesP = nSlices;
    int maxTracksP = maxTracks;
    int minTracksP = minTracks;

    tMetaData.Branch("runNumber", &runNumber);
    tMetaData.Branch("nSlicesPhiZ", &nSlicesP);
    tMetaData.Branch("maxTracks", &maxTracksP);
    tMetaData.Branch("minTracks", &minTracksP);
    tMetaData.Branch("nTracksProcessed", &nTracksProcessed_final);
    tMetaData.Branch("fileOutput", &fileOutput);
    tMetaData.Branch("voxMapInput", &voxMapInput);
    tMetaData.Branch("trackSources", &trackSources);
    tMetaData.Branch("z2xBinning", &z2xBinning);
    tMetaData.Branch("y2xBinning", &y2xBinning);
    tMetaData.Branch("useSmoothed", &useSmoothed);
    tMetaData.Branch("createSpline", &createSpline);
    tMetaData.Branch("maxTracksPerSlice", &maxTracksPerSlice);
    tMetaData.Branch("minTracksPerSlice", &minTracksPerSlice);
    tMetaData.Branch("badRangeList", &badRangeList);
    tMetaData.Branch("firstTFTime", &firstTFTime);
    tMetaData.Branch("lastTFTime", &lastTFTime);
    tMetaData.Fill();
    tMetaData.Write();
  }

  outputfile->Write();
  //----------------------------------------------------------------

  const std::string fileOutputInfo = fmt::format("{}/{}.txt", outPath, pFileOutput.stem().c_str());
  std::ofstream fInfo(fileOutputInfo);
  fInfo << "meanIDC: " << meanIDC << "\n";
  fInfo << "meanCTP: " << meanCTP << "\n";
  fInfo << "meanTimeMS: " << meanTimeMS << "\n";
  fInfo << "medianIDC: " << medianIDC << "\n";
  fInfo << "medianCTP: " << medianCTP << "\n";
  fInfo << "medianTimeMS: " << medianTimeMS << "\n";
  fInfo.close();
  LOGP(info, "Found meanIDC: {}", meanIDC);
  LOGP(info, "Found meanCTP: {}", meanCTP);
  LOGP(info, "Found meanTimeMS: {}", meanTimeMS);
  LOGP(info, "Found medianIDC: {}", medianIDC);
  LOGP(info, "Found medianCTP: {}", medianCTP);
  LOGP(info, "Found medianTimeMS: {}", medianTimeMS);

  // This macro only produces the raw voxel-residual map. Turning that into a TPCFastTransform is a
  // separate step, done afterwards by TPCFastTransformInitCPM.C on this output, so that the two can be
  // rerun independently. The createSpline argument is kept because it is recorded in the output
  // metadata and consumed by that later step.

  // The per-thread input handles are locals inside doFileProcessing and are already released, in
  // dependency order, when each thread returns -- nothing to tear down here.
  mTreeOut.reset();

  const auto t_end_1 = std::chrono::high_resolution_clock::now();
  LOGP(info, "Wall-clock time for the whole application: {:.1f} s",
       std::chrono::duration<double>(t_end_1 - t_start).count());

  LOGP(info, "Done processing");

  printMemoryUsage("Memory usage at end");
}

std::vector<range> loadRunTimeSpans(const std::string& flname, int onlyRun, const std::string& selection = "ALL")
{
  std::ifstream inputFile(flname);
  if (!inputFile) {
    LOGP(fatal, "Failed to open selected run/timespans file {}", flname);
  }
  LOGP(info, "Reading bad ranges from file {}, for run {}", flname, onlyRun);
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbmgr.setURL("http://alice-ccdb.cern.ch");

  ccdbmgr.setCaching(true);
  ccdbmgr.setFatalWhenNull(false);

  std::vector<range> badRanges;

  std::string line;
  size_t cntl = 0, cntr = 0;
  int64_t orbitResetTimeMS = 0;
  int lastRunOrbitReset = -1;
  while (std::getline(inputFile, line)) {
    cntl++;
    for (char& ch : line) { // Replace semicolons and tabs with spaces for uniform processing
      if (ch == ';' || ch == '\t' || ch == ',') {
        ch = ' ';
      }
    }
    o2::utils::Str::trim(line);
    if (line.size() < 1 || line[0] == '#') {
      continue;
    }
    auto tokens = o2::utils::Str::tokenize(line, ' ');
    auto logError = [&cntl, &line]() { LOGP(error, "Expected format for selection is tripplet <run> <range_min> <range_max>, failed on line#{}: {}", cntl, line); };
    if (tokens.size() >= 3) {
      int run = 0;
      long rmin, rmax;
      try {
        run = std::stoi(tokens[0]);
        rmin = std::stol(tokens[1]);
        rmax = std::stol(tokens[2]);
      } catch (...) {
        logError();
        continue;
      }

      if (onlyRun != run) {
        continue;
      }

      if (selection != "ALL") {
        bool isSelection = false;
        for (int iToken = 3; iToken < int(tokens.size()); ++iToken) {
          if (tokens[iToken] == selection) {
            isSelection = true;
          }
        }
        if (isSelection == false) {
          continue;
        }
      }

      constexpr long ISTimeStamp = 1514761200000L;
      int isTimeStampMin = rmin > ISTimeStamp ? 1 : 0, isTimeStampMax = rmax > ISTimeStamp ? 1 : 0; // values above ISTimeStamp are timestamps (need to be converted to orbits)
      if (rmin > rmax) {
        LOGP(fatal, "Provided range limits are not in increasing order, entry is {}", line);
      }
      if (isTimeStampMin != isTimeStampMax) {
        LOGP(fatal, "Provided range limits should be both consistent either with orbit number or with unix timestamp in ms, entry is {}", line);
      }
      if (isTimeStampMin) {
        if (lastRunOrbitReset != run) {
          LOGP(info, "Input needs conversion from time stamps to orbit");
          const auto [sor, eor] = ccdbmgr.getRunDuration(run);
          const long timeMeanRun = (sor + eor) / 2.;
          const double lengthRun = (eor - sor);
          const auto orbitResetTimeNS = ccdbmgr.getSpecific<std::vector<int64_t>>("CTP/Calib/OrbitReset", timeMeanRun);
          orbitResetTimeMS = (*orbitResetTimeNS)[0] * 1e-3;

          LOGP(info, "Run {}, sor {}, eor {}, duration {} (min)", run, sor, eor, lengthRun / 1000. / 60.);
          LOGP(info, "Orbit reset time in MS is {}", orbitResetTimeMS);
          lastRunOrbitReset = run;
        }
        const auto orbitToMS = o2::constants::lhc::LHCOrbitMUS * 1e-3;
        const auto rMinIn = rmin, rMaxIn = rmax;
        rmin = long((rmin - orbitResetTimeMS) / orbitToMS);
        rmax = long(std::ceil((rmax - orbitResetTimeMS) / orbitToMS));
        LOGP(info, "Run {} input range [{} - {}] ms -> [{} - {}] orbits", run, rMinIn, rMaxIn, rmin, rmax);
      }

      badRanges.emplace_back(rmin, rmax);
      cntr++;
    } else {
      logError();
    }
  }
  return badRanges;
}
