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

#ifndef O2_SIMSETUP_STATICTLSRESERVE_H
#define O2_SIMSETUP_STATICTLSRESERVE_H

#include <cerrno>
#include <charconv>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unistd.h>

namespace o2
{

/// Static TLS reserve in bytes for a Geant4 built with the initial-exec TLS model and loaded by dlopen
constexpr unsigned long Geant4StaticTLSReserve = 32768;

/// The GLIBC_TUNABLES value with a static TLS reserve of at least `reserve` bytes, or an empty string if
/// `tunables` already has one. As in glibc, the last reserve counts and numbers may be hex (0x) or octal (0).
/// Other tunables are kept; entries without a value are dropped, since glibc then ignores the whole string.
inline std::string withStaticTLSReserve(const char* tunables, unsigned long reserve = Geant4StaticTLSReserve)
{
  constexpr std::string_view name = "glibc.rtld.optional_static_tls";
  std::string kept;
  bool haveReserve = false;
  bool malformed = false;
  unsigned long current = 0;
  std::string_view rest = tunables ? tunables : "";
  while (!rest.empty()) {
    const auto colon = rest.find(':');
    const std::string_view entry = rest.substr(0, colon);
    rest = colon == std::string_view::npos ? std::string_view{} : rest.substr(colon + 1);
    const auto equal = entry.find('=');
    if (equal == std::string_view::npos || equal == 0) {
      malformed = true; // also the truncated reserve as it arrives in GRID jobs
      continue;
    }
    if (entry.substr(0, equal) == name) {
      std::string_view value = entry.substr(equal + 1);
      int base = 10;
      if (value.size() > 2 && value[0] == '0' && (value[1] == 'x' || value[1] == 'X')) {
        base = 16;
        value.remove_prefix(2);
      } else if (value.size() > 1 && value[0] == '0') {
        base = 8;
        value.remove_prefix(1);
      }
      unsigned long bytes = 0;
      const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), bytes, base);
      if (!value.empty() && ec == std::errc() && end == value.data() + value.size()) {
        haveReserve = true;
        current = bytes;
      } else {
        haveReserve = false; // a broken last value leaves no reserve
      }
      continue; // our own entry replaces all of them
    }
    kept += (kept.empty() ? "" : ":") + std::string(entry);
  }
  if (!malformed && haveReserve && current >= reserve) {
    return {};
  }
  return std::string(name) + "=" + std::to_string(reserve) + (kept.empty() ? "" : ":" + kept);
}

/// Sets the static TLS reserve in the environment, for the programs started from here. Returns true if it changed.
inline bool setStaticTLSReserve(unsigned long reserve = Geant4StaticTLSReserve)
{
  const auto tunables = withStaticTLSReserve(getenv("GLIBC_TUNABLES"), reserve);
  if (tunables.empty()) {
    return false;
  }
  setenv("GLIBC_TUNABLES", tunables.c_str(), 1);
  return true;
}

/// Restarts the program once with the static TLS reserve if it lacks one. glibc reads GLIBC_TUNABLES only at
/// program start, so the reserve cannot be set later. Call this first in main, before any thread is started.
inline void ensureStaticTLSReserve([[maybe_unused]] char* argv[],
                                   [[maybe_unused]] unsigned long reserve = Geant4StaticTLSReserve)
{
#ifdef __linux__
  constexpr const char* guard = "O2_STATIC_TLS_REEXEC";
  if (getenv(guard)) {
    unsetenv(guard); // restarted already; do not loop if glibc ignores the tunable
    return;
  }
  if (withStaticTLSReserve(getenv("GLIBC_TUNABLES"), reserve).empty()) {
    return;
  }
  // the real path keeps the process name; /proc/self/exe would rename it to "exe"
  char self[PATH_MAX];
  const auto length = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (length <= 0) {
    return;
  }
  self[length] = '\0';
  const char* base = strrchr(self, '/');
  if (strncmp(base ? base + 1 : self, "ld-", 3) == 0) {
    return; // started through the dynamic loader, which a restart would not reproduce
  }
  setStaticTLSReserve(reserve);
  setenv(guard, "1", 1);
  fprintf(stderr, "Restarting %s with GLIBC_TUNABLES=%s for the static TLS of Geant4\n", self, getenv("GLIBC_TUNABLES"));
  execv(self, argv);
  const int error = errno;
  unsetenv(guard);
  fprintf(stderr, "Restart of %s failed: %s\n", self, strerror(error));
#endif
}

} // namespace o2

#endif
