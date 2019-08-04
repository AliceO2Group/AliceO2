// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_LOGGER_H_
#define O2_FRAMEWORK_LOGGER_H_

// FIXME: until we actually have fmt widely available we simply print out the
// format string.
// If FairLogger is not available, we use fmt::printf directly, with no level.
#if __has_include(<fairlogger/Logger.h>)
#include <fairlogger/Logger.h>
#if __has_include(<fmt/format.h>)
#include <fmt/format.h>
#include <fmt/printf.h>
FMT_BEGIN_NAMESPACE
template <typename S, typename Char = FMT_CHAR(S)>
inline int vfprintf(fair::Logger& logger,
                    const S& format,
                    basic_format_args<typename basic_printf_context_t<
                      internal::basic_buffer<Char>>::type>
                      args)
{
  basic_memory_buffer<Char> buffer;
  printf(buffer, to_string_view(format), args);
  logger << std::string_view(buffer.data(), buffer.size());
  return static_cast<int>(buffer.size());
}

template <typename S, typename... Args>
inline FMT_ENABLE_IF_T(internal::is_string<S>::value, int)
  fprintf(fair::Logger& logger,
          const S& format_str, const Args&... args)
{
  internal::check_format_string<Args...>(format_str);
  typedef internal::basic_buffer<FMT_CHAR(S)> buffer;
  typedef typename basic_printf_context_t<buffer>::type context;
  format_arg_store<context, Args...> as{args...};
  return vfprintf(logger, to_string_view(format_str),
                  basic_format_args<context>(as));
}

FMT_END_NAMESPACE

#define LOGF(severity, ...)                                                                                                                                        \
  for (bool fairLOggerunLikelyvariable = false; fair::Logger::Logging(fair::Severity::severity) && !fairLOggerunLikelyvariable; fairLOggerunLikelyvariable = true) \
  fmt::fprintf(fair::Logger(fair::Severity::severity, __FILE__, CONVERTTOSTRING(__LINE__), __FUNCTION__).Log(), __VA_ARGS__)
#define LOGP(level, ...) LOG(level) << fmt::format(__VA_ARGS__)
#else
#define O2_FIRST_ARG(N, ...) N
#define LOGF(level, ...) LOG(level) << O2_FIRST_ARG(__VA_ARGS__)
#define LOGP(level, ...) LOG(level) << O2_FIRST_ARG(__VA_ARGS__)
#endif
#define O2DEBUG(...) LOGF(debug, __VA_ARGS__)
#define O2INFO(...) LOGF(info, __VA_ARGS__)
#define O2ERROR(...) LOGF(error, __VA_ARGS__)
#elif __has_include(<fmt/format.h>)
#include <fmt/format.h>
#define LOGF(level, ...) fmt::printf(__VA_ARGS__)
#define LOGP(level, ...) fmt::print(__VA_ARGS__)
#define O2DEBUG(...) LOGF("dummy", __VA_ARGS__)
#define O2INFO(...) LOGF("dummy", __VA_ARGS__)
#define O2ERROR(...) LOGF("dummy", __VA_ARGS__)
#endif

#endif // O2_FRAMEWORK_LOGGER_H_
