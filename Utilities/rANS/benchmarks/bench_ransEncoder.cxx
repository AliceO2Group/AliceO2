
#include <vector>
#include <cstring>
#include <random>
#include <algorithm>
#include <execution>
#include <iterator>

#include <benchmark/benchmark.h>

#include "rANS/rans.h"
#include "rANS/SimpleEncoder.h"
#include "rANS/SimpleDecoder.h"
#include "rANS/SIMDEncoder.h"
#include "rANS/SIMDDecoder.h"

#ifdef ENABLE_VTUNE_PROFILER
#include <ittnotify.h>
#endif

double_t
  computeExpectedCodewordLength(const o2::rans::FrequencyTable& frequencies, const o2::rans::RenormedFrequencyTable& rescaled)
{
  if (frequencies.getNumSamples() > 0 && frequencies.getMinSymbol() == rescaled.getMinSymbol() && frequencies.getMaxSymbol() == rescaled.getMaxSymbol()) {
    return std::inner_product(frequencies.begin(), frequencies.end(), rescaled.begin(), static_cast<double>(0),
                              std::minus<>(),
                              [&frequencies, &rescaled](const auto& frequency, const auto& rescaledFrequency) {
                                const double_t trueProbability = static_cast<double_t>(frequency) / frequencies.getNumSamples();
                                const double_t rescaledProbability = static_cast<double_t>(rescaledFrequency) / rescaled.getNumSamples();
                                if (rescaledProbability) {
                                  return trueProbability * std::log2(rescaledProbability);
                                } else {
                                  return 0.0;
                                }
                              });
  } else {
    throw std::runtime_error("incompatible frequency tables");
  }
};

inline constexpr size_t MessageSize = 1ull << 22;

template <typename source_T>
class SourceMessageProxy
{
 public:
  SourceMessageProxy(size_t messageSize)
  {
    if (mSourceMessage.empty()) {
      std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
      const size_t draws = std::min(1ul << 20, static_cast<size_t>(std::numeric_limits<source_T>::max()));
      const double probability = 0.5;
      std::binomial_distribution<source_T> dist(draws, probability);
      const size_t sourceSize = messageSize / sizeof(source_T) + 1;
      mSourceMessage.resize(sourceSize);
      std::generate(std::execution::par_unseq, mSourceMessage.begin(), mSourceMessage.end(), [&dist, &mt]() { return dist(mt); });
    }
  }

  const auto& get() const { return mSourceMessage; };

 private:
  std::vector<source_T> mSourceMessage{};
};

inline const SourceMessageProxy<uint8_t> sourceMessage8{MessageSize};
inline const SourceMessageProxy<uint16_t> sourceMessage16{MessageSize};
inline const SourceMessageProxy<uint32_t> sourceMessage32{MessageSize};

template <typename T>
const auto& getMessage()
{
  if constexpr (std::is_same_v<uint8_t, T>) {
    return sourceMessage8.get();
  } else if constexpr (std::is_same_v<uint16_t, T>) {
    return sourceMessage16.get();
  } else {
    return sourceMessage32.get();
  }
};

template <template <class> class Encoder_T, template <class> class Decoder_T, typename source_T>
struct Fixture : public benchmark::Fixture {
 public:
  using source_t = source_T;

  void SetUp(const ::benchmark::State& state) final
  {
    const auto& sourceMessage = getMessage<source_T>();
    o2::rans::FrequencyTable frequencies = o2::rans::makeFrequencyTableFromSamples(std::begin(sourceMessage), std::end(sourceMessage));
    o2::rans::RenormedFrequencyTable rescaledFrequencies = o2::rans::renorm(frequencies);

    encoder = Encoder_T<source_t>{rescaledFrequencies};
    decoder = Decoder_T<source_t>{rescaledFrequencies};

    nUsedAlphabetSymbols = frequencies.getNUsedAlphabetSymbols();
    alphabetRangeBits = encoder.getAlphabetRangeBits();
    symbolTablePrecision = encoder.getSymbolTablePrecision();
    entropy = o2::rans::computeEntropy(frequencies);
    expectedCodewordLength = computeExpectedCodewordLength(frequencies, rescaledFrequencies);
  }

  void TearDown(const ::benchmark::State& state) final
  {
  }

  Encoder_T<source_t> encoder;
  Decoder_T<source_t> decoder;
  size_t nUsedAlphabetSymbols{};
  size_t alphabetRangeBits{};
  size_t symbolTablePrecision{};
  double_t entropy{};
  double_t expectedCodewordLength{};
};

template <class fixture_T>
static void ransCompressionBenchmark(benchmark::State& st, fixture_T& fixture)
{
  const auto& sourceMessage = getMessage<typename fixture_T::source_t>();
  std::vector<uint32_t> encodeBuffer(2 * sourceMessage.size());
  std::vector<typename fixture_T::source_t> decodeBuffer;
  auto encodeBufferEnd = &(*encodeBuffer.end());

#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif
  for (auto _ : st) {
    benchmark::DoNotOptimize(encodeBufferEnd = fixture.encoder.process(sourceMessage.data(), sourceMessage.data() + sourceMessage.size(), encodeBuffer.data()));
  }
#ifdef ENABLE_VTUNE_PROFILER
  __itt_pause();
#endif
  decodeBuffer.clear();
  fixture.decoder.process(encodeBufferEnd, std::back_inserter(decodeBuffer), sourceMessage.size());

  if (!std::equal(sourceMessage.begin(), sourceMessage.end(), decodeBuffer.begin(), decodeBuffer.end())) {
    st.SkipWithError("Missmatch between encoded and decoded Message");
  };

  st.SetItemsProcessed(static_cast<int64_t>(sourceMessage.size()) * static_cast<int64_t>(st.iterations()));
  st.SetBytesProcessed(static_cast<int64_t>(sourceMessage.size()) * sizeof(typename fixture_T::source_t) * static_cast<int64_t>(st.iterations()));
  st.counters["AlphabetRangeBits"] = fixture.alphabetRangeBits;
  st.counters["nUsedAlphabetSymbols"] = fixture.nUsedAlphabetSymbols;
  st.counters["SymbolTablePrecision"] = fixture.symbolTablePrecision;
  st.counters["Entropy"] = fixture.entropy;
  st.counters["ExpectedCodewordLength"] = fixture.expectedCodewordLength;
  st.counters["SourceSize"] = sourceMessage.size() * sizeof(typename fixture_T::source_t);
  st.counters["CompressedSize"] = std::distance(encodeBuffer.data(), encodeBufferEnd) * sizeof(uint32_t);
  st.counters["Compression"] = st.counters["SourceSize"] / static_cast<double>(st.counters["CompressedSize"]);
  st.counters["LowerBound"] = sourceMessage.size() * (static_cast<double>(st.counters["Entropy"]) / 8);
  st.counters["CompressionWRTEntropy"] = st.counters["CompressedSize"] / st.counters["LowerBound"];
};

template <typename source_T>
using simpleRansEncoder_t = typename o2::rans::SimpleEncoder<uint64_t, uint32_t, source_T>;
template <typename source_T>
using simpleRansDecoder_t = typename o2::rans::SimpleDecoder<uint64_t, uint32_t, source_T>;

template <typename source_T>
using sseRansEncoder_t = typename o2::rans::SIMDEncoder<uint64_t, uint32_t, source_T, 8, 2>;
template <typename source_T>
using sseRansDecoder_t = typename o2::rans::SIMDDecoder<uint64_t, uint32_t, source_T, 8, 2>;

template <typename source_T>
using avxRansEncoder_t = typename o2::rans::SIMDEncoder<uint64_t, uint32_t, source_T, 16, 4>;
template <typename source_T>
using avxRansDecoder_t = typename o2::rans::SIMDDecoder<uint64_t, uint32_t, source_T, 16, 4>;

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, simpleRansCompression_64_32_8, simpleRansEncoder_t, simpleRansDecoder_t, uint8_t)
(benchmark::State& st)
{
  ransCompressionBenchmark(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, simpleRansCompression_64_32_16, simpleRansEncoder_t, simpleRansDecoder_t, uint16_t)
(benchmark::State& st)
{
  ransCompressionBenchmark(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, simpleRansCompression_64_32_32, simpleRansEncoder_t, simpleRansDecoder_t, uint32_t)
(benchmark::State& st)
{
  ransCompressionBenchmark(st, *this);
};
BENCHMARK_TEMPLATE_DEFINE_F(Fixture, ransCompression_64_32_8, o2::rans::Encoder64, o2::rans::Decoder64, uint8_t)
(benchmark::State& st)
{
  ransCompressionBenchmark(st, *this);
};
BENCHMARK_TEMPLATE_DEFINE_F(Fixture, ransCompression_64_32_16, o2::rans::Encoder64, o2::rans::Decoder64, uint16_t)
(benchmark::State& st)
{
  ransCompressionBenchmark(st, *this);
};
BENCHMARK_TEMPLATE_DEFINE_F(Fixture, ransCompression_64_32_32, o2::rans::Encoder64, o2::rans::Decoder64, uint32_t)
(benchmark::State& st)
{
  ransCompressionBenchmark(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, SSEransCompression_64_32_8, sseRansEncoder_t, sseRansDecoder_t, uint8_t)
(benchmark::State& st)
{
  ransCompressionBenchmark(st, *this);
};
BENCHMARK_TEMPLATE_DEFINE_F(Fixture, SSEransCompression_64_32_16, sseRansEncoder_t, sseRansDecoder_t, uint16_t)
(benchmark::State& st)
{
  ransCompressionBenchmark(st, *this);
};
BENCHMARK_TEMPLATE_DEFINE_F(Fixture, SSEransCompression_64_32_32, sseRansEncoder_t, sseRansDecoder_t, uint32_t)
(benchmark::State& st)
{
  ransCompressionBenchmark(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, AVXransCompression_64_32_8, avxRansEncoder_t, avxRansDecoder_t, uint8_t)
(benchmark::State& st)
{
  ransCompressionBenchmark(st, *this);
};
BENCHMARK_TEMPLATE_DEFINE_F(Fixture, AVXransCompression_64_32_16, avxRansEncoder_t, avxRansDecoder_t, uint16_t)
(benchmark::State& st)
{
  ransCompressionBenchmark(st, *this);
};
BENCHMARK_TEMPLATE_DEFINE_F(Fixture, AVXransCompression_64_32_32, avxRansEncoder_t, avxRansDecoder_t, uint32_t)
(benchmark::State& st)
{
  ransCompressionBenchmark(st, *this);
};

inline constexpr size_t nIterations8 = 2000;
inline constexpr size_t nIterations16 = 4000;
inline constexpr size_t nIterations32 = 8000;

BENCHMARK_REGISTER_F(Fixture, simpleRansCompression_64_32_8);
BENCHMARK_REGISTER_F(Fixture, simpleRansCompression_64_32_16);
BENCHMARK_REGISTER_F(Fixture, simpleRansCompression_64_32_32);
BENCHMARK_REGISTER_F(Fixture, ransCompression_64_32_8);
BENCHMARK_REGISTER_F(Fixture, ransCompression_64_32_16);
BENCHMARK_REGISTER_F(Fixture, ransCompression_64_32_32);
BENCHMARK_REGISTER_F(Fixture, SSEransCompression_64_32_8);
BENCHMARK_REGISTER_F(Fixture, SSEransCompression_64_32_16);
BENCHMARK_REGISTER_F(Fixture, SSEransCompression_64_32_32);
// BENCHMARK_REGISTER_F(Fixture, AVXransCompression_64_32_8);
// BENCHMARK_REGISTER_F(Fixture, AVXransCompression_64_32_16);
// BENCHMARK_REGISTER_F(Fixture, AVXransCompression_64_32_32);

// BENCHMARK_REGISTER_F(Fixture, simpleRansCompression_64_32_8)->Iterations(nIterations8);
// BENCHMARK_REGISTER_F(Fixture, simpleRansCompression_64_32_16)->Iterations(nIterations16);
// BENCHMARK_REGISTER_F(Fixture, simpleRansCompression_64_32_32)->Iterations(nIterations32);
// BENCHMARK_REGISTER_F(Fixture, ransCompression_64_32_8)->Iterations(nIterations8);
// BENCHMARK_REGISTER_F(Fixture, ransCompression_64_32_16)->Iterations(nIterations16);
// BENCHMARK_REGISTER_F(Fixture, ransCompression_64_32_32)->Iterations(nIterations32);
// BENCHMARK_REGISTER_F(Fixture, SSEransCompression_64_32_8)->Iterations(nIterations8);
// BENCHMARK_REGISTER_F(Fixture, SSEransCompression_64_32_16)->Iterations(nIterations16);
// BENCHMARK_REGISTER_F(Fixture, SSEransCompression_64_32_32)->Iterations(nIterations32);
BENCHMARK_REGISTER_F(Fixture, AVXransCompression_64_32_8)->Iterations(nIterations8);
BENCHMARK_REGISTER_F(Fixture, AVXransCompression_64_32_16)->Iterations(nIterations16);
BENCHMARK_REGISTER_F(Fixture, AVXransCompression_64_32_32)->Iterations(nIterations32);

BENCHMARK_MAIN();