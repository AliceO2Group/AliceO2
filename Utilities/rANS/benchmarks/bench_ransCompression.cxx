
#include <vector>
#include <cstring>
#include <random>
#include <algorithm>
#include <execution>

#include <benchmark/benchmark.h>

#include "rANS/SimpleEncoder.h"
#include "rANS/SimpleDecoder.h"
#include "rANS/rans.h"

inline constexpr size_t SourceSize = o2::rans::internal::pow2(24) + 3;

double_t computeExpectedCodewordLength(const o2::rans::FrequencyTable& frequencies, const o2::rans::RenormedFrequencyTable& rescaled)
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

template <typename coder_T, typename stream_T, typename source_T>
struct Fixture : public benchmark::Fixture {
 public:
  using source_t = source_T;
  using stream_t = stream_T;
  using coder_t = coder_T;

  void SetUp(const ::benchmark::State& state) final
  {
    std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
    const size_t draws = std::numeric_limits<source_t>::max();
    const double probability = 0.5;
    std::binomial_distribution<source_t> dist(draws, probability);

    if (sourceMessage.empty()) {
      sourceMessage.resize(SourceSize);

      std::generate(std::execution::par_unseq, sourceMessage.begin(), sourceMessage.end(), [&dist, &mt]() { return dist(mt); });
    }
  }

  void TearDown(const ::benchmark::State& state) final
  {
  }

  std::vector<source_t> sourceMessage{};
};

template <typename coder_T, typename stream_T, typename source_T>
void ransCompressionBenchmark(benchmark::State& st, const std::vector<source_T>& sourceMessage)
{
  const size_t SymbolTablePrecision = st.range(0);

  o2::rans::FrequencyTable frequencies = o2::rans::makeFrequencyTableFromSamples(std::begin(sourceMessage), std::end(sourceMessage));
  o2::rans::RenormedFrequencyTable rescaledFrequencies = o2::rans::renorm(frequencies, SymbolTablePrecision);

  o2::rans::SimpleEncoder<coder_T, stream_T, source_T> encoder{rescaledFrequencies};
  o2::rans::SimpleDecoder<coder_T, stream_T, source_T> decoder{rescaledFrequencies};

  std::vector<stream_T> encodeBuffer{};
  std::vector<source_T> decodeBuffer{};

  for (auto _ : st) {
    st.PauseTiming();

    encodeBuffer.clear();
    decodeBuffer.clear();

    st.ResumeTiming();
    encoder.process(std::begin(sourceMessage), std::end(sourceMessage), std::back_inserter(encodeBuffer));
    decoder.process(encodeBuffer.end(), std::back_inserter(decodeBuffer), sourceMessage.size());
    st.PauseTiming();

    if (!std::equal(sourceMessage.begin(), sourceMessage.end(), decodeBuffer.begin(), decodeBuffer.end())) {
      st.SkipWithError("Missmatch between encoded and decoded Message");
    };
  }

  st.SetItemsProcessed(sourceMessage.size() * st.iterations());
  st.SetBytesProcessed(sourceMessage.size() * sizeof(source_T) * st.iterations());
  st.counters["SourceSize"] = sourceMessage.size() * sizeof(source_T);
  st.counters["CompressedSize"] = encodeBuffer.size() * sizeof(stream_T);
  st.counters["Compression"] = st.counters["SourceSize"] / static_cast<double>(st.counters["CompressedSize"]);
  st.counters["nUsedAlphabetSymbols"] = frequencies.getNUsedAlphabetSymbols();
  st.counters["Entropy"] = o2::rans::computeEntropy(frequencies);
  st.counters["ExpectedCodewordLength"] = computeExpectedCodewordLength(frequencies, rescaledFrequencies);
  st.counters["LowerBound"] = sourceMessage.size() * (static_cast<double>(st.counters["Entropy"]) / 8);
  st.counters["CompressionWRTEntropy"] = st.counters["CompressedSize"] / st.counters["LowerBound"];
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, ransCompression_64_32_8, uint64_t, uint32_t, uint8_t)
(benchmark::State& st)
{
  ransCompressionBenchmark<coder_t, stream_t, source_t>(st, sourceMessage);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, ransCompression_64_32_16, uint64_t, uint32_t, uint16_t)
(benchmark::State& st)
{
  ransCompressionBenchmark<coder_t, stream_t, source_t>(st, sourceMessage);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, ransCompression_64_32_32, uint64_t, uint32_t, uint32_t)
(benchmark::State& st)
{
  ransCompressionBenchmark<coder_t, stream_t, source_t>(st, sourceMessage);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, ransCompression_64_16_8, uint64_t, uint16_t, uint8_t)
(benchmark::State& st)
{
  ransCompressionBenchmark<coder_t, stream_t, source_t>(st, sourceMessage);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, ransCompression_64_16_16, uint64_t, uint16_t, uint16_t)
(benchmark::State& st)
{
  ransCompressionBenchmark<coder_t, stream_t, source_t>(st, sourceMessage);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, ransCompression_64_16_32, uint64_t, uint16_t, uint32_t)
(benchmark::State& st)
{
  ransCompressionBenchmark<coder_t, stream_t, source_t>(st, sourceMessage);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, ransCompression_32_16_8, uint32_t, uint16_t, uint8_t)
(benchmark::State& st)
{
  ransCompressionBenchmark<coder_t, stream_t, source_t>(st, sourceMessage);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, ransCompression_32_16_16, uint32_t, uint16_t, uint8_t)
(benchmark::State& st)
{
  ransCompressionBenchmark<coder_t, stream_t, source_t>(st, sourceMessage);
};

BENCHMARK_REGISTER_F(Fixture, ransCompression_64_32_8)->DenseRange(16, 28, 2);
BENCHMARK_REGISTER_F(Fixture, ransCompression_64_32_16)->DenseRange(18, 28, 2);
BENCHMARK_REGISTER_F(Fixture, ransCompression_64_32_32)->DenseRange(26, 28, 2);
BENCHMARK_REGISTER_F(Fixture, ransCompression_64_16_8)->DenseRange(16, 28, 2);
BENCHMARK_REGISTER_F(Fixture, ransCompression_64_16_16)->DenseRange(18, 28, 2);
BENCHMARK_REGISTER_F(Fixture, ransCompression_64_16_32)->DenseRange(26, 28, 2);
BENCHMARK_REGISTER_F(Fixture, ransCompression_32_16_8)->DenseRange(16, 24, 2);
BENCHMARK_REGISTER_F(Fixture, ransCompression_32_16_16)->DenseRange(18, 24, 2);

BENCHMARK_MAIN();