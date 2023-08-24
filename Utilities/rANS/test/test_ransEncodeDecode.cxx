// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   test_ransEncodeDecode.h
/// @author Michael Lettrich
/// @brief  Test rANS encoder/ decoder

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <vector>
#include <cstring>

#include <boost/test/unit_test.hpp>
#include <boost/mp11.hpp>

#include <gsl/span>

#include "rANS/factory.h"
#include "rANS/histogram.h"
#include "rANS/encode.h"

using namespace o2::rans;

inline const std::string str = R"(Sed ut perspiciatis, unde omnis iste natus error sit voluptatem accusantium
doloremque laudantium, totam rem aperiam eaque ipsa, quae ab illo inventore veritatis
et quasi architecto beatae vitae dicta sunt, explicabo. nemo enim ipsam voluptatem,
quia voluptas sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores
eos, qui ratione voluptatem sequi nesciunt, neque porro quisquam est, qui dolorem ipsum,
quia dolor sit, amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora
incidunt, ut labore et dolore magnam aliquam quaerat voluptatem. ut enim ad minima veniam,
quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea
commodi consequatur? quis autem vel eum iure reprehenderit, qui in ea voluptate velit
esse, quam nihil molestiae consequatur, vel illum, qui dolorem eum fugiat,
quo voluptas nulla pariatur?)";

template <typename T>
struct Empty {
  inline static std::vector<T> Data{};

  using source_type = T;
  using iterator_type = decltype(Data.begin());
};

template <typename T>
struct Full {

 public:
  inline static std::vector<T> Data{str.begin(), str.end()};

  using source_type = T;
  using iterator_type = decltype(Data.begin());
};

template <typename L>
struct hasSameTemplateParam : std::is_same<typename boost::mp11::mp_at_c<L, 0>::source_type, typename boost::mp11::mp_at_c<L, 1>::source_type> {
};

using source_types = boost::mp11::mp_list<int8_t, int16_t, int32_t>;

using testInput_templates = boost::mp11::mp_list<boost::mp11::mp_quote<Empty>, boost::mp11::mp_quote<Full>>;

using testInputAll_types = boost::mp11::mp_product<boost::mp11::mp_invoke_q, testInput_templates, source_types>;
using testInputProduct_types = boost::mp11::mp_product<boost::mp11::mp_list, testInputAll_types, testInputAll_types>;
using testInput_types = boost::mp11::mp_copy_if<testInputProduct_types, hasSameTemplateParam>;

using coder_types = boost::mp11::mp_list<std::integral_constant<CoderTag, CoderTag::Compat>
#ifdef RANS_SINGLE_STREAM
                                         ,
                                         std::integral_constant<CoderTag, CoderTag::SingleStream>
#endif /* RANS_SINGLE_STREAM */
#ifdef RANS_SEE
                                         ,
                                         std::integral_constant<CoderTag, CoderTag::SSE>
#endif /*RANS_SSE */
#ifdef RANS_AVX2
                                         ,
                                         std::integral_constant<CoderTag, CoderTag::AVX2>
#endif /* RANS_AVX2 */
                                         >;

using testCase_types = boost::mp11::mp_product<boost::mp11::mp_list, coder_types, testInput_types>;

inline constexpr size_t RansRenormingPrecision = 16;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_encodeDecode, test_types, testCase_types)
{
  using coder_type = boost::mp11::mp_at_c<test_types, 0>;
  using testCase_types = boost::mp11::mp_at_c<test_types, 1>;
  using dictString_type = boost::mp11::mp_at_c<testCase_types, 0>;
  using encodeString_type = boost::mp11::mp_at_c<testCase_types, 1>;
  using stream_type = uint32_t;
  using source_type = typename dictString_type::source_type;

  constexpr CoderTag coderTag = coder_type::value;
  const auto& dictString = dictString_type::Data;
  const auto& encodeString = encodeString_type::Data;

  // TODO(milettri): renorming is not satisfactory.
  size_t precision = dictString.size() == 0 ? 0 : RansRenormingPrecision;
  auto renormed = renorm(makeDenseHistogram::fromSamples(dictString.begin(), dictString.end()), precision);
  auto encoder = makeDenseEncoder<coderTag>::fromRenormed(renormed);
  auto decoder = makeDecoder<>::fromRenormed(renormed);

  if (dictString == encodeString) {
    std::vector<stream_type> encodeBuffer(encodeString.size());
    auto encodeBufferEnd = encoder.process(encodeString.begin(), encodeString.end(), encodeBuffer.begin());
    std::vector<stream_type> encodeBuffer2(encodeString.size());
    auto encodeBuffer2End = encoder.process(gsl::span<const source_type>(encodeString), gsl::make_span(encodeBuffer2));

    BOOST_CHECK_EQUAL_COLLECTIONS(encodeBuffer.begin(), encodeBufferEnd, encodeBuffer2.data(), encodeBuffer2End);

    std::vector<source_type> decodeBuffer(encodeString.size());
    decoder.process(encodeBufferEnd, decodeBuffer.begin(), encodeString.size(), encoder.getNStreams());

    BOOST_CHECK_EQUAL_COLLECTIONS(decodeBuffer.begin(), decodeBuffer.end(), encodeString.begin(), encodeString.end());
  }

  std::vector<source_type> literals(encodeString.size());
  std::vector<stream_type> encodeBuffer(encodeString.size());
  auto [encodeBufferEnd, literalBufferEnd] = encoder.process(encodeString.begin(), encodeString.end(), encodeBuffer.begin(), literals.begin());
  std::vector<stream_type> encodeBuffer2(encodeString.size());
  std::vector<source_type> literals2(encodeString.size());
  auto [encodeBuffer2End, literalBuffer2End] = encoder.process(gsl::span<const source_type>(encodeString), gsl::make_span(encodeBuffer2), literals2.begin());

  BOOST_CHECK_EQUAL_COLLECTIONS(encodeBuffer.begin(), encodeBufferEnd, encodeBuffer2.data(), encodeBuffer2End);
  BOOST_CHECK_EQUAL_COLLECTIONS(literals.begin(), literalBufferEnd, literals2.begin(), literalBuffer2End);

  std::vector<source_type> decodeBuffer(encodeString.size());
  decoder.process(encodeBufferEnd, decodeBuffer.begin(), encodeString.size(), encoder.getNStreams(), literalBufferEnd);

  BOOST_CHECK_EQUAL_COLLECTIONS(decodeBuffer.begin(), decodeBuffer.end(), encodeString.begin(), encodeString.end());
};

#ifndef RANS_SINGLE_STREAM
BOOST_AUTO_TEST_CASE(test_NoSingleStream)
{
  BOOST_TEST_WARN(" Tests were not Compiled for a uint128_t capable CPU, cannot run all tests");
}
#endif /* RANS_SINGLE_STREAM */
#ifndef RANS_SSE
BOOST_AUTO_TEST_CASE(test_NoSSE)
{
  BOOST_TEST_WARN("Tests were not Compiled for a SSE 4.2 capable CPU, cannot run all tests");
}
#endif /* RANS_SSE */
#ifndef RANS_AVX2
BOOST_AUTO_TEST_CASE(test_NoAVX2)
{
  BOOST_TEST_WARN("Tests were not Compiled for a AVX2 capable CPU, cannot run all tests");
}
#endif /* RANS_AVX2 */