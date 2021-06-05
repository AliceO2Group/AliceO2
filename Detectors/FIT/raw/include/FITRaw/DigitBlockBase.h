// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//file DigitBlockBase.h base class for processing RAW data into Digits
//
// Artur.Furs
// afurs@cern.ch

#ifndef ALICEO2_FIT_DIGITBLOCKBASE_H_
#define ALICEO2_FIT_DIGITBLOCKBASE_H_
#include <iostream>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <utility>
#include <array>
#include <Rtypes.h>
#include <CommonDataFormat/InteractionRecord.h>
#include <CommonDataFormat/RangeReference.h>
#include <Framework/Logger.h>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/remove_if.hpp>
#include <boost/mpl/lambda.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/size.hpp>

#include <gsl/span>
namespace o2
{
namespace fit
{

namespace DigitBlockHelper
{
//Check template specialisation
//Is there analog of this metafunction in Common O2 lib?
template <template <typename...> class Template, typename T>
struct IsSpecOfType : std::false_type {
};
template <template <typename...> class Template, typename... T>
struct IsSpecOfType<Template, Template<T...>> : std::true_type {
};
//Check if RangeReference is a single field in main digit structure
template <typename T, typename = void>
struct HasRef : std::false_type {
};
template <typename T>
struct HasRef<T, std::enable_if_t<std::is_same<decltype(std::declval<T>().ref), typename o2::dataformats::RangeReference<int, int>>::value>> : std::true_type {
};
//For FV0
template <typename T>
struct HasRef<T, std::enable_if_t<std::is_same<decltype(std::declval<T>().ref), typename o2::dataformats::RangeRefComp<6>>::value>> : std::true_type {
};

//Check if RangeReference is an array field in main digit structure
template <typename T, typename = void>
struct HasArrayRef : std::false_type {
};
template <typename T>
struct HasArrayRef<T, std::enable_if_t<std::is_same<decltype(std::declval<T>().ref), typename std::array<typename o2::dataformats::RangeReference<int, int>, std::tuple_size<decltype(std::declval<T>().ref)>::value>>::value>> : std::true_type {
};
//For FV0
template <typename T>
struct HasArrayRef<T, std::enable_if_t<std::is_same<decltype(std::declval<T>().ref), typename std::array<typename o2::dataformats::RangeRefComp<6>, std::tuple_size<decltype(std::declval<T>().ref)>::value>>::value>> : std::true_type {
};

//Get RangeReference number of dimentions.
template <typename T, typename = void>
struct GetDigitRefsN {
  constexpr static std::size_t value = 0;
};
template <typename T>
struct GetDigitRefsN<T, std::enable_if_t<HasRef<T>::value>> {
  constexpr static std::size_t value = 1;
};
template <typename T>
struct GetDigitRefsN<T, std::enable_if_t<HasArrayRef<T>::value && (std::tuple_size<decltype(std::declval<T>().ref)>::value > 1)>> {
  constexpr static std::size_t value = std::tuple_size<decltype(std::declval<T>().ref)>::value;
};
//Check if InteractionRecord field exists
template <typename T, typename = void>
struct HasIntRecord : std::false_type {
};
template <typename T>
struct HasIntRecord<T, std::enable_if_t<std::is_same<decltype(std::declval<T>().mIntRecord), o2::InteractionRecord>::value>> : std::true_type {
};
//Temporary for FV0
template <typename T>
struct HasIntRecord<T, std::enable_if_t<std::is_same<decltype(std::declval<T>().ir), o2::InteractionRecord>::value>> : std::true_type {
};
//Dividing to sub-digit structures with InteractionRecord(single one, e.g. for TCM extended mode) and without
template <typename T>
using GetVecSubDigit = typename boost::mpl::remove_if<T, boost::mpl::lambda<HasIntRecord<boost::mpl::_1>>::type>::type;
template <typename T>
using GetVecSingleSubDigit = typename boost::mpl::remove_if<T, boost::mpl::lambda<boost::mpl::not_<HasIntRecord<boost::mpl::_1>>>::type>::type;
//Converting empty Boost MPL vector to empty std::tuple
template <typename T, typename = void>
struct GetSubDigitField {
  typedef std::tuple<> type;
  typedef std::tuple<> vector_type;
  constexpr static bool sIsEmpty = true;
  constexpr static bool sIsTuple = true;
  constexpr static std::size_t size = 0;
};
//Converting Boost MPL vector with 1 element to std::vector
template <typename T>
struct GetSubDigitField<T, std::enable_if_t<boost::mpl::size<T>::value == 1>> {
  typedef std::vector<typename boost::mpl::front<T>::type> vector_type;
  typedef typename boost::mpl::front<T>::type type;
  constexpr static bool sIsEmpty = false;
  constexpr static bool sIsTuple = false;
  constexpr static std::size_t size = 1;
};
//
//Converting Boost MPL vector to std::tuple of std::vectors
template <typename T>
struct GetSubDigitField<T, std::enable_if_t<(boost::mpl::size<T>::value > 1)>> {
  template <typename Arg1, typename Arg2>
  struct MakeTuple;
  template <typename... Args, typename LastArg>
  struct MakeTuple<std::tuple<Args...>, LastArg> {
    typedef std::tuple<Args..., LastArg> type;
  };
  typedef typename boost::mpl::fold<T, std::tuple<>, MakeTuple<boost::mpl::_1, std::vector<boost::mpl::_2>>>::type vector_type;
  typedef typename boost::mpl::fold<T, std::tuple<>, MakeTuple<boost::mpl::_1, boost::mpl::_2>>::type type;
  constexpr static bool sIsEmpty = false;
  constexpr static bool sIsTuple = true;
  constexpr static std::size_t size = boost::mpl::size<T>::value;
};

} // namespace DigitBlockHelper

// DigitBlock - digit block, for interacting with Digits structures as one
// Digit - primary digit struct which should contain InteractionRecord field and ReferenceRange to SubDigit(not all of them)
// SubDigit - sub digit structure which shouldn't contain InteractionRecord field
// SingleSubDigit - separated SubDigits which contain InteractionRecord field and not referred by ReferenceRange field in Digit structure.
// For example extended TCM mode uses such SingleSubDigits
template <typename DigitType, typename... SubDigitTypes>
class DigitBlockBase //:public DigitBlock
{
 public:
  DigitBlockBase(o2::InteractionRecord intRec)
  {
    mDigit.setIntRecord(intRec);
  }
  DigitBlockBase(const DigitType& digit) : mDigit(digit)
  {
  }
  DigitBlockBase() = default;
  DigitBlockBase(const DigitBlockBase& other) = default;
  ~DigitBlockBase() = default;
  typedef DigitType Digit_t;
  typedef boost::mpl::vector<SubDigitTypes...> VecAllSubDigit_t;
  typedef DigitBlockHelper::GetVecSubDigit<VecAllSubDigit_t> VecSubDigit_t;
  typedef DigitBlockHelper::GetVecSingleSubDigit<VecAllSubDigit_t> VecSingleSubDigit_t;
  typedef typename DigitBlockHelper::GetSubDigitField<VecSubDigit_t>::vector_type SubDigit_t;
  typedef typename DigitBlockHelper::GetSubDigitField<VecSingleSubDigit_t>::type SingleSubDigit_t;
  constexpr static std::size_t sNSubDigits = DigitBlockHelper::GetSubDigitField<VecSubDigit_t>::size;
  constexpr static std::size_t sNSingleSubDigits = DigitBlockHelper::GetSubDigitField<VecSingleSubDigit_t>::size;
  Digit_t mDigit;
  SubDigit_t mSubDigit;
  SingleSubDigit_t mSingleSubDigit;
  template <typename... T>
  auto getSubDigits(std::vector<Digit_t>& vecDigits, std::vector<T>&... vecSubDigits)
    -> std::enable_if_t<sizeof...(T) == sNSubDigits>
  {
    if constexpr (sNSubDigits > 0) {
      getSubDigit<sizeof...(T), sizeof...(T)>(std::tie(vecSubDigits...));
    }
    vecDigits.push_back(std::move(mDigit));
  }

  template <typename... T>
  auto getSingleSubDigits(std::vector<T>&... vecSingleSubDigits)
    -> std::enable_if_t<sizeof...(T) == sNSingleSubDigits>
  {
    if constexpr (sNSingleSubDigits > 0) {
      getSingleSubDigit<sizeof...(T), sizeof...(T)>(std::tie(vecSingleSubDigits...));
    }
  }

  template <std::size_t N, std::size_t N_TOTAL, typename... T>
  auto getSubDigit(std::tuple<T...> tupleVecSubDigits) -> std::enable_if_t<(N_TOTAL > 1)>
  {
    mDigit.ref[N - 1].set(std::get<N - 1>(tupleVecSubDigits).size(), std::get<N - 1>(mSubDigit).size());
    std::move(std::get<N - 1>(mSubDigit).begin(), std::get<N - 1>(mSubDigit).end(), std::back_inserter(std::get<N - 1>(tupleVecSubDigits)));
    if constexpr (N > 1) {
      getSubDigit<N - 1, N_TOTAL>(tupleVecSubDigits);
    }
  }
  template <std::size_t N, std::size_t N_TOTAL, typename... T>
  auto getSubDigit(std::tuple<T...> tupleVecSubDigits) -> std::enable_if_t<(N_TOTAL == 1)>
  {
    mDigit.ref.set(std::get<0>(tupleVecSubDigits).size(), mSubDigit.size());
    std::move(mSubDigit.begin(), mSubDigit.end(), std::back_inserter(std::get<0>(tupleVecSubDigits)));
  }

  template <std::size_t N, std::size_t N_TOTAL, typename... T>
  auto getSingleSubDigit(std::tuple<T...> tupleVecSingleSubDigits) -> std::enable_if_t<(N_TOTAL > 1)>
  {
    std::get<N - 1>(tupleVecSingleSubDigits).push_back(std::move(std::get<N - 1>(mSingleSubDigit)));
    if constexpr (N > 1) {
      getSingleSubDigit<N - 1, N_TOTAL>(tupleVecSingleSubDigits);
    }
  }
  template <std::size_t N, std::size_t N_TOTAL, typename... T>
  auto getSingleSubDigit(std::tuple<T...> tupleVecSingleSubDigits) -> std::enable_if_t<(N_TOTAL == 1)>
  {
    std::get<0>(tupleVecSingleSubDigits).push_back(std::move(mSingleSubDigit));
  }
  // 1-Dim SubDigit
  template <typename DigitBlockType, typename DigitT, typename SubDigitT>
  static auto makeDigitBlock(const std::vector<DigitT>& vecDigits, const std::vector<SubDigitT>& vecSubDigits) -> std::enable_if_t<DigitBlockHelper::GetDigitRefsN<DigitT>::value == 1 && DigitBlockHelper::IsSpecOfType<DigitBlockBase, typename DigitBlockType::DigitBlockBase_t>::value, std::vector<DigitBlockType>>
  {
    std::vector<DigitBlockType> vecResult;
    vecResult.reserve(vecDigits.size());
    for (const auto& digit : vecDigits) {
      auto itBegin = vecSubDigits.begin();
      std::advance(itBegin, digit.ref.getFirstEntry());
      auto itLast = itBegin;
      std::advance(itLast, digit.ref.getEntries());
      vecResult.push_back({digit});
      vecResult.back().mSubDigit.reserve(digit.ref.getEntries());
      std::copy(itBegin, itLast, std::back_inserter(vecResult.back().mSubDigit));
    }
    return vecResult;
  }

  // Multi-Dim SubDigits
  template <typename DigitBlockType, typename DigitT, typename... SubDigitT>
  static auto makeDigitBlock(const std::vector<DigitT>& vecDigits, const std::vector<SubDigitT>&... vecSubDigits) -> std::enable_if_t<(DigitBlockHelper::GetDigitRefsN<DigitT>::value > 1) && (DigitBlockHelper::IsSpecOfType<DigitBlockBase, typename DigitBlockType::DigitBlockBase_t>::value), std::vector<DigitBlockType>>
  {
    std::vector<DigitBlockType> vecResult;
    vecResult.reserve(vecDigits.size());
    for (const auto& digit : vecDigits) {
      vecResult.push_back({digit});
      auto& refTuple = vecResult.back().mSubDigit;
      fillSubDigitTuple<sizeof...(SubDigitT)>(digit, std::tie(vecSubDigits...), refTuple);
    }
    return vecResult;
  }

  template <std::size_t N, typename DigitT, typename... T>
  static void fillSubDigitTuple(const DigitT& digit, const std::tuple<T...>& tupleSrc, std::tuple<T...>& tupleDest)
  {
    const auto& vecSrc = std::get<N>(tupleSrc);
    auto& vecDest = std::get<N>(tupleSrc);
    auto itBegin = vecSrc.begin();
    std::advance(itBegin, digit.ref[N - 1].getFirstEntry());
    auto itLast = itBegin;
    std::advance(itLast, digit.ref[N - 1].getEntries());
    vecDest.reserve(digit.ref[N - 1].getEntries());
    std::copy(itBegin, itLast, std::back_inserter(vecDest));
    if constexpr (N > 1) {
      fillSubDigitTuple<N - 1>(digit, tupleSrc, tupleDest);
    }
  }
  void print() const
  {
    mDigit.printLog();
    if constexpr (DigitBlockHelper::IsSpecOfType<std::tuple, decltype(mSubDigit)>::value) {
      if constexpr ((std::tuple_size<decltype(mSubDigit)>::value) > 1) {
        LOG(INFO) << "______________SUB DIGITS____________";
        std::apply([](const auto&... vecSubDigit) {
          ((std::for_each(vecSubDigit.begin(), vecSubDigit.end(), [](const auto& subDigit) {
             subDigit.printLog();
           })),
           ...);
        },
                   mSubDigit);
      }
    } else {
      LOG(INFO) << "______________SUB DIGITS____________";
      std::for_each(mSubDigit.begin(), mSubDigit.end(), [](const auto& subDigit) {
        subDigit.printLog();
      });
    }
    if constexpr (DigitBlockHelper::IsSpecOfType<std::tuple, decltype(mSingleSubDigit)>::value) {
      if constexpr ((std::tuple_size<decltype(mSingleSubDigit)>::value) > 1) {
        LOG(INFO) << "______________SINGLE SUB DIGITS____________";
        std::apply([](const auto&... singleSubDigit) {
          ((singleSubDigit.printLog()), ...);
        },
                   mSingleSubDigit);
      }
    } else {
      LOG(INFO) << "______________SINGLE SUB DIGITS____________";
      mSingleSubDigit.printLog();
    }
    LOG(INFO) << std::dec;
    LOG(INFO) << "______________________________________";
  }
};

} // namespace fit
} // namespace o2
#endif
