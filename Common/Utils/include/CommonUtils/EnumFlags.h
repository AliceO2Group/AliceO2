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
#ifndef O2_FRAMEWORK_FLAGS_H_
#define O2_FRAMEWORK_FLAGS_H_

#include <algorithm>
#include <array>
#include <concepts>
#include <exception>
#include <ostream>
#include <source_location>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <string>
#include <sstream>
#include <limits>
#include <bit>
#include <bitset>
#include <initializer_list>
#include <cstdint>
#include <cstddef>
#include <cctype>
#include <utility>
#include <optional>
#include <iostream>
#include <iomanip>

#ifndef GPUCA_GPUCODE
#include "CommonUtils/StringUtils.h"
#endif

namespace o2::utils
{

namespace details::enum_flags
{

// Require that an enum with an underlying unsigned type.
template <typename E>
concept EnumFlagHelper = requires {
  requires std::is_enum_v<E>;
  requires std::is_unsigned_v<std::underlying_type_t<E>>;
  requires std::same_as<E, std::decay_t<E>>;
};

// Static constexpr only helper struct to implement modicum of enum reflection
// functions and also check via concepts expected properties of the enum.
// This is very much inspired by much more extensive libraries like magic_enum.
// Inspiration by its c++20 version (https://github.com/fix8mt/conjure_enum).
// NOTE: Cannot detect if bit values past the underlying type are defined.
#ifndef GPUCA_GPUCODE
template <EnumFlagHelper E>
struct FlagsHelper final {
  using U = std::underlying_type_t<E>;
  using UMax = uint64_t; // max represetable type
  static_assert(std::numeric_limits<U>::digits <= std::numeric_limits<UMax>::digits, "Underlying type has more digits than max supported digits");

  static constexpr bool isScoped() noexcept
  {
    return std::is_enum_v<E> && !std::is_convertible_v<E, std::underlying_type_t<E>>;
  }

  // Return line at given position.
  template <E e>
  static consteval const char* tpeek() noexcept
  {
    return std::source_location::current().function_name();
  }
  // string_view value of function above
  template <E e>
  static constexpr std::string_view tpeek_v{tpeek<e>()};

  // Compiler Specifics
  static constexpr auto CSpecifics{std::to_array<
    std::tuple<std::string_view, char, std::string_view, char>>({
#if defined __clang__
    {"e = ", ']', "(anonymous namespace)", '('},
    {"T = ", ']', "(anonymous namespace)", '('},
#else // assuming __GNUC__
    {"e = ", ';', "<unnamed>", '<'},
    {"T = ", ']', "{anonymous}", '{'},
#endif
  })};
  enum class SVal : uint8_t { Start,
                              End,
                              AnonStr,
                              AnonStart };
  enum class SType : uint8_t { Enum_t,
                               Type_t,
                               eT0,
                               eT1,
                               eT2,
                               eT3 };
  // Extract a compiler specification.
  template <SVal v, SType t>
  static constexpr auto getSpec() noexcept
  {
    return std::get<static_cast<size_t>(v)>(CSpecifics[static_cast<size_t>(t)]);
  }

  // Range that is scanned by the compiler
  static constexpr size_t MinScan{0};
  static constexpr size_t MarginScan{1};                                // Scan one past to check for overpopulation
  static constexpr size_t MaxUnderScan{std::numeric_limits<U>::digits}; // Maximum digits the underlying type has
  static constexpr size_t MaxScan{MaxUnderScan + MarginScan};

  // Checks if a given 'location' contains an enum.
  template <E e>
  static constexpr bool isValid() noexcept
  {
    constexpr auto tp{tpeek_v<e>.rfind(getSpec<SVal::Start, SType::Enum_t>())};
    if constexpr (tp == std::string_view::npos) {
      return false;
    }
#if defined __clang__
    else if constexpr (tpeek_v<e>[tp + getSpec<SVal::Start, SType::Enum_t>().size()] == '(') {
      if constexpr (tpeek_v<e>[tp + getSpec<SVal::Start, SType::Enum_t>().size() + 1] == '(') {
        return false;
      }
      if constexpr (tpeek_v<e>.find(getSpec<SVal::AnonStr, SType::Enum_t>(), tp + getSpec<SVal::Start, SType::Enum_t>().size()) != std::string_view::npos) {
        return true;
      }
    } else if constexpr (tpeek_v<e>.find_first_of(getSpec<SVal::End, SType::Enum_t>(), tp + getSpec<SVal::Start, SType::Enum_t>().size()) != std::string_view::npos) {
      // check if this is an anonymous enum
      return true;
    }
#elif __GNUC__
    else if constexpr (tpeek_v<e>[tp + getSpec<SVal::Start, SType::Enum_t>().size()] != '(' && tpeek_v<e>.find_first_of(getSpec<SVal::End, SType::Enum_t>(), tp + getSpec<SVal::Start, SType::Enum_t>().size()) != std::string_view::npos) {
      return true;
    }
#else
#error Unsupported compiler
#endif
    return false;
  }

  // Extract which values are present in the enum by checking all values in
  // the min-max-range above.
  template <size_t... I>
  static constexpr auto getValues(std::index_sequence<I...> /*unused*/) noexcept
  {
    constexpr std::array<bool, sizeof...(I)> valid{isValid<static_cast<E>(MinScan + I)>()...};
    constexpr auto count{std::count_if(valid.cbegin(), valid.cend(), [](bool v) noexcept { return v; })};
    static_assert(count > 0, "EnumFlag requires at least one enum value. Check that your enum has consecutive values starting from 0.");
    static_assert(count <= MaxUnderScan, "Too many enum values for underlying type. Consider using a larger underlying type or fewer enum values.");
    std::array<E, count> values{};
    for (size_t idx{}, n{}; n < count; ++idx) {
      if (valid[idx]) {
        values[n++] = static_cast<E>(MinScan + idx);
      }
    }
    return values;
  }
  static constexpr auto Values{getValues(std::make_index_sequence<MaxScan - MinScan - MarginScan>())}; // Enum Values
  static constexpr auto count() noexcept { return Values.size(); }                                     // Number of enum members
  static constexpr auto Min_v{Values.front()};                                                         // Enum first entry
  static constexpr auto Max_v{Values.back()};                                                          // Enum last entry
  static constexpr auto Min_u_v{static_cast<size_t>(Min_v)};                                           // Enum first entry as size_t
  static constexpr auto Max_u_v{static_cast<size_t>(Max_v)};                                           // Enum last entry as size_t
  static_assert(Max_u_v < std::numeric_limits<U>::digits, "Max Bit is beyond allow range deferred from underlying type");
  static constexpr bool isContinuous() noexcept { return (Max_u_v - Min_u_v + 1) == count(); } // Is the enum continuous
  static constexpr UMax makeMaxRep(size_t min, size_t max)
  {
    const size_t width = max - min + 1;
    if (width >= std::numeric_limits<UMax>::digits) {
      return std::numeric_limits<UMax>::max();
    }
    return ((UMax(1) << width) - 1) << min;
  }
  static constexpr auto MaxRep{makeMaxRep(Min_u_v, Max_u_v)}; // largest representable value

  template <E e>
  static constexpr std::string_view getName()
  {
    constexpr auto tp{tpeek_v<e>.rfind(getSpec<SVal::Start, SType::Enum_t>())};
    if constexpr (tp == std::string_view::npos) {
      return {};
    }
    if constexpr (tpeek_v<e>[tp + getSpec<SVal::Start, SType::Enum_t>().size()] == getSpec<SVal::AnonStart, SType::Enum_t>()) {
#if defined __clang__
      if constexpr (tpeek_v<e>[tp + getSpec<SVal::Start, SType::Enum_t>().size() + 1] == getSpec<SVal::AnonStart, SType::Enum_t>()) {
        return {};
      }
#endif
      if (constexpr auto lstr{tpeek_v<e>.substr(tp + getSpec<SVal::Start, SType::Enum_t>().size())}; lstr.find(getSpec<SVal::AnonStr, SType::Enum_t>()) != std::string_view::npos) { // is anon
        if constexpr (constexpr auto lc{lstr.find_first_of(getSpec<SVal::End, SType::Enum_t>())}; lc != std::string_view::npos) {
          return lstr.substr(getSpec<SVal::AnonStr, SType::Enum_t>().size() + 2, lc - (getSpec<SVal::AnonStr, SType::Enum_t>().size() + 2));
        }
      }
    }
    constexpr std::string_view result{tpeek_v<e>.substr(tp + getSpec<SVal::Start, SType::Enum_t>().size())};
    if constexpr (constexpr auto lc{result.find_first_of(getSpec<SVal::End, SType::Enum_t>())}; lc != std::string_view::npos) {
      return result.substr(0, lc);
    } else {
      return {};
    }
  }

  static constexpr std::string_view removeScope(std::string_view s)
  {
    if (const auto lc{s.find_last_of(':')}; lc != std::string_view::npos) {
      return s.substr(lc + 1);
    }
    return s;
  }

  static constexpr std::string_view findScope(std::string_view s)
  {
    const auto pos1 = s.rfind("::");
    if (pos1 == std::string_view::npos) {
      return s;
    }
    const auto pos2 = s.rfind("::", pos1 - 1);
    if (pos2 == std::string_view::npos) {
      return s.substr(0, pos1);
    }
    return s.substr(pos2 + 2, pos1 - pos2 - 2);
  }

  template <E e>
  static constexpr auto getNameValue{getName<e>()};

  template <bool with_scope, size_t... I>
  static constexpr auto getNames(std::index_sequence<I...> /*unused*/)
  {
    if constexpr (with_scope) {
      return std::array<std::string_view, sizeof...(I)>{getNameValue<Values[I]>...};
    } else {
      return std::array<std::string_view, sizeof...(I)>{removeScope(getNameValue<Values[I]>)...};
    }
  }

  static constexpr auto Names{getNames<false>(std::make_index_sequence<count()>())};      // Enum names without scope
  static constexpr auto NamesScoped{getNames<true>(std::make_index_sequence<count()>())}; // Enum names with scope
  static constexpr auto Scope{findScope(NamesScoped.front())};                            // Enum scope

  static constexpr auto getLongestName() noexcept
  {
    size_t max{0};
    for (size_t i{0}; i < count(); ++i) {
      max = std::max(max, Names[i].size());
    }
    return max;
  }

  static constexpr auto NamesLongest{getLongestName()}; // Size of longest name

  template <E e>
  static constexpr std::string_view toString() noexcept
  {
    return getNameValue<e>();
  }

  static constexpr std::optional<E> fromString(std::string_view str) noexcept
  {
    for (size_t i{0}; i < count(); ++i) {
      if (isIEqual(Names[i], str) || isIEqual(NamesScoped[i], str)) {
        return Values[i];
      }
    }
    return std::nullopt;
  }

  // Convert char to lower.
  static constexpr unsigned char toLower(const unsigned char c) noexcept
  {
    return (c >= 'A' && c <= 'Z') ? (c - 'A' + 'a') : c;
  }

  // Are these chars equal (case-insensitive).
  static constexpr bool isIEqual(const unsigned char a, const unsigned char b) noexcept
  {
    return toLower(a) == toLower(b);
  }

  // Case-insensitive comparison for string_view.
  static constexpr bool isIEqual(std::string_view s1, std::string_view s2) noexcept
  {
    if (s1.size() != s2.size()) {
      return false;
    }
    for (size_t i{0}; i < s1.size(); ++i) {
      if (!isIEqual(s1[i], s2[i])) {
        return false;
      }
    }
    return true;
  }

  static constexpr std::string_view None{"none"};
  static constexpr bool hasNone() noexcept
  {
    // check that enum does not contain member named 'none'
    for (size_t i{0}; i < count(); ++i) {
      if (isIEqual(Names[i], None)) {
        return true;
      }
    }
    return false;
  }

  static constexpr std::string_view All{"all"};
  static constexpr bool hasAll() noexcept
  {
    // check that enum does not contain member named 'all'
    for (size_t i{0}; i < count(); ++i) {
      if (isIEqual(Names[i], All)) {
        return true;
      }
    }
    return false;
  }
};
#endif

} // namespace details::enum_flags

// Require an enum to fullfil what one would except from a bitset.
#ifndef GPUCA_GPUCODE
template <typename E>
concept EnumFlag = requires {
  // range checks
  requires details::enum_flags::FlagsHelper<E>::Min_u_v == 0;                                           // the first bit should be at position 0
  requires details::enum_flags::FlagsHelper<E>::Max_u_v < details::enum_flags::FlagsHelper<E>::count(); //  the maximum is less than the total
  requires details::enum_flags::FlagsHelper<E>::isContinuous();                                         // do not allow missing bits

  // type checks
  requires !details::enum_flags::FlagsHelper<E>::hasNone(); // added automatically
  requires !details::enum_flags::FlagsHelper<E>::hasAll();  // added automatically
};
#else
template <typename E>
concept EnumFlag = details::enum_flags::EnumFlagHelper<E>;
#endif

/**
 * \brief Class to aggregate and manage enum-based on-off flags.
 *
 * This class manages flags as bits in the underlying type of an enum (upto 64 bits), allowing
 * manipulation via enum member names. It supports operations akin to std::bitset
 * but is fully constexpr and is ideal for aggregating multiple on-off booleans,
 * e.g., enabling/disabling algorithm features.
 *
 * Example:
 * enum class AlgoOptions {
 *     Feature1,
 *     Feature2,
 *     Feature3,
 * };
 * ...
 * EnumFlags<AlgoOptions> opts;
 * opts.set("Feature1 | Feature3"); // Set Feature1 and Feature3.
 * if (opts[AlgoOptions::Feature1]) { // Do some work. } // Check if Feature1 is set.
 *
 * Additional examples of how to use this class are in testEnumFlags.cxx.
 */
template <EnumFlag E>
class EnumFlags
{
  static constexpr int DefaultBase{2};
#ifndef GPUCA_GPUCODE
  using H = details::enum_flags::FlagsHelper<E>;
#endif
  using U = std::underlying_type_t<E>;
  U mBits{0};

  // Converts enum to its underlying type.
  constexpr auto to_underlying(E e) const noexcept
  {
    return static_cast<U>(e);
  }

  // Returns the bit representation of a flag.
  constexpr auto to_bit(E e) const noexcept
  {
    return U(1) << to_underlying(e);
  }

 public:
  // Default constructor.
  constexpr explicit EnumFlags() = default;
  // Constructor to initialize with a single flag.
  constexpr explicit EnumFlags(E e) : mBits(to_bit(e)) {}
  // Copy constructor.
  constexpr EnumFlags(const EnumFlags&) = default;
  // Move constructor.
  constexpr EnumFlags(EnumFlags&&) = default;
  // Constructor to initialize with the underlying type.
  constexpr explicit EnumFlags(U u) : mBits(u) {}
  // Initialize with a list of flags.
  constexpr EnumFlags(std::initializer_list<E> flags) noexcept
  {
    for (const E f : flags) {
      mBits |= to_bit(f);
    }
  }
#ifndef GPUCA_GPUCODE
  // Init from a string.
  //
  explicit EnumFlags(const std::string& str, int base = DefaultBase)
  {
    set(str, base);
  }
#endif

  static constexpr U None{0}; // Represents no flags set.
#ifndef GPUCA_GPUCODE
  static constexpr U All{H::MaxRep}; // Represents all flags set.

  // Return list of all enum values
  static constexpr auto getValues() noexcept
  {
    return H::Values;
  }

  // Return list of all enum Names
  static constexpr auto getNames() noexcept
  {
    return H::Names;
  }

  // Sets flags from a string representation.
  // This can be either from a number representation (binary or digits) or
  // a concatenation of the enums members name e.g., 'Enum1|Enum2|...'
  void set(const std::string& s, int base = DefaultBase)
  {
    if (s.empty()) { // no-op
      return;
    }
    // on throw restore previous state and rethrow
    const U prev = mBits;
    reset();
    try {
      setImpl(s, base);
    } catch (const std::exception& e) {
      mBits = prev;
      throw;
    }
  }
#endif
  // Returns the raw bitset value.
  [[nodiscard]] constexpr auto value() const noexcept
  {
    return mBits;
  }

  // Resets all flags.
  constexpr void reset() noexcept
  {
    mBits = U(0);
  }

  // Resets a specific flag.
  template <std::same_as<E> T>
  constexpr void reset(T t)
  {
    mBits &= ~to_bit(t);
  }

  // Tests if a specific flag is set.
  template <std::same_as<E> T>
  [[nodiscard]] constexpr bool test(T t) const noexcept
  {
    return (mBits & to_bit(t)) != None;
  }

  // Tests if all specified flags are set.
  template <std::same_as<E>... Ts>
  [[nodiscard]] constexpr bool test(Ts... flags) const noexcept
  {
    return ((test(flags) && ...));
  }

  // Sets a specific flag.
  template <std::same_as<E> T>
  constexpr void set(T t) noexcept
  {
    mBits |= to_bit(t);
  }

  // Sets multiple specific flags.
  template <std::same_as<E>... Ts>
  constexpr void set(Ts... flags) noexcept
  {
    (set(flags), ...);
  }

  // Toggles a specific flag.
  template <std::same_as<E> T>
  constexpr void toggle(T t) noexcept
  {
    mBits ^= to_bit(t);
  }

  // Checks if any flag is set.
  [[nodiscard]] constexpr bool any() const noexcept
  {
    return mBits != None;
  }

  // Checks if all flags are set.
#ifndef GPUCA_GPUCODE
  [[nodiscard]] constexpr bool all() const noexcept
  {
    return mBits == All;
  }

  // Returns the bitset as a binary string.
  [[nodiscard]] std::string string() const
  {
    std::ostringstream oss;
    oss << std::bitset<H::count()>(mBits);
    return oss.str();
  }

  // Returns the bitset as a pretty multiline binary string.
  [[nodiscard]] std::string pstring(bool withNewline = false) const
  {
    std::ostringstream oss;
    if (withNewline) {
      oss << '\n';
    }
    oss << "0b";
    const std::bitset<H::count()> bits(mBits);
    oss << bits;
    if constexpr (H::isScoped()) {
      oss << " " << H::Scope;
    }
    oss << '\n';
    for (size_t i = 0; i < H::count(); ++i) {
      oss << "  ";
      for (size_t j = 0; j < H::count() - i - 1; ++j) {
        oss << "┃";
      }
      oss << "┗";
      for (size_t a{2 + i}; --a != 0U;) {
        oss << "━";
      }
      oss << " " << std::setw(H::NamesLongest) << std::left
          << H::Names[i] << " " << (bits[i] ? "[Active]" : "[Inactive]");
      if (i != H::count() - 1) {
        oss << "\n";
      }
    }
    return oss.str();
  }
#endif

  // Checks if any flag is set (Boolean context).
  [[nodiscard]] constexpr explicit operator bool() const noexcept
  {
    return any();
  }

  // Check if given flag is set.
  template <std::same_as<E> T>
  [[nodiscard]] constexpr bool operator[](const T t) const noexcept
  {
    return test(t);
  }

  // Checks if two flag sets are equal.
  [[nodiscard]] constexpr bool operator==(const EnumFlags& o) const noexcept
  {
    return mBits == o.mBits;
  }

  // Checks if two flag sets are not equal.
  [[nodiscard]] constexpr bool operator!=(const EnumFlags& o) const noexcept
  {
    return mBits != o.mBits;
  }

  // Copy assignment operator
  constexpr EnumFlags& operator=(const EnumFlags& o) = default;

  // Move assignment operator
  constexpr EnumFlags& operator=(EnumFlags&& o) = default;

  // Performs a bitwise OR with a flag.
  template <std::same_as<E> T>
  constexpr EnumFlags& operator|=(T t) noexcept
  {
    mBits |= to_bit(t);
    return *this;
  }

  // Performs a bitwise AND with a flag.
  template <std::same_as<E> T>
  constexpr EnumFlags& operator&=(T t) noexcept
  {
    mBits &= to_bit(t);
    return *this;
  }

  // Returns a flag set with a bitwise AND.
  template <std::same_as<E> T>
  constexpr EnumFlags operator&(T t) const noexcept
  {
    return EnumFlags(mBits & to_bit(t));
  }

  // Returns a flag set with all bits inverted.
  constexpr EnumFlags operator~() const noexcept
  {
    return EnumFlags(~mBits);
  }

  // Performs a bitwise OR with another flag set.
  constexpr EnumFlags operator|(const EnumFlags& o) const noexcept
  {
    return EnumFlags(mBits | o.mBits);
  }

  // Performs a bitwise OR assignment.
  constexpr EnumFlags& operator|=(const EnumFlags& o) noexcept
  {
    mBits |= o.mBits;
    return *this;
  }

  // Performs a bitwise XOR with another flag set.
  constexpr EnumFlags operator^(const EnumFlags& o) const noexcept
  {
    return EnumFlags(mBits ^ o.mBits);
  }

  // Performs a bitwise and with another flag set.
  constexpr EnumFlags operator&(const EnumFlags& o) const noexcept
  {
    return EnumFlags(mBits & o.mBits);
  }

  // Performs a bitwise XOR assignment.
  constexpr EnumFlags& operator^=(const EnumFlags& o) noexcept
  {
    mBits ^= o.mBits;
    return *this;
  }

  // Checks if all specified flags are set.
  template <typename... Ts>
  [[nodiscard]] constexpr bool all_of(Ts... flags) const noexcept
  {
    return test(flags...);
  }

  // Checks if none of the specified flags are set.
  template <typename... Ts>
  [[nodiscard]] constexpr bool none_of(Ts... flags) const noexcept
  {
    return (!(test(flags) || ...));
  }

  // Serializes the flag set to a string.
#ifndef GPUCA_GPUCODE
  [[nodiscard]] std::string serialize() const
  {
    return std::to_string(mBits);
  }

  // Deserializes a string into the flag set.
  void deserialize(const std::string& data)
  {
    typename H::UMax v = std::stoul(data);
    if (v > H::MaxRep) {
      throw std::out_of_range("Values exceeds enum range.");
    }
    mBits = static_cast<U>(v);
  }
#endif

  // Counts the number of set bits (active flags).
  [[nodiscard]] constexpr size_t count() const noexcept
  {
    return std::popcount(mBits);
  }

  // Returns the union of two flag sets.
  [[nodiscard]] constexpr EnumFlags union_with(const EnumFlags& o) const noexcept
  {
    return EnumFlags(mBits | o.mBits);
  }

  // Returns the intersection of two flag sets.
  [[nodiscard]] constexpr EnumFlags intersection_with(const EnumFlags& o) const noexcept
  {
    return EnumFlags(mBits & o.mBits);
  }

  // Checks if all flags in another Flags object are present in the current object.
  [[nodiscard]] constexpr bool contains(const EnumFlags& other) const noexcept
  {
    return (mBits & other.mBits) == other.mBits;
  }

 private:
  // Set implementation, bits was zeroed before.
#ifndef GPUCA_GPUCODE
  void setImpl(const std::string& s, int base = 2)
  {
    // Helper to check if character is valid for given base
    auto isValidForBase = [](unsigned char c, int base) -> bool {
      if (base == 2) {
        return c == '0' || c == '1';
      }
      if (base == 10) {
        return std::isdigit(c);
      }
      if (base == 16) {
        return std::isdigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
      }
      return false;
    };

    // hex
    if (base == 16) {
      std::string_view hex_str{s};
      // Strip optional 0x or 0X prefix
      if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        hex_str.remove_prefix(2);
      }
      if (hex_str.empty()) {
        throw std::invalid_argument("Empty hexadecimal string.");
      }
      if (!std::all_of(hex_str.begin(), hex_str.end(), [&](unsigned char c) { return isValidForBase(c, 16); })) {
        throw std::invalid_argument("Invalid hexadecimal string.");
      }
      typename H::UMax v = std::stoul(std::string(hex_str), nullptr, 16);
      if (v > H::MaxRep) {
        throw std::out_of_range("Value exceeds enum range.");
      }
      mBits = static_cast<U>(v);
      return;
    }

    // decimal and binary
    if (std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); })) {
      if (base == 2) {
        // Binary: check only 0 and 1
        if (!std::all_of(s.begin(), s.end(), [&](unsigned char c) { return isValidForBase(c, 2); })) {
          throw std::invalid_argument("Invalid binary string.");
        }
      }
      typename H::UMax v = std::stoul(std::string(s), nullptr, base);
      if (v > H::MaxRep) {
        throw std::out_of_range("Value exceeds enum range.");
      }
      mBits = static_cast<U>(v);
    }
    // enum name strings
    else if (std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isalnum(c) != 0 || c == '|' || c == ' ' || c == ':' || c == ',' || c == ';'; })) {
      std::string cs{s};
      std::transform(cs.begin(), cs.end(), cs.begin(), [](unsigned char c) { return std::tolower(c); });

      if (cs == H::All) {
        mBits = All;
      } else if (cs == H::None) {
        mBits = None;
      } else {
        // Detect delimiter and ensure only one type is used
        char token = ' ';
        size_t pipePos = s.find('|');
        size_t commaPos = s.find(',');
        size_t semiPos = s.find(';');

        // Count how many different delimiters exist
        int delimiterCount = (pipePos != std::string_view::npos ? 1 : 0) +
                             (commaPos != std::string_view::npos ? 1 : 0) +
                             (semiPos != std::string_view::npos ? 1 : 0);

        if (delimiterCount > 1) {
          throw std::invalid_argument("Mixed delimiters not allowed!");
        }

        if (pipePos != std::string_view::npos) {
          token = '|';
        } else if (commaPos != std::string_view::npos) {
          token = ',';
        } else if (semiPos != std::string_view::npos) {
          token = ';';
        }

        for (const auto& tok : Str::tokenize(std::string(s), token)) {
          if (auto e = H::fromString(tok)) {
            mBits |= to_bit(*e);
          } else {
            throw std::invalid_argument(tok + " is not a valid enum value!");
          }
        }
      }
    } else {
      throw std::invalid_argument("Cannot parse string!");
    }
  }
#endif
};

#ifndef GPUCA_GPUCODE
template <EnumFlag E>
std::ostream& operator<<(std::ostream& os, const EnumFlags<E>& f)
{
  os << f.pstring(true);
  return os;
}
#endif

} // namespace o2::utils

#endif
