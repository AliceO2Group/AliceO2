// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <type_traits>
#include <utility>

namespace o2::framework
{
namespace detail
{
// Original from https://github.com/ricab/scope_guard
// which is licensed to public domain
// Type trait determining whether a type is callable with no arguments
template <typename T, typename = void>
struct is_noarg_callable_t
  : public std::false_type {
}; // in general, false

template <typename T>
struct is_noarg_callable_t<T, decltype(std::declval<T&&>()())>
  : public std::true_type {
}; // only true when call expression valid

// Type trait determining whether a no-argument callable returns void
template <typename T>
struct returns_void_t
  : public std::is_same<void, decltype(std::declval<T&&>()())> {
};

/* Type trait determining whether a no-arg callable is nothrow invocable if
  required. This is where SG_REQUIRE_NOEXCEPT logic is encapsulated. */
template <typename T>
struct is_nothrow_invocable_if_required_t
  : public std::is_nothrow_invocable<T> /* Note: _r variants not enough to
                                        confirm void return: any return can be
                                        discarded so all returns are
                                        compatible with void */
{
};

template <typename A, typename B, typename... C>
struct and_t : public and_t<A, and_t<B, C...>> {
};

template <typename A, typename B>
struct and_t<A, B> : public std::conditional<A::value, B, A>::type {
};

template <typename T>
struct is_proper_sg_callback_t
  : public and_t<is_noarg_callable_t<T>,
                 returns_void_t<T>,
                 is_nothrow_invocable_if_required_t<T>,
                 std::is_nothrow_destructible<T>> {
};

template <typename Callback,
          typename = typename std::enable_if<
            is_proper_sg_callback_t<Callback>::value>::type>
class scope_guard;

template <typename Callback>
detail::scope_guard<Callback> make_scope_guard(Callback&& callback) noexcept(std::is_nothrow_constructible<Callback, Callback&&>::value);

template <typename Callback>
class scope_guard<Callback> final
{
 public:
  typedef Callback callback_type;

  scope_guard(scope_guard&& other) noexcept(std::is_nothrow_constructible<Callback, Callback&&>::value);

  ~scope_guard() noexcept; // highlight noexcept dtor

  void dismiss() noexcept;

 public:
  scope_guard() = delete;
  scope_guard(const scope_guard&) = delete;
  scope_guard& operator=(const scope_guard&) = delete;
  scope_guard& operator=(scope_guard&&) = delete;

 private:
  explicit scope_guard(Callback&& callback) noexcept(std::is_nothrow_constructible<Callback, Callback&&>::value); /*
                                                    meant for friends only */

  friend scope_guard<Callback> make_scope_guard<Callback>(Callback&&) noexcept(std::is_nothrow_constructible<Callback, Callback&&>::value); /*
    only make_scope_guard can create scope_guards from scratch (i.e. non-move)
    */

 private:
  Callback mCallback;
  bool mActive;
};

} // namespace detail

using detail::make_scope_guard; // see comment on declaration above

template <typename Callback>
detail::scope_guard<Callback>::scope_guard(Callback&& callback) noexcept(std::is_nothrow_constructible<Callback, Callback&&>::value)
  : mCallback(std::forward<Callback>(callback)) /* use () instead of {} because
    of DR 1467 (https://is.gd/WHmWuo), which still impacts older compilers
    (e.g. GCC 4.x and clang <=3.6, see https://godbolt.org/g/TE9tPJ and
    https://is.gd/Tsmh8G) */
    ,
    mActive{true}
{
}

template <typename Callback>
detail::scope_guard<Callback>::~scope_guard() noexcept
{
  if (mActive) {
    mCallback();
  }
}

template <typename Callback>
detail::scope_guard<Callback>::scope_guard(scope_guard&& other) noexcept(std::is_nothrow_constructible<Callback, Callback&&>::value)
  : mCallback(std::forward<Callback>(other.mCallback)) // idem
    ,
    mActive{std::move(other.mActive)}
{
  other.mActive = false;
}

template <typename Callback>
inline void detail::scope_guard<Callback>::dismiss() noexcept
{
  mActive = false;
}

template <typename Callback>
inline auto detail::make_scope_guard(Callback&& callback) noexcept(std::is_nothrow_constructible<Callback, Callback&&>::value)
  -> detail::scope_guard<Callback>
{
  return detail::scope_guard<Callback>{std::forward<Callback>(callback)};
}

} // namespace o2::framework
