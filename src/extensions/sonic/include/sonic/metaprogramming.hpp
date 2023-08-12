#include <tuple>

namespace sonic {

namespace metaprogramming {

template <template <auto...> typename T, auto... a_values, auto... b_values>
constexpr auto merge(const T<a_values...>&, const T<b_values...>&) {
  return T<a_values..., b_values...>{};
}

template <typename first_t, typename... rest_t>
std::tuple<rest_t...> rest(const std::tuple<first_t, rest_t...>& t) {
  return std::apply([](auto, auto... rest) { return std::make_tuple(rest...); }, t);
}

template <typename>
inline constexpr bool always_false_v = false;

template <typename T>
inline constexpr void raise_static_error() {
  static_assert(always_false_v<T>, "Unsupported case!");
}

}  // namespace metaprogramming

}  // namespace sonic