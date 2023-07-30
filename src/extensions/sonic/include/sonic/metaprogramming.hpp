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

}  // namespace metaprogramming

}  // namespace sonic