#pragma once

namespace sonic {

namespace shape {

template <std::size_t... Dimensions>
struct shape_t {
  static constexpr std::size_t volume = (1 * ... * Dimensions);
  static constexpr std::size_t rank = sizeof...(Dimensions);
};

}  // namespace shape

}  // namespace sonic