#pragma once

namespace sonic {

namespace shape {

template <auto... Dims>
struct Shape {
  static constexpr auto volume = (1 * ... * Dims);
};

}  // namespace shape

}  // namespace sonic