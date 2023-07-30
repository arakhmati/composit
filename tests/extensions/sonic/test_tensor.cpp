#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.hpp"

#include "sonic/tensor.hpp"
#include "sonic/vector.hpp"

TEST_CASE("test tensor exp") {
    using namespace sonic::tensor;

    using data_type_t = float;
    using shape_t = Shape<1, 4, 8>;

    constexpr auto input = random<data_type_t, shape_t>();
    constexpr auto output = exp(input);

    const auto golden_output_data = std::array<data_type_t, shape_t::volume>{1.20405,1.99079,2.04601,2.00271,1.28034,0.793552,0.667023,0.412065,0.634646,0.956313,1.86701,0.960746,0.807001,1.95846,0.722377,1.34493,0.768345,2.49505,0.487094,2.0963,0.948585,1.82544,1.04181,1.43012,1.55467,1.17826,1.07761,1.67738,0.454669,0.94857,0.534013,1.60614};
    constexpr auto golden_output = as_tensor<data_type_t, shape_t>(std::move(golden_output_data));

    CHECK(allclose(output, golden_output));
}