#include "process.hpp"

torch::Tensor BrownianMotionProcess::SimulateSpotPrices(
  const uint32_t& steps_amt,
  const uint32_t& paths_amt,
  const torch::Tensor& spot_price,
  const torch::Tensor& strike,
  const torch::Tensor& time_to_maturity,
  const torch::Tensor& volatility,
  const torch::Tensor& dividend_rate,
  const torch::Tensor& risk_free_rate
) const {
  const auto kDeltaTime = 1.0 / steps_amt;
  const auto kSqrtDt = sqrt(kDeltaTime);

  auto spot_prices = torch::zeros({steps_amt + 1, paths_amt}, torch::kFloat64);
  spot_prices[0] = spot_price;

  for (uint32_t i = 1; i <= steps_amt; i++) {
    const auto kStdNormal = torch::normal(0.0, 1.0, {paths_amt});
    // not to modify the original tensor and compute gradients
    const auto kPrev = spot_prices[i - 1].clone();
    spot_prices[i] = kPrev * 
      exp((risk_free_rate - volatility * volatility / 2) * kDeltaTime + 
      volatility * kSqrtDt * kStdNormal);
  }

  return spot_prices;
}