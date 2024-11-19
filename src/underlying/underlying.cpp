#include "underlying.hpp"

#include <torch/torch.h>

Underlying::Underlying(
  const torch::Tensor& spot_price,
  const torch::Tensor& strike,
  const torch::Tensor& time_to_maturity,
  const torch::Tensor& volatility,
  const torch::Tensor& dividend_rate,
  const torch::Tensor& risk_free_rate
) : spot_price_(spot_price),
    strike_(strike),
    time_to_maturity_(time_to_maturity),
    volatility_(volatility),
    dividend_rate_(dividend_rate),
    risk_free_rate_(risk_free_rate) {}

void Underlying::Simulate(
  const uint32_t& steps_amt,
  const BaseProcess& generator
) {
  spot_prices_ = generator.SimulateSpotPrices(
    steps_amt,
    1,
    spot_price_,
    strike_,
    time_to_maturity_,
    volatility_,
    dividend_rate_,
    risk_free_rate_
  );
}

torch::Tensor Underlying::GetSpotPrices() const {
  return spot_prices_;
}

torch::Tensor Underlying::SpotPriceAt(
  const torch::Tensor& time
) const {
  if (time.item<double>() > time_to_maturity_.item<double>() || time.item<double>() < 0) {
    // TODO: throw exception
    return torch::tensor({-1.0});
  }

  const auto kStepsAmt = spot_prices_.size(0) - 1;
  const auto kDeltaTime = time_to_maturity_ / kStepsAmt;
  const auto kIdx = (time / kDeltaTime).item<int>();

  return spot_prices_[kIdx];
}

torch::Tensor Underlying::GetInitialPrice() const {
  return spot_price_;
}

torch::Tensor Underlying::GetStrike() const {
  return strike_;
}

torch::Tensor Underlying::GetTimeToMaturity() const {
  return time_to_maturity_;
}

torch::Tensor Underlying::GetVolatility() const {
  return volatility_;
}

torch::Tensor Underlying::GetRiskFreeRate() const {
  return risk_free_rate_;
}

torch::Tensor Underlying::GetDividendRate() const {
  return dividend_rate_;
}