#pragma once

#include <torch/torch.h>

#include "../process/process.hpp"

class Underlying {
 public:
  Underlying(
    const torch::Tensor& spot_price,       // S_0
    const torch::Tensor& strike,           // K
    const torch::Tensor& time_to_maturity, // T
    const torch::Tensor& volatility,       // sigma
    const torch::Tensor& dividend_rate,    // q
    const torch::Tensor& risk_free_rate    // r
  );

  ~Underlying() = default;

  void Simulate(
    const uint32_t& steps_amt,
    const BaseProcess& generator
  );

  torch::Tensor GetSpotPrices() const;

  torch::Tensor SpotPriceAt(
    const torch::Tensor& time
  ) const;

  torch::Tensor GetInitialPrice() const;

  torch::Tensor GetStrike() const;

  torch::Tensor GetTimeToMaturity() const;

  torch::Tensor GetVolatility() const;

  torch::Tensor GetDividendRate() const;

  torch::Tensor GetRiskFreeRate() const;

 private:
  torch::Tensor spot_price_;
  torch::Tensor strike_;
  torch::Tensor time_to_maturity_;
  torch::Tensor volatility_;
  torch::Tensor dividend_rate_;
  torch::Tensor risk_free_rate_;

  torch::Tensor spot_prices_;
};