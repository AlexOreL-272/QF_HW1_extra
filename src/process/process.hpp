# pragma once

#include <torch/torch.h>

class BaseProcess {
 public:
  virtual ~BaseProcess() = default;

  virtual torch::Tensor SimulateSpotPrices(
    const uint32_t& steps_amt,
    const uint32_t& paths_amt,
    const torch::Tensor& spot_price,
    const torch::Tensor& strike,
    const torch::Tensor& time_to_maturity,
    const torch::Tensor& volatility,
    const torch::Tensor& dividend_rate,
    const torch::Tensor& risk_free_rate
  ) const = 0;
};

class BrownianMotionProcess : public BaseProcess {
 public:
  ~BrownianMotionProcess() override = default;

  torch::Tensor SimulateSpotPrices(
    const uint32_t& steps_amt,
    const uint32_t& paths_amt,
    const torch::Tensor& spot_price,
    const torch::Tensor& strike,
    const torch::Tensor& time_to_maturity,
    const torch::Tensor& volatility,
    const torch::Tensor& dividend_rate,
    const torch::Tensor& risk_free_rate
  ) const override;
};