#pragma once

#include <torch/torch.h>

#include "../underlying/underlying.hpp"

enum class OptionType {
  kCall = 0,
  kPut = 1,
};

class BaseOption {
 public:
  BaseOption(
    const OptionType& type,
    const Underlying& underlying
  );

  virtual ~BaseOption() = default;

  torch::Tensor Payoff() const;

  torch::Tensor GetUnderlyingPrice() const;

  virtual torch::Tensor PriceAt(
    const torch::Tensor& time
  ) const = 0;

 protected:
  OptionType type_;
  Underlying underlying_;
};

class EuropeanOption : public BaseOption {
 public:
  EuropeanOption(
    const OptionType& type,
    const Underlying& underlying
  );

  ~EuropeanOption() = default;

  torch::Tensor PriceAt(
    const torch::Tensor& time
  ) const override;
};