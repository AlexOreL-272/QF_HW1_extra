#include "calculator.hpp"

#include <torch/torch.h>

Calculator::Calculator() = default;

double Calculator::CalculateDelta(
  const BaseOption& option
) {
  const auto kTime = torch::tensor({0.0}, torch::requires_grad());
  auto price = option.PriceAt(kTime);
  price.backward();

  const auto kInitialPrice = option.GetUnderlyingPrice();
  return kInitialPrice.grad().item<double>();
}