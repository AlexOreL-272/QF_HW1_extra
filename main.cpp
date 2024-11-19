#include <iostream>
#include <torch/torch.h>

#include "src/delta_calculator/calculator.hpp"
#include "src/option/option.hpp"
#include "src/process/process.hpp"

int main() { 
  torch::manual_seed(420);

  auto S0 = torch::tensor({80.0}, torch::requires_grad());
  auto K = torch::tensor({85.0}, torch::requires_grad());
  auto T = torch::tensor({1.0}, torch::requires_grad());
  auto sigma = torch::tensor({0.2}, torch::requires_grad());
  auto q = torch::tensor({0.00}, torch::requires_grad());
  auto r = torch::tensor({0.05}, torch::requires_grad());

  Underlying underlying(
    S0,
    K,
    T,
    sigma,
    q,
    r
  );

  BrownianMotionProcess generator;
  underlying.Simulate(360, generator);

  EuropeanOption option(
    OptionType::kCall,
    underlying
  );

  Calculator calculator;
  const auto kDelta = calculator.CalculateDelta(option);
  std::cout << "Delta: " << kDelta << std::endl;

  return 0;
}