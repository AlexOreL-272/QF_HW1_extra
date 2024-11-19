#pragma once

#include <torch/torch.h>

#include "../option/option.hpp"

class Calculator {
 public:
  Calculator();

  double CalculateDelta(const BaseOption& option);

 private:
  
};