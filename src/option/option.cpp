#include "option.hpp"

#include <torch/torch.h>

BaseOption::BaseOption(
  const OptionType& type,
  const Underlying& underlying
) : type_(type), underlying_(underlying) {}

torch::Tensor BaseOption::Payoff() const {
  const auto kSpotPrice = underlying_.GetSpotPrices();
  const auto kStrike = underlying_.GetStrike();
  
  if (type_ == OptionType::kCall) {
    return torch::max(kSpotPrice - kStrike, torch::zeros_like(kSpotPrice));
  }
  
  return torch::max(kStrike - kSpotPrice, torch::zeros_like(kSpotPrice));
}

torch::Tensor BaseOption::GetUnderlyingPrice() const {
  return underlying_.GetInitialPrice();
}

torch::Tensor norm_cdf(const torch::Tensor& x) {
  return 0.5 * (1 + torch::erf(x / std::sqrt(2)));
}

EuropeanOption::EuropeanOption(
  const OptionType& type,
  const Underlying& underlying
) : BaseOption(type, underlying) {}

torch::Tensor EuropeanOption::PriceAt(
  const torch::Tensor& time
) const {
  const auto kSpotPrice = BaseOption::underlying_.SpotPriceAt(time);
  const auto kStrike = BaseOption::underlying_.GetStrike();
  const auto kTimeToMaturity = BaseOption::underlying_.GetTimeToMaturity();

  if (time.item<double>() < 0.0 || kTimeToMaturity.item<double>() < time.item<double>()) {
    // TODO: throw exception
    return torch::tensor({-1.0});
  }

  const auto kSigma = BaseOption::underlying_.GetVolatility();
  const auto kRiskFreeRate = BaseOption::underlying_.GetRiskFreeRate();
  const auto kDividentRate = BaseOption::underlying_.GetDividendRate();

  const auto kTimeToExpiry = kTimeToMaturity - time;
  const auto kCoeff = kSigma * sqrt(kTimeToExpiry);
  const auto kDPlus = 1.0 / kCoeff * (log(kSpotPrice / kStrike) + 
    (kRiskFreeRate - kDividentRate + kSigma * kSigma / 2.0) * kTimeToExpiry);
  const auto kDMinus = kDPlus - kCoeff;

  if (type_ == OptionType::kCall) {
    const auto result = kSpotPrice * exp(-kDividentRate * kTimeToExpiry) * norm_cdf(kDPlus) - 
      kStrike * exp(-kRiskFreeRate * kTimeToExpiry) * norm_cdf(kDMinus);
    return result;
  }

  const auto result = kStrike * exp(-kRiskFreeRate * kTimeToExpiry) * norm_cdf(-kDMinus) - 
    kSpotPrice * exp(-kDividentRate * kTimeToExpiry) * norm_cdf(-kDPlus);
  return result;
}