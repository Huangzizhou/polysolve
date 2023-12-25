#include "GradientDescent.hpp"

#include <polysolve/Utils.hpp>

namespace polysolve::nonlinear
{

    GradientDescent::GradientDescent(const json &solver_params_,
                                     const bool is_stochastic,
                                     const double characteristic_length,
                                     spdlog::logger &logger)
        : Superclass(solver_params_, characteristic_length, logger), is_stochastic_(is_stochastic)
    {
        if (is_stochastic_)
            erase_component_probability_ = extract_param("StochasticGradientDescent", "erase_component_probability", solver_params_);
    }

    bool GradientDescent::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        direction = -grad;

        if (is_stochastic_)
        {
            Eigen::VectorXd mask = (Eigen::VectorXd::Random(direction.size()).array() + 1.) / 2.;
            direction = (mask.array() < erase_component_probability_).select(Eigen::VectorXd::Zero(direction.size()), direction);
        }

        return true;
    }

} // namespace polysolve::nonlinear
