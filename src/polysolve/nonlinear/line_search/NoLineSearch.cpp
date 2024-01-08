#include "NoLineSearch.hpp"

namespace polysolve::nonlinear::line_search
{
    NoLineSearch::NoLineSearch(const json &params, spdlog::logger &logger)
        : Backtracking(params, logger)
    {
        max_energy_incre = params["line_search"]["max_energy_incre"];
    }

    bool NoLineSearch::criteria(
        const TVector &delta_x,
        Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy,
        const TVector &old_grad,
        const TVector &new_x,
        const double new_energy,
        const double step_size) const
    {
        // if (use_grad_norm)
        // {
        //     TVector new_grad;
        //     objFunc.gradient(new_x, new_grad);
        //     return new_grad.norm() < old_grad.norm(); // TODO: cache old_grad.norm()
        // }
        return max_energy_incre < 0 || (new_energy <= old_energy + abs(old_energy) * max_energy_incre);
    }
} // namespace polysolve::nonlinear::line_search
