#pragma once
#include "Backtracking.hpp"

namespace polysolve::nonlinear::line_search
{
    class NoLineSearch : public Backtracking
    {
    public:
        using Superclass = LineSearch;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        NoLineSearch(const json &params, spdlog::logger &logger);

        virtual std::string name() override { return "None"; }

    protected:
        bool criteria(
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy,
            const TVector &old_grad,
            const TVector &new_x,
            const double new_energy,
            const double step_size) const override;
        
    private:
        double max_energy_incre;
    };
} // namespace polysolve::nonlinear::line_search