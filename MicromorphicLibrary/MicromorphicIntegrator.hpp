#ifndef MICROMORPHICINTEGRATOR_HPP
#define MICROMORPHICINTEGRATOR_HPP

#include "mfem.hpp"
#include "MicromorphicPartialAssemblyKernels.hpp"
#include <memory>

#include <iostream> // TAKE OUT BEFORE PUSHING


enum Block
{
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight
};

class MicromorphicIntegrator : public mfem::BilinearFormIntegrator
{
    private:

        Block block;
        mfem::Coefficient *lambda, *mu;
        mfem::Coefficient *lambda2, *mu2; // For Block::BottomRight only
        mfem::DenseMatrix dNdeta, dNdeta2, dNdx, dNdx2, W, R, W2, R2;
        mfem::Vector N, N2;

        // Partial assembly variables
        const mfem::DofToQuad *maps;       
        const mfem::GeometricFactors *geom; 
        const mfem::FiniteElementSpace *trial_fespace;
        const mfem::FiniteElementSpace *test_fespace;
        std::unique_ptr<mfem::QuadratureSpace> q_space;
        std::unique_ptr<mfem::QuadratureFunction> q_vec, q_vec2;
        std::unique_ptr<mfem::CoefficientVector> lambda_quad, mu_quad;
        std::unique_ptr<mfem::CoefficientVector> lambda2_quad, mu2_quad; // For Block::BottomRight only

    public:

        MicromorphicIntegrator(mfem::Coefficient *l, mfem::Coefficient *m, 
                                Block b) : lambda(l), mu(m), block(b) {}

        MicromorphicIntegrator(mfem::Coefficient *l, mfem::Coefficient *m, mfem::Coefficient *l2, 
                                mfem::Coefficient *m2, Block b) : lambda(l), mu(m), lambda2(l2), 
                                mu2(m2), block(b) {}

        void AssembleElementMatrix(const mfem::FiniteElement& el,mfem::ElementTransformation& Tr, 
                                    mfem::DenseMatrix& elmat);

        void AssembleElementMatrix2(const mfem::FiniteElement &trial_fe, 
            const mfem::FiniteElement &test_fe, mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat);

        // Partial assembly functions
        void AssemblePA(const mfem::FiniteElementSpace& fes);
        void AssemblePA(const mfem::FiniteElementSpace& trial_fes, const mfem::FiniteElementSpace& test_fes);
        void AddMultPA(const mfem::Vector& x, mfem::Vector& y) const;
        void AssembleDiagonalPA(mfem::Vector& diag);

};

#endif
