#ifndef MICROMORPHICINTEGRATOR_HPP
#define MICROMORPHICINTEGRATOR_HPP

#include "mfem.hpp"
#include "MicromorphicPartialAssemblyKernels.hpp"
#include <memory>

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
        mfem::Coefficient *lambda2, *mu2;
        mfem::DenseMatrix dNdeta, dNdeta2, dNdx, dNdx2, W, R, W2, R2;
        mfem::Vector N, N2;
        const mfem::DofToQuad *maps;       
        const mfem::GeometricFactors *geom; 
        const mfem::FiniteElementSpace *trial_fespace;
        const mfem::FiniteElementSpace *test_fespace;
        std::unique_ptr<mfem::QuadratureSpace> q_space;
        std::unique_ptr<mfem::QuadratureFunction> q_vec, q_vec2;
        std::unique_ptr<mfem::CoefficientVector> lambda_quad, mu_quad;
        std::unique_ptr<mfem::CoefficientVector> lambda2_quad, mu2_quad;

    public:

        // Constructor for Block::TopLeft, Block::TopRight, and Block::BottomLeft
        MicromorphicIntegrator(mfem::Coefficient *l, mfem::Coefficient *m, 
                                Block b) : lambda(l), mu(m), block(b) {}

        // This constructor should be used for Block::BottomRight only
        MicromorphicIntegrator(mfem::Coefficient *l, mfem::Coefficient *m, mfem::Coefficient *l2, 
                                mfem::Coefficient *m2, Block b) : lambda(l), mu(m), lambda2(l2), 
                                mu2(m2), block(b) {}

        // Assemble the micromorphic element matrix for a given BilinearForm
        void AssembleElementMatrix(const mfem::FiniteElement& el,mfem::ElementTransformation& Tr, 
                                    mfem::DenseMatrix& elmat);

        // Asssemble the micromorphic element matrix for a given MixedBilinearForm
        void AssembleElementMatrix2(const mfem::FiniteElement &trial_fe, 
            const mfem::FiniteElement &test_fe, mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat);

        // Computes the required data to use AddMultPA for a given BilinearForm
        void AssemblePA(const mfem::FiniteElementSpace& fes);

        // Computes the required data to use AddMultPA for a given MixedBilinearForm
        void AssemblePA(const mfem::FiniteElementSpace& trial_fes, const mfem::FiniteElementSpace& test_fes);

        // Compute the matrix-vector product of the Operator that owns this integrator and Vector x
        void AddMultPA(const mfem::Vector& x, mfem::Vector& y) const;
        
        // Assemble the diagonal for a given BilinearForm to use OperatorJacobiSmooother
        void AssembleDiagonalPA(mfem::Vector& diag);
};

#endif
