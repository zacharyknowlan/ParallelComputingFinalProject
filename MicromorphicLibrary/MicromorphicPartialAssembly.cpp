#include "MicromorphicIntegrator.hpp"
 
 void MicromorphicIntegrator::AssemblePA(const mfem::FiniteElementSpace& fes)
 {
    MFEM_VERIFY(fes.GetOrdering() == mfem::Ordering::byNODES, "Micromorphic PA only implemented for byNODES ordering.");
    
    // Set both test and trial space to fes
    trial_fespace = &fes;
    test_fespace = &fes;

    // Assume that all elements are the same
    const auto Trans = trial_fespace->GetElementTransformation(0);
    const int IntegrationOrder = 2*Trans->OrderGrad(trial_fespace->GetFE(0));
    auto IntRule = &mfem::IntRules.Get(Trans->GetGeometryType(), IntegrationOrder);
 
    // Get required fespace data
    int vdim = trial_fespace->GetVDim();
    int dim = trial_fespace->GetMesh()->SpaceDimension();

    // Define QuadratureSpace and initialize 1st quadrature vector 
    q_space.reset(new mfem::QuadratureSpace(*trial_fespace->GetMesh(), *IntRule)); // Valid only when the mesh has one element type
    q_vec.reset(new mfem::QuadratureFunction(*q_space, vdim*dim)); // Get dervatives of vdim variables in dim dimensions

    // Project the material properties onto the QuadratureSpace
    lambda_quad.reset(new mfem::CoefficientVector(lambda, *q_space, mfem::CoefficientStorage::FULL));
    mu_quad.reset(new mfem::CoefficientVector(mu, *q_space, mfem::CoefficientStorage::FULL));

    // Additional setup for BottomRight
    if (block == Block::BottomRight)
    {
        lambda2_quad.reset(new mfem::CoefficientVector(lambda2, *q_space, mfem::CoefficientStorage::FULL));
        mu2_quad.reset(new mfem::CoefficientVector(mu2, *q_space, mfem::CoefficientStorage::FULL));
        q_vec2.reset(new mfem::QuadratureFunction(*q_space, vdim)); // Components of phi without derivatives
    }

    // DofToQuad::LEXICOGRAPHIC_FULL assumes quad elements and only one type over the mesh
    maps = &trial_fespace->GetFE(0)->GetDofToQuad(*IntRule, mfem::DofToQuad::LEXICOGRAPHIC_FULL);
    geom = trial_fespace->GetMesh()->GetGeometricFactors(*IntRule, mfem::GeometricFactors::JACOBIANS);
 }

 void MicromorphicIntegrator::AssemblePA(const mfem::FiniteElementSpace& trial_fes, const mfem::FiniteElementSpace& test_fes)
 {
    MFEM_VERIFY(trial_fes.GetOrdering() == mfem::Ordering::byNODES, "Micromorphic PA only implemented for byNODES ordering.");
    MFEM_VERIFY(test_fes.GetOrdering() == mfem::Ordering::byNODES, "Micromorphic PA only implemented for byNODES ordering.");
    
    // Set the finite element spaces
    trial_fespace = &trial_fes;
    test_fespace = &test_fes;

    // Get The integration order assuming all elements are the same
    const auto TrialTrans = trial_fespace->GetElementTransformation(0);
    const int TrialIntegrationOrder = 2*TrialTrans->OrderGrad(trial_fespace->GetFE(0));
    const auto TestTrans = test_fespace->GetElementTransformation(0);
    const int TestIntegrationOrder = 2*TestTrans->OrderGrad(test_fespace->GetFE(0));
    const int IntegrationOrder = TrialIntegrationOrder >= TestIntegrationOrder ? TrialIntegrationOrder : TestIntegrationOrder;
    auto IntRule = &mfem::IntRules.Get(TrialTrans->GetGeometryType(), IntegrationOrder);

    // Create QuadratureSpace
    q_space.reset(new mfem::QuadratureSpace(*trial_fespace->GetMesh(), *IntRule)); // Valid only when the mesh has one element type
   
    // Project material properties onto integration points defined by QuadratureSpace
    lambda_quad.reset(new mfem::CoefficientVector(lambda, *q_space, mfem::CoefficientStorage::FULL));
    mu_quad.reset(new mfem::CoefficientVector(mu, *q_space, mfem::CoefficientStorage::FULL));

    // Get required fespace data
    const int vdim = trial_fespace->GetVDim();
    const int dim = trial_fespace->GetMesh()->SpaceDimension();

    switch (block)
    {
        case Block::TopRight:
            q_vec.reset(new mfem::QuadratureFunction(*q_space, vdim)); // Values of vdim (phi) variables
            break;
        case Block::BottomLeft:
            q_vec.reset(new mfem::QuadratureFunction(*q_space, vdim*dim)); // Get dervatives of vdim variables in dim dimensions
            break;
        default:
            MFEM_ABORT("MixedBilinearForm must be initialized with Block::TopRight or Block::BottomLeft.")
            break;
    }

    // Get map data for the testfunction and jacobians based on the trial space
    maps = &test_fespace->GetFE(0)->GetDofToQuad(*IntRule, mfem::DofToQuad::LEXICOGRAPHIC_FULL); // Assumes mesh of only quad-type elements
    geom = trial_fespace->GetMesh()->GetGeometricFactors(*IntRule, mfem::GeometricFactors::JACOBIANS);
 }

// This function gets called by the BilinearFormExtension operator representation
void MicromorphicIntegrator::AddMultPA(const mfem::Vector &x, mfem::Vector &y) const
{
    const int ndofs = test_fespace->GetFE(0)->GetDof(); // Assume all elements are the type and order
    
    switch (trial_fespace->GetMesh()->SpaceDimension())
    {
        case 2:
            switch (block)
            {
                case Block::TopLeft:
                    MicromorphicTopLeftAddMultPA<2>(ndofs, *trial_fespace, *lambda_quad, *mu_quad, *geom, *maps, *q_vec, x, y);
                    break;
                case Block::TopRight:
                    MicromorphicTopRightAddMultPA<2>(ndofs, *trial_fespace, *lambda_quad, *mu_quad, *geom, *maps, *q_vec, x, y);
                    break;
                case Block::BottomLeft:
                    MicromorphicBottomLeftAddMultPA<2>(ndofs, *trial_fespace, *lambda_quad, *mu_quad, *geom, *maps, *q_vec, x, y);
                    break;
                case Block::BottomRight:
                    MicromorphicBottomRightAddMultPA<2>(ndofs, *trial_fespace, *lambda_quad, *mu_quad, *lambda2_quad, *mu2_quad, 
                                                        *geom, *maps, *q_vec, *q_vec2, x, y);
                    break;
                default:
                    MFEM_ABORT("Block must be specified.");
            }
            break;
        case 3:
            switch (block)
            {
                case Block::TopLeft:
                    MicromorphicTopLeftAddMultPA<3>(ndofs, *trial_fespace, *lambda_quad, *mu_quad, *geom, *maps, *q_vec, x, y);
                    break;
                case Block::TopRight:
                    MicromorphicTopRightAddMultPA<3>(ndofs, *trial_fespace, *lambda_quad, *mu_quad, *geom, *maps, *q_vec, x, y);
                    break;
                case Block::BottomLeft:
                    MicromorphicBottomLeftAddMultPA<3>(ndofs, *trial_fespace, *lambda_quad, *mu_quad, *geom, *maps, *q_vec, x, y);
                    break;
                case Block::BottomRight:
                    MicromorphicBottomRightAddMultPA<3>(ndofs, *trial_fespace, *lambda_quad, *mu_quad, *lambda2_quad, *mu2_quad, 
                                                        *geom, *maps, *q_vec, *q_vec2, x, y);
                    break;
                default:
                    MFEM_ABORT("Block must be specified.");
            }
            break;
        default:
            MFEM_ABORT("Only dimensions 2 and 3 supported.");
    }
}

void MicromorphicIntegrator::AssembleDiagonalPA(mfem::Vector &diag)
{
    int vdim = trial_fespace->GetVDim();
    int dim = trial_fespace->GetMesh()->SpaceDimension();
    int ndofs = trial_fespace->GetFE(0)->GetDof(); // Assume all elements are the type and order
    q_vec->SetVDim(dim*dim*vdim); // dim for each shape function derivative and vdim for the number of vdofs

    if (block == Block::BottomRight)
    {
        q_vec2->SetVDim(vdim);
    }

    switch (dim)
    {
        case 2:
            switch (block)
            {
                case Block::TopLeft:
                    MicromorphicTopLeftAssembleDiagonalPA<2>(ndofs, *lambda_quad, *mu_quad, *geom, *maps, *q_vec, diag);
                    break;
                case Block::BottomRight:
                    MicromorphicBottomRightAssembleDiagonalPA<2>(ndofs, *lambda_quad, *mu_quad, *lambda2_quad, *mu2_quad, 
                                                                    *geom, *maps, *q_vec, *q_vec2, diag);
                    break;
                default:
                    MFEM_ABORT("AssembleDiagonalPA only valid for Block::TopLeft and Block::BottomRight");
            }
            break;
        case 3:
            switch (block)
            {
                case Block::TopLeft:
                    MicromorphicTopLeftAssembleDiagonalPA<3>(ndofs, *lambda_quad, *mu_quad, *geom, *maps, *q_vec, diag);
                    break;
                case Block::BottomRight:
                    MicromorphicBottomRightAssembleDiagonalPA<3>(ndofs, *lambda_quad, *mu_quad, *lambda2_quad, *mu2_quad, 
                                                                    *geom, *maps, *q_vec, *q_vec2, diag);
                    break;
                default:
                    MFEM_ABORT("AssembleDiagonalPA only valid for Block::TopLeft and Block::BottomRight");
            }
        default:
            MFEM_ABORT("Only dimensions 2 and 3 supported.");
    }
}
