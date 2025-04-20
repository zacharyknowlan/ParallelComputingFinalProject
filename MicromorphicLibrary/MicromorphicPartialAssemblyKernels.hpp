#ifndef MICROMORPHIC_PARTIAL_ASSEMBLY_KERNELS_HPP
#define MICROMORPHIC_PARTIAL_ASSEMBLY_KERNELS_HPP

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/tensor.hpp"

template<int dim>
void MicromorphicTopLeftAddMultPA(const int nDofs, const mfem::FiniteElementSpace &trial_fespace,
                                    const mfem::CoefficientVector &LambdaAlpha, const mfem::CoefficientVector &MuAlpha,
                                    const mfem::GeometricFactors &geom, const mfem::DofToQuad &maps, 
                                    mfem::QuadratureFunction &QVec, const mfem::Vector &x, mfem::Vector &y)
{
    static constexpr int d = dim;
    const auto &ir = QVec.GetIntRule(0); // All elements must be the same
    const int numPoints = ir.GetNPoints();
    const int numEls = trial_fespace.GetNE();

    // Interpolate physical derivatives to quadrature points
    const mfem::QuadratureInterpolator *E_To_Q_Map = trial_fespace.GetQuadratureInterpolator(ir);
    E_To_Q_Map->SetOutputLayout(mfem::QVectorLayout::byNODES); // Must have byNODES ordering to work
    E_To_Q_Map->PhysDerivatives(x, QVec);

    // Copy and reshape the data onto the device
    const auto LambdaAlphaDev = mfem::Reshape(LambdaAlpha.Read(), numPoints, numEls);
    const auto MuAlphaDev = mfem::Reshape(MuAlpha.Read(), numPoints, numEls);
    const auto J = mfem::Reshape(geom.J.Read(), numPoints, d, d, numEls);
    auto Q = mfem::Reshape(QVec.ReadWrite(), numPoints, d, d, numEls);
    const mfem::real_t* ipWeights = ir.GetWeights().Read();

    mfem::forall_2D(numEls, numPoints, 1, [=] MFEM_HOST_DEVICE (int e)
    {
        MFEM_FOREACH_THREAD(p, x, numPoints)
        {
            // Get inverse jacobian
            auto invJ = mfem::internal::inv(mfem::internal::make_tensor<d, d>([&](int i, int j) {return J(p,i,j,e);}));

            // Set weight (note 1/det(invJ) = det(J))
            const mfem::real_t w = ipWeights[p] / mfem::internal::det(invJ);
            
            // Get the displacement gradient at the integration point
            mfem::internal::tensor<mfem::real_t, d, d> grad_u;
            
            for (int i=0; i<d; i++)
            {
                for (int j=0; j<d; j++)
                {
                    grad_u(i,j) = Q(p,i,j,e);
                }
            }

            // Compute divergence of u
            mfem::real_t div = 0.;
            for (int i=0; i<d; i++)
            {
                div += grad_u(i,i);
            }

            // Compute weak form integrand (test function derivative is not symmetrized)
            for (int m=0; m<d; m++)
            {
                for (int i=0; i<d; i++)
                {
                    mfem::real_t contraction = 0.;
                    for (int j=0; j<d; j++)
                    {
                        contraction += invJ(j,m)*(grad_u(i,j) + grad_u(j,i));
                    }
                    Q(p,m,i,e) = w*(LambdaAlphaDev(p,e)*invJ(i,m)*div + MuAlphaDev(p,e)*contraction);
                }
            }
        }
    });

    // Reduce quadrature function to an E-Vector
    const auto QRead = mfem::Reshape(QVec.Read(), numPoints, d, d, numEls);
    const auto G = mfem::Reshape(maps.G.Read(), numPoints, d, nDofs);
    auto yDev = mfem::Reshape(y.ReadWrite(), nDofs, d, numEls);
    mfem::forall_2D(numEls, d, nDofs, [=] MFEM_HOST_DEVICE (int e)
    {
        MFEM_FOREACH_THREAD(i, y, nDofs)
        {
            MFEM_FOREACH_THREAD(q, x, d)
            {
                mfem::real_t sum = 0.;
                for (int m = 0; m<d; m++)
                {
                    for (int p=0; p<numPoints; p++)
                    {
                        sum += QRead(p,m,q,e)*G(p,m,i);
                    }
                }
                yDev(i,q,e) += sum;
            }
        }
    });
}

template<int dim>
void MicromorphicTopRightAddMultPA(const int nDofs, const mfem::FiniteElementSpace &trial_fespace,
                                    const mfem::CoefficientVector &LambdaBeta, const mfem::CoefficientVector &MuBeta,
                                    const mfem::GeometricFactors &geom, const mfem::DofToQuad &maps, 
                                    mfem::QuadratureFunction &QVec, const mfem::Vector &x, mfem::Vector &y)
{
    static constexpr int d = dim;
    const auto &ir = QVec.GetIntRule(0); // All elements must be the same
    const int numPoints = ir.GetNPoints();
    const int numEls = trial_fespace.GetNE(); 

    // Interpolate values onto quadrature points
    const mfem::QuadratureInterpolator *E_To_Q_Map = trial_fespace.GetQuadratureInterpolator(ir);
    E_To_Q_Map->SetOutputLayout(mfem::QVectorLayout::byNODES); // Must have byNODES ordering to work
    E_To_Q_Map->Values(x, QVec); 

    // Copy and reshape the data onto the device
    const auto LambdaBetaDev = mfem::Reshape(LambdaBeta.Read(), numPoints, numEls);
    const auto MuBetaDev = mfem::Reshape(MuBeta.Read(), numPoints, numEls);
    const auto J = mfem::Reshape(geom.J.Read(), numPoints, d, d, numEls);
    auto Q = mfem::Reshape(QVec.ReadWrite(), numPoints, d*d, numEls); // Use flat representation of phi
    const mfem::real_t* ipWeights = ir.GetWeights().Read();

    mfem::forall_2D(numEls, numPoints, 1, [=] MFEM_HOST_DEVICE (int e)
    {
        MFEM_FOREACH_THREAD(p, x, numPoints) // x is shadowed not x vector
        {
            // Get inverse jacobian
            auto invJ = mfem::internal::inv(mfem::internal::make_tensor<d, d>([&](int i, int j) {return J(p,i,j,e);}));

            // Set weight (note 1/det(invJ) = det(J))
            const mfem::real_t w = ipWeights[p] / mfem::internal::det(invJ);
            
            // Get micro displacement gradient at the integration point
            mfem::internal::tensor<mfem::real_t, d, d> phi;
            for (int i=0; i<d; i++)
            {
                for (int j=0; j<d; j++)
                {
                    phi(i,j) = Q(p,d*i+j,e);
                }
            }

            // Compute micro divergence
            mfem::real_t micro_div = 0.;
            for (int i=0; i<d; i++)
            {
                micro_div += phi(i,i);
            }

            // Compute weak form integrand (test function derivative is not symmetrized)
            for (int m=0; m<d; m++)
            {
                for (int i=0; i<d; i++)
                {
                    mfem::real_t contraction = 0.;
                    for (int j=0; j<d; j++)
                    {
                        contraction += invJ(j,m)*(phi(i,j) + phi(j,i));
                    }
                    Q(p,m*d+i,e) = -w*(LambdaBetaDev(p,e)*invJ(i,m)*micro_div + MuBetaDev(p,e)*contraction);
                }
            }
        }
    });

    // Reduce quadrature function to an E-Vector
    const auto QRead = mfem::Reshape(QVec.Read(), numPoints, d*d, numEls);
    const auto G = mfem::Reshape(maps.G.Read(), numPoints, d, nDofs);
    auto yDev = mfem::Reshape(y.ReadWrite(), nDofs, d, numEls);
    mfem::forall_2D(numEls, d, nDofs, [=] MFEM_HOST_DEVICE (int e)
    {
        MFEM_FOREACH_THREAD(i, y, nDofs)
        {
            MFEM_FOREACH_THREAD(q, x, d)
            {
                mfem::real_t sum = 0.;
                for (int m = 0; m<d; m++)
                {
                    for (int p=0; p<numPoints; p++)
                    {
                        sum += QRead(p,m*d+q,e)*G(p,m,i);
                    }
                }
                yDev(i,q,e) += sum;
            }
        }
    });
}

template<int dim>
void MicromorphicBottomLeftAddMultPA(const int nDofs, const mfem::FiniteElementSpace &trial_fespace,
                                    const mfem::CoefficientVector &LambdaBeta, const mfem::CoefficientVector &MuBeta,
                                    const mfem::GeometricFactors &geom, const mfem::DofToQuad &maps, 
                                    mfem::QuadratureFunction &QVec, const mfem::Vector &x, mfem::Vector &y)
{
    static constexpr int d = dim;
    const auto &ir = QVec.GetIntRule(0); // All elements must be the same
    const int numPoints = ir.GetNPoints();
    const int numEls = trial_fespace.GetNE();

    // Interpolate physical derivatives to quadrature points
    const mfem::QuadratureInterpolator *E_To_Q_Map = trial_fespace.GetQuadratureInterpolator(ir);
    E_To_Q_Map->SetOutputLayout(mfem::QVectorLayout::byNODES); // Must have byNODES ordering to work
    E_To_Q_Map->PhysDerivatives(x, QVec); 

    // Transfer and reshape data onto the device
    const auto LambdaBetaDev = mfem::Reshape(LambdaBeta.Read(), numPoints, numEls);
    const auto MuBetaDev = mfem::Reshape(MuBeta.Read(), numPoints, numEls);
    const auto J = mfem::Reshape(geom.J.Read(), numPoints, d, d, numEls);
    auto Q = mfem::Reshape(QVec.ReadWrite(), numPoints, d, d, numEls);
    const mfem::real_t* ipWeights = ir.GetWeights().Read();

    // Use "batched" forall_2D looping over the elements in the outer loop N 
    // and the integration points of an  element X, Y is just set to 1
    mfem::forall_2D(numEls, numPoints, 1, [=] MFEM_HOST_DEVICE (int e)
    {
        MFEM_FOREACH_THREAD(p, x, numPoints)
        {
            // Get jacobian determinant 
            auto detJ = mfem::internal::det(mfem::internal::make_tensor<d, d>([&](int i, int j) {return J(p,i,j,e);}));
            
            // Get gradient at the integration point
            mfem::internal::tensor<mfem::real_t, d, d> grad_u;
            for (int j=0; j<d; j++)
            {
                for (int i=0; i<d; i++)
                {
                    grad_u(i,j) = Q(p,i,j,e);
                }
            }

            // Integration point weight scaled by element size
            const mfem::real_t w = ipWeights[p] * detJ;

            // Compute divergence
            mfem::real_t div = 0.;
            for (int i=0; i<d; i++)
            {
                div += grad_u(i,i);
            }

            // Compute weak form (note no inverse Jacobian is needed)
            for (int i=0; i<d; i++)
            {
                for (int j=0; j<d; j++)
                {
                    Q(p,i,j,e) = -w*(LambdaBetaDev(p,e)*div*(i==j) + MuBetaDev(p,e)*(grad_u(i,j) + grad_u(j,i)));
                }
            }
        }
    });

    // Note that nDofs corresponds to the test space
    const auto QRead = mfem::Reshape(QVec.Read(), numPoints, d*d, numEls);
    const auto B = mfem::Reshape(maps.B.Read(), numPoints, nDofs);
    auto yDev = mfem::Reshape(y.ReadWrite(), nDofs, d*d, numEls);
    mfem::forall_2D(numEls, d, nDofs, [=] MFEM_HOST_DEVICE (int e)
    {
        MFEM_FOREACH_THREAD(i, y, nDofs)
        {
            MFEM_FOREACH_THREAD(q, x, d*d)
            {
                mfem::real_t sum = 0.;
                for (int p=0; p<numPoints; p++ )
                {
                    sum += QRead(p,q,e)*B(p,i);
                }
                yDev(i,q,e) += sum;
            }
        }
    });
}

template<int dim>
void MicromorphicBottomRightAddMultPA(const int nDofs, const mfem::FiniteElementSpace &trial_fespace,
                                        const mfem::CoefficientVector &LambdaA, const mfem::CoefficientVector &MuA,
                                        const mfem::CoefficientVector &LambdaGamma, const mfem::CoefficientVector &MuGamma,
                                        const mfem::GeometricFactors &geom, const mfem::DofToQuad &maps, 
                                        mfem::QuadratureFunction &QVec, mfem::QuadratureFunction &QVec2,
                                        const mfem::Vector &x, mfem::Vector &y)
{
    static constexpr int d = dim;
    const auto &ir = QVec.GetIntRule(0); // All elements must be the same
    const int numPoints = ir.GetNPoints();
    const int numEls = trial_fespace.GetNE();
    
    // Interpolate physical derivatives and values onto the quadrature points 
    const mfem::QuadratureInterpolator *E_To_Q_Map = trial_fespace.GetQuadratureInterpolator(ir);
    E_To_Q_Map->SetOutputLayout(mfem::QVectorLayout::byNODES); // Must have byNODES ordering to work
    E_To_Q_Map->PhysDerivatives(x, QVec); // Interpolate physical derivatives to quadrature points
    E_To_Q_Map->Values(x, QVec2); // Interpolate values onto quadrature points

    // Transfer data onto the device 
    const auto LambdaADev = mfem::Reshape(LambdaA.Read(), numPoints, numEls);
    const auto MuADev = mfem::Reshape(MuA.Read(), numPoints, numEls);
    const auto LambdaGammaDev = mfem::Reshape(LambdaGamma.Read(), numPoints, numEls);
    const auto MuGammaDev = mfem::Reshape(MuGamma.Read(), numPoints, numEls);
    const auto J = mfem::Reshape(geom.J.Read(), numPoints, d, d, numEls);
    auto Q = mfem::Reshape(QVec.ReadWrite(), numPoints, d*d, d, numEls);
    auto Q2 = mfem::Reshape(QVec2.ReadWrite(), numPoints, d*d, numEls);
    const mfem::real_t* ipWeights = ir.GetWeights().Read();

    mfem::forall_2D(numEls, numPoints, 1, [=] MFEM_HOST_DEVICE (int e)
    {
        MFEM_FOREACH_THREAD(p, x, numPoints)
        {
            // Get inverse jacobian
            auto invJ = mfem::internal::inv(mfem::internal::make_tensor<d, d>([&](int i, int j) {return J(p,i,j,e);}));
            
            // Integration point weight scaled by element size
            const mfem::real_t w = ipWeights[p] / mfem::internal::det(invJ);

            // Get gradient at the integration point
            mfem::internal::tensor<mfem::real_t, d*d, d> grad_phi;
            for (int i=0; i<d; i++)
            {
                for (int j=0; j<d; j++)
                {
                    for (int k=0; k<d; k++)
                    {
                        grad_phi(i*d+j,k) = Q(p,i*d+j,k,e);
                    }
                }
            }

            mfem::internal::tensor<mfem::real_t, d, d> phi;
            for (int i=0; i<d; i++)
            {
                for (int j=0; j<d; j++)
                {
                    phi(i,j) = Q2(p,i*d+j,e);
                }
            }

            // The three divergence terms for the length scale terms 
            mfem::internal::tensor<mfem::real_t, d> div1, div2, div3;
            for (int i=0; i<d; i++)
            {
                div1(i) = 0.;
                div2(i) = 0.;
                div3(i) = 0.;
                for (int j=0; j<d; j++)
                {
                    div1(i) += grad_phi(i*d+j,j);
                    div2(i) += grad_phi(j*d+i,j);
                    div3(i) += grad_phi(j*d+j,i);
                }
            }

            // Compute the divergence of phi
            mfem::real_t div = 0.;
            for (int i=0; i<d; i++)
            {
                div += phi(i,i);
            }

            // Compute the length scale dependent terms of the weak form
            for (int m=0; m<d; m++)
            {
                for (int i=0; i<d; i++)
                {
                    for (int j=0; j<d; j++)
                    {
                        mfem::real_t div_contraction = 0.;
                        mfem::real_t contraction = 0.;
                        for (int k=0; k<d; k++)
                        {
                            div_contraction += invJ(k,m)*(div1(k)*(i==j) + div1(i)*(j==k) + div1(j)*(i==k));
                            div_contraction += invJ(k,m)*(div2(k)*(i==j) + div2(i)*(j==k) + div2(j)*(i==k));
                            div_contraction += invJ(k,m)*(div3(k)*(i==j) + div3(i)*(j==k) + div3(j)*(i==k));
                            contraction += invJ(k,m)*(grad_phi(i*d+j,k) + grad_phi(k*d+i,j) + grad_phi(j*d+k,i));
                            contraction += invJ(k,m)*(grad_phi(k*d+j,i) + grad_phi(i*d+k,j) + grad_phi(j*d+i,k));
                        }
                        Q(p,i*d+j,m,e) = w*(LambdaADev(p,e)*div_contraction + MuADev(p,e)*contraction);
                    }
                }
            }

            // Compute the length scale independent phi terms 
            for (int i=0; i<d; i++)
            {
                for (int j=0; j<d; j++)
                {
                    Q2(p,i*d+j,e) = w*(LambdaGammaDev(p,e)*div*(i==j) + MuGammaDev(p,e)*(phi(i,j) + phi(j,i)));
                }
            }
        }
    });

    const auto QRead = mfem::Reshape(QVec.Read(), numPoints, d*d, d, numEls);
    const auto Q2Read = mfem::Reshape(QVec2.Read(), numPoints, d*d, numEls);
    const auto G = mfem::Reshape(maps.G.Read(), numPoints, d, nDofs);
    const auto B = mfem::Reshape(maps.B.Read(), numPoints, nDofs);
    auto yDev = mfem::Reshape(y.ReadWrite(), nDofs, d*d, numEls);
    mfem::forall_2D(numEls, d, nDofs, [=] MFEM_HOST_DEVICE (int e)
    {
        MFEM_FOREACH_THREAD(i, y, nDofs)
        {
            MFEM_FOREACH_THREAD(q, x, d*d)
            {
                mfem::real_t sum = 0., sum2 = 0.;
                for (int p=0; p<numPoints; p++)
                {
                    for (int m = 0; m<d; m++)
                    {
                        sum += QRead(p,q,m,e)*G(p,m,i);
                    }
                    sum2 += Q2Read(p,q,e)*B(p,i);
                }
                yDev(i,q,e) += sum + sum2;
            }
        }
    });
}

template<int dim>
void MicromorphicTopLeftAssembleDiagonalPA(const int nDofs, const mfem::CoefficientVector &LambdaAlpha,
                                            const mfem::CoefficientVector &MuAlpha, const mfem::GeometricFactors &geom,
                                            const mfem::DofToQuad &maps, mfem::QuadratureFunction &QVec, mfem::Vector &diag)
{
    // Assuming all elements are the same
    const auto &ir = QVec.GetIntRule(0);
    static constexpr int d = dim;
    const int numPoints = ir.GetNPoints();
    const int numEls = LambdaAlpha.Size()/numPoints;

    // Transfer data onto the device
    const auto LambdaAlphaDev = mfem::Reshape(LambdaAlpha.Read(), numPoints, numEls);
    const auto MuAlphaDev = mfem::Reshape(MuAlpha.Read(), numPoints, numEls);
    const auto J = mfem::Reshape(geom.J.Read(), numPoints, d, d, numEls);
    auto Q = mfem::Reshape(QVec.ReadWrite(), numPoints, d, d, d, numEls);
    const mfem::real_t *ipWeights = ir.GetWeights().Read();
    mfem::forall_2D(numEls, numPoints, 1, [=] MFEM_HOST_DEVICE (int e)
    {
        MFEM_FOREACH_THREAD(p, x,numPoints)
        {
            auto invJ = mfem::internal::inv(mfem::internal::make_tensor<d, d>([&](int i, int j) {return J(p, i, j, e);}));
            const mfem::real_t w = ipWeights[p] / det(invJ);

            for (int m=0; m<d; m++)
            {
                for (int n=0; n<d; n++)
                {
                    for(int i=0; i<d; i++)
                    {
                        mfem::real_t contraction = 0.;
                        for (int j=0; j<d; j++)
                        {
                            contraction += invJ(j,m)*(invJ(j,n) + (i==j)*invJ(i,n));
                        }
                        Q(p,m,n,i,e) = w*(LambdaAlphaDev(p,e)*invJ(i,m)*invJ(i,n) + MuAlphaDev(p,e)*contraction);
                    }
                }
            }
        }
    });

   // Reduce quadrature function to an E-Vector
   const auto QRead = mfem::Reshape(QVec.Read(), numPoints, d, d, d, numEls);
   auto diagDev = mfem::Reshape(diag.Write(), nDofs, d, numEls);
   const auto G = mfem::Reshape(maps.G.Read(), numPoints, d, nDofs);
   mfem::forall_2D(numEls, d, nDofs, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(i, y, nDofs)
      {
         MFEM_FOREACH_THREAD(q, x, d)
         {
            mfem::real_t sum = 0.;
            for (int n = 0; n < d; n++)
            {
               for (int m = 0; m < d; m++)
               {
                  for (int p = 0; p < numPoints; p++ )
                  {
                     sum += QRead(p,m,n,q,e)*G(p,m,i)*G(p,n,i);
                  }
               }
            }
            diagDev(i, q, e) = sum;
         }
      }
   });
}

template<int dim>
void MicromorphicBottomRightAssembleDiagonalPA(const int nDofs, const mfem::CoefficientVector &LambdaGamma, 
                                                const mfem::CoefficientVector &MuGamma, const mfem::CoefficientVector &LambdaA,
                                                const mfem::CoefficientVector &MuA, const mfem::GeometricFactors &geom,
                                                const mfem::DofToQuad &maps, mfem::QuadratureFunction &QVec, mfem::QuadratureFunction &QVec2,
                                                mfem::Vector &diag)
{
    // Assuming all elements are the same
    const auto &ir = QVec.GetIntRule(0);
    static constexpr int d = dim;
    const int numPoints = ir.GetNPoints();
    const int numEls = LambdaGamma.Size()/numPoints;

    // Transfer data onto the device 
    const auto LambdaADev = mfem::Reshape(LambdaA.Read(), numPoints, numEls);
    const auto MuADev = mfem::Reshape(MuA.Read(), numPoints, numEls);
    const auto LambdaGammaDev = mfem::Reshape(LambdaGamma.Read(), numPoints, numEls);
    const auto MuGammaDev = mfem::Reshape(MuGamma.Read(), numPoints, numEls);
    const auto J = mfem::Reshape(geom.J.Read(), numPoints, d, d, numEls);
    auto Q = mfem::Reshape(QVec.ReadWrite(), numPoints, d, d, d*d, numEls);
    auto Q2 = mfem::Reshape(QVec2.ReadWrite(), numPoints, d*d, numEls);
    const mfem::real_t *ipWeights = ir.GetWeights().Read();
    mfem::forall_2D(numEls, numPoints, 1, [=] MFEM_HOST_DEVICE (int e)
    {
        MFEM_FOREACH_THREAD(p, x,numPoints)
        {
            auto invJ = mfem::internal::inv(mfem::internal::make_tensor<d, d>([&](int i, int j) {return J(p, i, j, e);}));
            const mfem::real_t w = ipWeights[p] / det(invJ);

            for (int m=0; m<d; m++)
            {
                for (int n=0; n<d; n++)
                {
                    for (int i=0; i<d; i++)
                    {
                        for (int j=0; j<d; j++)
                        {
                            mfem::real_t div_contraction = 0.;
                            mfem::real_t contraction = 0.;
                            for (int k=0; k<d; k++)
                            {
                                for (int z=0; z<d; z++)
                                {
                                    div_contraction += invJ(k,m)*invJ(z,n)*(i==k)*(j==z)*(i==j);
                                    div_contraction += invJ(k,m)*invJ(i,n)*(i==z)*(j==z)*(j==k);
                                    div_contraction += invJ(k,m)*invJ(j,n)*(i==z)*(j==z)*(i==k);
                                    div_contraction += invJ(k,m)*invJ(z,n)*(i==z)*(j==k)*(i==j);
                                    div_contraction += invJ(k,m)*invJ(k,n)*(i==z)*(j==z)*(i==j);
                                    div_contraction += invJ(k,m)*invJ(z,n)*(z==j)*(j==k);
                                    div_contraction += invJ(k,m)*invJ(z,n)*(i==z)*(i==j)*(j==k);
                                    div_contraction += invJ(k,m)*invJ(z,n)*(i==j)*(j==z)*(i==k);
                                    div_contraction += invJ(k,m)*invJ(z,n)*(i==z)*(i==k);
                                }
                                contraction += invJ(k,m)*invJ(k,n);
                                contraction += invJ(k,m)*invJ(i,n)*(i==j)*(j==k);
                                contraction += invJ(k,m)*invJ(j,n)*(i==k)*(j==i);
                                contraction += invJ(k,m)*invJ(j,n)*(k==j);
                                contraction += invJ(k,m)*invJ(k,n)*(i==j); 
                                contraction += invJ(k,m)*invJ(i,n)*(i==k);
                            }
                            Q(p,m,n,i*d+j,e) = w*(LambdaADev(p,e)*div_contraction + MuADev(p,e)*contraction);
                        }
                    }
                }
            }

            for (int i=0; i<d; i++)
            {
                for (int j=0; j<d; j++)
                {
                    Q2(p,i*d+j,e) = w*(LambdaGammaDev(p,e)*(i==j) + MuGammaDev(p,e)*(1 + (i==j)));
                }
            }
        }
    });

   const auto QRead = mfem::Reshape(QVec.Read(), numPoints, d, d, d*d, numEls);
   const auto QRead2 = mfem::Reshape(QVec2.Read(), numPoints, d*d, numEls);
   auto diagDev = mfem::Reshape(diag.Write(), nDofs, d*d, numEls);
   const auto G = mfem::Reshape(maps.G.Read(), numPoints, d, nDofs);
   const auto B = mfem::Reshape(maps.B.Read(), numPoints, nDofs);
   mfem::forall_2D(numEls, d, nDofs, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(i, y, nDofs)
      {
         MFEM_FOREACH_THREAD(q, x, d*d)
         {
            mfem::real_t sum = 0.;
            for (int p = 0; p < numPoints; p++ )
            {
                for (int n = 0; n < d; n++)
                {
                    for (int m = 0; m < d; m++)
                    {
                        sum += QRead(p,m,n,q,e)*G(p,m,i)*G(p,n,i);
                    }
                }
                sum += QRead2(p,q,e)*B(p,i)*B(p,i);
            }
            diagDev(i, q, e) = sum;
        }
      }
   });
}

#endif
