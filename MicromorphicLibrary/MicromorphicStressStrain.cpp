#include "MicromorphicStressStrain.hpp"

void CalcStrain(const mfem::GridFunction &u, mfem::GridFunction &eps)
{// Check that u and eps are defined on the same mesh.
    MFEM_VERIFY(u.FESpace()->GetNE() == eps.FESpace()->GetNE(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    MFEM_VERIFY(u.FESpace()->GetMesh()->SpaceDimension() == eps.FESpace()->GetMesh()->SpaceDimension(), 
                "GridFunction finite element spaces must be defined on the same mesh.");

    mfem::DenseMatrix dNdeta, dNdx; 
    mfem::Array<int> vdofs;
    int NE = u.FESpace()->GetNE();
    int dim = u.FESpace()->GetMesh()->SpaceDimension();
    eps = 0.0;

    for(int el=0; el<NE; el++)
    {
        const mfem::FiniteElement *FE = u.FESpace()->GetFE(el);
        
        int dof = FE->GetDof();

        dNdeta.SetSize(dof, dim);
        dNdx.SetSize(dof, dim);
        
        vdofs.SetSize(dim*dof);
        u.FESpace()->GetElementVDofs(el,vdofs);

        const mfem::IntegrationRule *int_rule = &mfem::IntRules.Get(FE->GetGeomType(), 2*FE->GetOrder());
        mfem::ElementTransformation *Tr = u.FESpace()->GetElementTransformation(el);

        for(int ip=0; ip<int_rule->GetNPoints(); ip++)
        {
            const mfem::IntegrationPoint &int_point = int_rule->IntPoint(ip);
            Tr->SetIntPoint(&int_point);
            FE->CalcDShape(int_point, dNdeta);
            mfem::Mult(dNdeta, Tr->InverseJacobian(), dNdx);
            
            for(int ii=0; ii<dim; ii++)
            {
                for(int jj=0; jj<dim; jj++)
                {
                    for(int kk=0; kk<dof;kk++)
                    {
                        eps[el+(dim*ii+jj)*NE] += int_point.weight * dNdx(kk,jj) * u[vdofs[kk+ii*dof]] / 2.0;
                        eps[el+(dim*ii+jj)*NE] += int_point.weight * dNdx(kk,ii) * u[vdofs[kk+jj*dof]] / 2.0;
                    }
                }
            }
        }
    }
}

void CalcCoupleStrain(const mfem::GridFunction &u, const mfem::GridFunction &phi, mfem::GridFunction &e)
{// Check that u, phi, and e are defined on the same mesh.
    MFEM_VERIFY(u.FESpace()->GetNE() == phi.FESpace()->GetNE(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    MFEM_VERIFY(u.FESpace()->GetMesh()->SpaceDimension() == phi.FESpace()->GetMesh()->SpaceDimension(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    MFEM_VERIFY(u.FESpace()->GetNE() == e.FESpace()->GetNE(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    MFEM_VERIFY(u.FESpace()->GetMesh()->SpaceDimension() == e.FESpace()->GetMesh()->SpaceDimension(), 
                "GridFunction finite element spaces must be defined on the same mesh.");

    mfem::DenseMatrix dNdeta, dNdx; 
    mfem::Vector N;
    mfem::Array<int> u_vdofs, phi_vdofs;
    int NE = u.FESpace()->GetNE();
    int dim = u.FESpace()->GetMesh()->SpaceDimension();
    e = 0.0;

    for(int el=0; el<NE; el++)
    {
        const mfem::FiniteElement *FE_u = u.FESpace()->GetFE(el);
        int dof_u = FE_u->GetDof();
        const mfem::FiniteElement *FE_phi = phi.FESpace()->GetFE(el);
        int dof_phi = FE_phi->GetDof();

        dNdeta.SetSize(dof_u, dim);
        dNdx.SetSize(dof_u, dim);
        N.SetSize(dof_phi);
        
        u_vdofs.SetSize(dim*dof_u);
        u.FESpace()->GetElementVDofs(el,u_vdofs);

        phi_vdofs.SetSize(dim*dim*dof_phi);
        phi.FESpace()->GetElementVDofs(el,phi_vdofs);

        const mfem::IntegrationRule *int_rule = &mfem::IntRules.Get(FE_u->GetGeomType(), 2*FE_u->GetOrder());
        mfem::ElementTransformation *Tr = u.FESpace()->GetElementTransformation(el);

        for(int ip=0; ip<int_rule->GetNPoints(); ip++)
        {
            const mfem::IntegrationPoint &int_point = int_rule->IntPoint(ip);
            Tr->SetIntPoint(&int_point);
            FE_u->CalcDShape(int_point, dNdeta);
            mfem::Mult(dNdeta, Tr->InverseJacobian(), dNdx);
            FE_phi->CalcShape(int_point, N);
            
            for(int ii=0; ii<dim; ii++)
            {
                for(int jj=0; jj<dim; jj++)
                {
                    for(int kk=0; kk<dof_u; kk++)
                    {
                        e[el+(dim*ii+jj)*NE] += int_point.weight * dNdx(kk,jj) * u[u_vdofs[kk+ii*dof_u]];
                    }
                    for(int ll=0; ll<dof_phi; ll++)
                    {
                        e[el+(dim*ii+jj)*NE] -= int_point.weight * N(ll) * phi[phi_vdofs[ll+(dim*ii+jj)*dof_phi]];
                    }
                }
            }
        }
    }
}

void CalcMicroStrain(const mfem::GridFunction &phi, mfem::GridFunction &K)
{// Check that phi and K are defined on the same mesh.
    MFEM_VERIFY(phi.FESpace()->GetNE() == K.FESpace()->GetNE(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    MFEM_VERIFY(phi.FESpace()->GetMesh()->SpaceDimension() == K.FESpace()->GetMesh()->SpaceDimension(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    
    mfem::DenseMatrix dNdeta, dNdx; 
    mfem::Array<int> vdofs;
    int NE = phi.FESpace()->GetNE();
    int dim = phi.FESpace()->GetMesh()->SpaceDimension();
    K = 0.0;

    for(int el=0; el<NE; el++)
    {
        const mfem::FiniteElement *FE = phi.FESpace()->GetFE(el);
        int dof = FE->GetDof();

        dNdeta.SetSize(dof, dim);
        dNdx.SetSize(dof, dim);
        
        vdofs.SetSize(dim*dim*dof);
        phi.FESpace()->GetElementVDofs(el,vdofs);

        const mfem::IntegrationRule *int_rule = &mfem::IntRules.Get(FE->GetGeomType(), 2*FE->GetOrder());
        mfem::ElementTransformation *Tr = phi.FESpace()->GetElementTransformation(el);

        for(int ip=0; ip<int_rule->GetNPoints(); ip++)
        {
            const mfem::IntegrationPoint &int_point = int_rule->IntPoint(ip);
            Tr->SetIntPoint(&int_point);
            FE->CalcDShape(int_point, dNdeta);
            mfem::Mult(dNdeta, Tr->InverseJacobian(), dNdx);
            
            for(int ii=0; ii<dim; ii++)
            {
                for(int jj=0; jj<dim; jj++)
                {
                    for(int kk=0; kk<dim; kk++)
                    {
                        for(int ll=0; ll<dof; ll++)
                        {
                            int idx_K = dim*dim*ii+dim*jj+kk;
                            int idx_phi = dim*ii+jj;
                            K[el+idx_K*NE] += int_point.weight * dNdx(ll,kk) * phi[vdofs[ll+idx_phi*dof]];
                        }
                    }
                }
            }
        }
    }
}

void CalcStress(const mfem::GridFunction &eps, const mfem::GridFunction &e, mfem::Coefficient &mu, mfem::Coefficient &lambda, 
                mfem::Coefficient &c1, mfem::Coefficient &c2, mfem::GridFunction &sigma)
{// Check that eps, e, and sigma are defined on the same mesh
    MFEM_VERIFY(eps.FESpace()->GetNE() == e.FESpace()->GetNE(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    MFEM_VERIFY(eps.FESpace()->GetMesh()->SpaceDimension() == e.FESpace()->GetMesh()->SpaceDimension(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    MFEM_VERIFY(eps.FESpace()->GetNE() == sigma.FESpace()->GetNE(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    MFEM_VERIFY(eps.FESpace()->GetMesh()->SpaceDimension() == sigma.FESpace()->GetMesh()->SpaceDimension(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    
    int NE = sigma.FESpace()->GetNE();
    int dim = sigma.FESpace()->GetFE(0)->GetDim();
    sigma = 0.0;

    for(int el=0; el<NE; el++)
    {
        const mfem::FiniteElement *FE = sigma.FESpace()->GetFE(el);
        const mfem::IntegrationRule *int_rule = &mfem::IntRules.Get(FE->GetGeomType(), 2*FE->GetOrder());
        mfem::ElementTransformation *Tr = sigma.FESpace()->GetElementTransformation(el);

        double eps_trace = 0.0;
        double e_trace = 0.0;

        for(int kk=0; kk<dim; kk++)
        {
            int idx_kk = NE*(dim*kk+kk)+el;
            eps_trace += eps[idx_kk];
            e_trace += e[idx_kk]; 
        }

        for(int ip=0; ip<int_rule->GetNPoints(); ip++)
        {
            const mfem::IntegrationPoint &int_point = int_rule->IntPoint(ip);
            Tr->SetIntPoint(&int_point);

            double mu_val = mu.Eval(*Tr, int_point) * int_point.weight;
            double lambda_val = lambda.Eval(*Tr, int_point) * int_point.weight;
            double c1_val = c1.Eval(*Tr, int_point) * int_point.weight;
            double c2_val = c2.Eval(*Tr, int_point) * int_point.weight;

            for(int ii=0; ii<dim; ii++)
            {
                for(int jj=0; jj<dim; jj++)
                {
                    int idx_ij = NE*(dim*ii+jj) + el;
                    int idx_ji = NE*(dim*jj+ii) + el;
                    
                    sigma[idx_ij] += 2.0 * mu_val * eps[idx_ij];
                    sigma[idx_ij] += c1_val * (e[idx_ij] + e[idx_ji]);
                    if(ii == jj)
                    {
                        sigma[idx_ij] += lambda_val * eps_trace;
                        sigma[idx_ij] += c2_val * e_trace;
                    }
                }
            }
        }
    }
}

void CalcCoupleStress(const mfem::GridFunction &eps, const mfem::GridFunction &e, mfem::Coefficient &b1, mfem::Coefficient &b2, 
                        mfem::Coefficient &c1, mfem::Coefficient &c2, mfem::GridFunction &s)
{// Check that eps, e, and s are defined on the same mesh.
    MFEM_VERIFY(eps.FESpace()->GetNE() == e.FESpace()->GetNE(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    MFEM_VERIFY(eps.FESpace()->GetMesh()->SpaceDimension() == e.FESpace()->GetMesh()->SpaceDimension(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    MFEM_VERIFY(eps.FESpace()->GetNE() == s.FESpace()->GetNE(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    MFEM_VERIFY(eps.FESpace()->GetMesh()->SpaceDimension() == s.FESpace()->GetMesh()->SpaceDimension(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    
    int NE = s.FESpace()->GetNE();
    int dim = s.FESpace()->GetFE(0)->GetDim();
    s = 0.0;

    for(int el=0; el<NE; el++)
    {
        const mfem::FiniteElement *FE = s.FESpace()->GetFE(el);
        const mfem::IntegrationRule *int_rule = &mfem::IntRules.Get(FE->GetGeomType(), 2*FE->GetOrder());
        mfem::ElementTransformation *Tr = s.FESpace()->GetElementTransformation(el);

        double eps_trace = 0.0;
        double e_trace = 0.0;

        for(int kk=0; kk<dim; kk++)
        {
            int idx_kk = NE*(dim*kk+kk)+el;
            eps_trace += eps[idx_kk];
            e_trace += e[idx_kk]; 
        }

        for(int ip=0; ip<int_rule->GetNPoints(); ip++)
        {
            const mfem::IntegrationPoint &int_point = int_rule->IntPoint(ip);
            Tr->SetIntPoint(&int_point);

            double b1_val = b1.Eval(*Tr, int_point) * int_point.weight;
            double b2_val = b2.Eval(*Tr, int_point) * int_point.weight;
            double c1_val = c1.Eval(*Tr, int_point) * int_point.weight;
            double c2_val = c2.Eval(*Tr, int_point) * int_point.weight;

            for(int ii=0; ii<dim; ii++)
            {
                for(int jj=0; jj<dim; jj++)
                {
                    int idx_ij = NE*(dim*ii+jj) + el;
                    int idx_ji = NE*(dim*jj+ii) + el;
                    
                    s[idx_ij] += b1_val * (e[idx_ij] + e[idx_ji]);
                    s[idx_ij] += 2.0 * c1_val * eps[idx_ij];
    
                    if(ii == jj)
                    {
                        s[idx_ij] += b2_val * e_trace;
                        s[idx_ij] += c2_val * eps_trace;
                    }
                }
            }
        }
    }
}

void CalcMicroStress(const mfem::GridFunction &K, mfem::Coefficient &A1, mfem::Coefficient &A2, mfem::GridFunction &S)
{// Check that K and S are defined on the same mesh.
    MFEM_VERIFY(K.FESpace()->GetNE() == S.FESpace()->GetNE(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    MFEM_VERIFY(K.FESpace()->GetMesh()->SpaceDimension() == S.FESpace()->GetMesh()->SpaceDimension(), 
                "GridFunction finite element spaces must be defined on the same mesh.");
    
    int NE = S.FESpace()->GetNE(); 
    int dim = S.FESpace()->GetFE(0)->GetDim();
    mfem::Vector K_aba(dim), K_aab(dim), K_baa(dim);
    S = 0.0;

    for(int el=0; el<NE; el++)
    {
        const mfem::FiniteElement *FE = S.FESpace()->GetFE(el);
        const mfem::IntegrationRule *int_rule = &mfem::IntRules.Get(FE->GetGeomType(), 2*FE->GetOrder());
        mfem::ElementTransformation *Tr = S.FESpace()->GetElementTransformation(el);

        K_aba = 0.0;
        K_aab = 0.0;
        K_baa = 0.0;

        for(int bb=0; bb<dim; bb++)
            for(int aa=0; aa<dim; aa++)
            {
                int idx_aba = NE * (dim*dim*aa + dim*bb + aa) + el;
                K_aba(bb) += K[idx_aba];

                int idx_aab = NE * (dim*dim*aa + dim*aa + bb) + el;
                K_aab(bb) += K[idx_aab];
                
                int idx_baa = NE * (dim*dim*bb + dim*aa + aa) + el;
                K_baa(bb) += K[idx_baa];
            }

        for(int ip=0; ip<int_rule->GetNPoints(); ip++)
        {
            const mfem::IntegrationPoint &int_point = int_rule->IntPoint(ip);
            Tr->SetIntPoint(&int_point);

            double A1_val = A1.Eval(*Tr, int_point) * int_point.weight;
            double A2_val = A2.Eval(*Tr, int_point) * int_point.weight;

            for(int pp=0; pp<dim; pp++)
            {
                for(int qq=0; qq<dim; qq++)
                {
                    for(int rr=0; rr<dim; rr++)
                    {
                        int idx_pqr = NE * (dim*dim*pp + dim*qq + rr) + el;
                        int idx_qrp = NE * (dim*dim*qq + dim*rr + pp) + el;
                        int idx_rpq = NE * (dim*dim*rr + dim*pp + qq) + el;
                        int idx_prq = NE * (dim*dim*pp + dim*rr + qq) + el;
                        int idx_qpr = NE * (dim*dim*qq + dim*pp + rr) + el;
                        int idx_rqp = NE * (dim*dim*rr + dim*qq + pp) + el;
                        
                        S[idx_pqr] += A1_val * K[idx_pqr]; // A_10
                        S[idx_pqr] += A1_val * K[idx_qrp]; // A_11
                        S[idx_pqr] += A1_val * K[idx_rpq]; // A_11
                        S[idx_pqr] += A1_val * K[idx_prq]; // A_13
                        S[idx_pqr] += A1_val * K[idx_qpr]; // A_14
                        S[idx_pqr] += A1_val * K[idx_rqp]; // A_15

                        if(pp == qq)
                        {
                            S[idx_pqr] += K_baa(rr) * A2_val; // A_1
                            S[idx_pqr] += K_aba(rr) * A2_val; // A_2
                            S[idx_pqr] += K_aab(rr) * A2_val; // A_3
                        }
                        if(qq == rr)
                        {
                            S[idx_pqr] += K_aab(pp) * A2_val; // A_1
                            S[idx_pqr] += K_baa(pp) * A2_val; // A_4
                            S[idx_pqr] += K_aba(pp) * A2_val; // A_5
                        }
                        if(pp == rr)
                        {
                            S[idx_pqr] += K_aab(qq) * A2_val; // A_2
                            S[idx_pqr] += K_baa(qq) * A2_val; // A_5
                            S[idx_pqr] += K_aba(qq) * A2_val; // A_8
                        }
                    }
                }
            }
        }
    }
}
