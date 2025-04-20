#include "MicromorphicIntegrator.hpp"

void MicromorphicIntegrator::AssembleElementMatrix(const mfem::FiniteElement &el,
            mfem::ElementTransformation &Tr, mfem::DenseMatrix &elmat)
{
    int dof = el.GetDof();
    int dim = el.GetDim();
    double L, M, L2, M2;
    int row, col;

    if (block == Block::TopLeft)
    {
        elmat.SetSize(dim*dof);
        W.SetSize(dim*dim, dim*dof);
        R.SetSize(dim*dim, dim*dof);
    }
    else if (block == Block::BottomRight)
    {
        elmat.SetSize(dim*dim*dof);
        W.SetSize(dim*dim, dim*dim*dof);
        R.SetSize(dim*dim, dim*dim*dof);
        W2.SetSize(dim*dim*dim, dim*dim*dof);
        R2.SetSize(dim*dim*dim, dim*dim*dof);
    }
    else 
    {
        MFEM_ABORT("Expected Block::TopLeft and Block::BottomRight for BilinearForm assembly.");
    }

    dNdeta.SetSize(dof, dim);
    dNdx.SetSize(dof, dim);
    N.SetSize(dof);

    elmat = 0.0;

    const mfem::IntegrationRule *ir= &mfem::IntRules.Get(el.GetGeomType(), 2*el.GetOrder());

    for (int ip = 0; ip < ir->GetNPoints(); ip++)
    {
        const mfem::IntegrationPoint &int_point = ir->IntPoint(ip);
        Tr.SetIntPoint(&int_point);

        M = mu->Eval(Tr, int_point) * int_point.weight * Tr.Weight();
        L = lambda->Eval(Tr, int_point) * int_point.weight * Tr.Weight();

        el.CalcDShape(int_point, dNdeta);
        mfem::Mult(dNdeta, Tr.InverseJacobian(), dNdx);
        el.CalcShape(int_point, N);

        W = 0.0;
        R = 0.0; 

        for(int ii=0; ii<dim; ii++)
        {
            for(int jj=0; jj<dim; jj++)
            {
                for(int kk=0; kk<dof; kk++)
                {
                    if(block == Block::TopLeft)
                    {
                        W((dim*ii+jj), (dof*ii+kk)) += dNdx(kk,jj);
                        R((dim*ii+jj), (dof*jj+kk)) += dNdx(kk,ii) * M;
                        R((dim*ii+jj), (dof*ii+kk)) += dNdx(kk,jj) * M;
                        R(((dim+1)*ii), (dof*jj+kk)) += dNdx(kk,jj) * L;
                    } 
                    else if (block == Block::BottomRight)
                    {
                        W((dim*ii+jj), ((dim*ii+jj)*dof+kk)) += N(kk);
                        R((dim*ii+jj), ((dim*ii+jj)*dof+kk)) += N(kk) * M;
                        R((dim*ii+jj), ((dim*jj+ii)*dof+kk)) += N(kk) * M;
                        R(((dim+1)*ii), ((dim+1)*dof*jj+kk)) += N(kk) * L;
                    }
                } 
            }
        }    
        mfem::AddMultAtB(W, R, elmat);

        if (block == Block::BottomRight)
        {
            W2 = 0.0;
            R2 = 0.0;

            M2 = mu2->Eval(Tr, int_point) * int_point.weight * Tr.Weight();
            L2 = lambda2->Eval(Tr, int_point) * int_point.weight * Tr.Weight();
            
            for(int ll=0; ll<dim; ll++)
            {
                for(int mm=0; mm<dim; mm++)
                {
                    for(int pp=0; pp<dim; pp++)
                    {
                        for(int qq=0; qq<dim; qq++)
                        {
                            for(int rr=0; rr<dim; rr++)
                            {
                                for(int kk=0; kk<dof; kk++)
                                {
                                    row = ((pp*dim*dim)+(qq*dim)+rr);
                                    col = ((ll*dim+mm)*dof + kk);
                    
                                    if(pp == ll && qq == mm)
                                    {
                                        R2(row,col) += dNdx(kk,rr) * M2;
                                        W2(row,col) += dNdx(kk,rr);
                                    }
                                    if (qq == ll && rr == mm)
                                    {
                                        R2(row,col) += dNdx(kk,pp) * M2;
                                    }
                                    if (rr == ll && pp == mm)
                                    {
                                        R2(row,col) += dNdx(kk,qq) * M2;
                                    }
                                    if (pp == ll && rr == mm)
                                    {
                                        R2(row,col) += dNdx(kk,qq) * M2;
                                    }
                                    if (qq == ll && pp == mm)
                                    {
                                        R2(row,col) += dNdx(kk,rr) * M2;
                                    }
                                    if (rr == ll && qq == mm)
                                    {  
                                        R2(row,col) += dNdx(kk,pp) * M2;
                                    }
                                    if (rr == ll && pp == qq)
                                    {
                                        R2(row,col) += dNdx(kk,mm) * L2;
                                    }
                                    if (ll == mm && qq == rr)
                                    {
                                        R2(row,col) += dNdx(kk,pp) * L2;
                                    }
                                    if (ll == mm && pp == rr)
                                    {
                                        R2(row,col) += dNdx(kk,qq) * L2;
                                    }
                                    if (rr == mm && pp == qq)
                                    {
                                        R2(row,col) += dNdx(kk,ll) * L2;
                                    }
                                    if (ll == mm && pp == qq)
                                    {
                                        R2(row,col) += dNdx(kk,rr) * L2;
                                    }
                                    if (pp == ll && qq == rr)
                                    {
                                        R2(row,col) += dNdx(kk,mm) * L2;
                                    }
                                    if (pp == mm && qq == rr)
                                    {
                                        R2(row,col) += dNdx(kk,ll) * L2;
                                    }
                                    if (qq == ll && pp == rr)
                                    {
                                        R2(row,col) += dNdx(kk,mm) * L2;
                                    }
                                    if (qq == mm && pp == rr)
                                    {
                                        R2(row,col) += dNdx(kk,ll) * L2;
                                    }  
                                }
                            }
                        }
                    }
                }
            }
            mfem::AddMultAtB(W2, R2, elmat);
        }        
    }
}

void MicromorphicIntegrator::AssembleElementMatrix2(const mfem::FiniteElement &trial_fe, 
const mfem::FiniteElement &test_fe, mfem::ElementTransformation &Tr, mfem::DenseMatrix &elmat)
{
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    int dim = trial_fe.GetDim();
    double L, M;

    if (block == Block::TopRight)
    {
        W.SetSize(dim*dim, test_dof*dim);
        R.SetSize(dim*dim, trial_dof*dim*dim);
        elmat.SetSize(test_dof*dim, trial_dof*dim*dim);
    }
    else if (block == Block::BottomLeft)
    {        
        W.SetSize(dim*dim, test_dof*dim*dim);
        R.SetSize(dim*dim, trial_dof*dim);
        elmat.SetSize(test_dof*dim*dim, trial_dof*dim);
    }
    else 
    {
        MFEM_ABORT("Expected Block::TopRight and Block::BottomLeft for MixedBilinearForm assembly.");
    }
    
    dNdeta.SetSize(trial_dof, dim);
    dNdx.SetSize(trial_dof, dim);
    N.SetSize(trial_dof);

    dNdeta2.SetSize(test_dof, dim);
    dNdx2.SetSize(test_dof, dim);
    N2.SetSize(test_dof);
    
    const mfem::IntegrationRule *ir= &mfem::IntRules.Get(trial_fe.GetGeomType(), 2*trial_fe.GetOrder());
   
    elmat = 0.0;
    
    for(int ip=0; ip<ir->GetNPoints(); ip++)
    {
        const mfem::IntegrationPoint &int_point = ir->IntPoint(ip);
        Tr.SetIntPoint(&int_point);

        M = mu->Eval(Tr, int_point) * int_point.weight * Tr.Weight();
        L = lambda->Eval(Tr, int_point) * int_point.weight * Tr.Weight();
        
        trial_fe.CalcDShape(int_point, dNdeta);
        mfem::Mult(dNdeta, Tr.InverseJacobian(), dNdx);
        trial_fe.CalcShape(int_point, N);
    
        test_fe.CalcDShape(int_point, dNdeta2);
        mfem::Mult(dNdeta2, Tr.InverseJacobian(), dNdx2);
        test_fe.CalcShape(int_point, N2);
        
        R = 0.0;
        W = 0.0;

        for(int ii=0; ii<dim; ii++)
        {
            for(int jj=0; jj<dim; jj++)
            {
                for(int kk=0; kk<trial_dof; kk++)
                {
                    if (block == Block::TopRight)
                    {
                        R((dim*ii+jj), ((dim*ii+jj)*trial_dof+kk)) -= N(kk) * M;
                        R((dim*ii+jj), ((dim*jj+ii)*trial_dof+kk)) -= N(kk) * M;
                        R(((dim+1)*ii), ((dim+1)*trial_dof*jj+kk)) -= N(kk) * L;
                    }
                    else if (block == Block::BottomLeft)
                    {
                        R((dim*ii+jj), (trial_dof*jj+kk)) -= dNdx(kk,ii) * M;
                        R((dim*ii+jj), (trial_dof*ii+kk)) -= dNdx(kk,jj) * M;
                        R(((dim+1)*ii), (trial_dof*jj+kk)) -= dNdx(kk,jj) * L;              
                    }
                }
                for(int ll=0; ll<test_dof; ll++)
                {
                    if (block == Block::TopRight)
                    {
                        W((dim*ii+jj), (test_dof*ii+ll)) += dNdx2(ll,jj);
                    }
                    else if (block == Block::BottomLeft)
                    {
                        W((dim*ii+jj), ((dim*ii+jj)*test_dof+ll)) += N2(ll);
                    }
                }
            }
        }
        mfem::AddMultAtB(W, R, elmat);
    }  
}
