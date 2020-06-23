% By solving min ||Ax-b||^2+\lambda ||x||_1, we can verify the efficiency of the proposed inexact theta scheme in the paper 
% "The Glowinski–Le Tallec splitting method revisited: A general convergence and convergence rate analysis"

% A is the traing sample matrix
% b is the training labels (continuous)
% lambda is the trade-off parameter
% l_1 norm is the penalty for obtaing the sparsity

%%%%Data generation %%%%%
m  = 200000;   % number of examples
n  = 3000000;     % number of features
p1 = 100/n;     % sparsity density of solution vector
p2 = 0.0001;    % sparsity density of A

x0 = sprandn(n, 1, p1);
A = sprandn(m, n, p2);
b = A*x0 + 0.1*randn(m,1);

lambda_max = norm(A'*b, 'inf');
lambda = 0.1*lambda_max; 

%%%%% Parameter settings %%%
LipA1=normest(A,1e-2);beta1=2./LipA1^2;beta2=2./LipA1^2;              %% Beta1 and Beta2
alpha=0.9;                                                            %% The proposed inexact criterion
tol1=1e-2;                                                            %% Toleracies of fixed precisions
tol=1e-5;                                                             %% Toleracy of lsqr     

%%Assistance functions
function p = objective(A, b, lambda,z)
    p = ( 1/2*sum((A*z - b).^2) + lambda*norm(z,1) );
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

function y=afun(eta)
global A1;
global beta22;
       temp=A1'*eta;
       y=beta22*(A1*temp)+eta;
end

%%%% Thetas-scheme with lsqr %%%%
function [z, history] = lasso_theta_lsqr(A, b, lambda, beta1,beta2)

t_start = tic;

MAX_ITER = 500;
[~, n] = size(A);

z = zeros(n,1);
etak = zeros(n,1);
Atb=A'*b;
gamma=1+beta1/beta2;beta21=beta2/beta1;bg=beta21*gamma;

for k=1:MAX_ITER
    
    zold=z;
    %%%%theta step%%%%%
    ztemp=z-beta1*(A'*(A*z)-Atb);
    s = bg*shrinkage(ztemp, lambda*beta1)-beta21*ztemp;
    
    %%%%1-theta step %%%%
    [etak, ~, ~, iters]=lsqr([sqrt(beta2)*A;speye(n)],[sqrt(beta2)*b;s],1e-6,200,[],[],etak);
    
    %%%%theta step %%%%
    vtemp=etak-beta1*(A'*(A*etak)-Atb);
    z=shrinkage(vtemp,lambda*beta1);
    history.objval(k)  = objective(A, b, lambda,z);
    history.lsqr_iters(k) = iters;
    history.primal(k)=norm(z-etak)+norm(etak-zold);
    history.time(k)=toc(t_start);

end
end

%%%% Theta-scheme with dynamic precisions(The proposed one)%%%%
function [z, history] = lasso_theta_cgs(A, b, lambda, beta1,beta2,alpha,objstar)

MAX_ITER = 500;
[m, n] = size(A);
t_start = tic;
global A1;
global beta22;
A1=A;
beta22=beta2;

z = zeros(n,1);
etak = zeros(m,1);
Atb=A'*b;
AATb=beta2*(A*Atb);
gamma=1+beta1/beta2;beta21=beta2/beta1;bg=beta21*gamma;
iter=0;
for k=1:MAX_ITER
    
    %%%%theta step%%%%%
    
    ztemp=z-beta1*(A'*(A*z)-Atb);
    s = bg*shrinkage(ztemp, lambda*beta1)-beta21*ztemp;
    hk=Atb+s./beta2;hktemp=AATb+A*s;
    %%%%1-theta step %%%%
    if k>1
        tolinner=(alpha*norm(afun(etak)-hktempold))/norm(hktemp);
        [etak,~,~,iter,~] = cgs(@afun,hktemp,tolinner,200,[],[],etak);
    end
    v=beta2*(hk-A'*etak);
    %%%%theta step %%%%
    vtemp=v-beta1*(A'*(A*v)-Atb);
    z=shrinkage(vtemp,lambda*beta1);
    hktempold=hktemp;
  
    history.objval(k)  = objective(A, b, lambda,z);
    history.cgs_iters(k) = iter;
    history.time(k)=toc(t_start);
    
    %%% stopping criterion %%%%
    if history.objval(k)<=objstar
        disp('lasso_theta_cgs:');
        break;
    end  
end
history.average=sum(history.cgs_iters)./k;
history.max=max(history.cgs_iters);
end


%%%% Theta-scheme with fixed precision %%%%

function [z, history] = lasso_theta_cgs1(A, b, lambda, beta1,beta2,objstar,tol)

MAX_ITER = 500;
[m, n] = size(A);
t_start = tic;
global A1;
global beta22;
A1=A;
beta22=beta2;

z = zeros(n,1);
etak = zeros(m,1);
Atb=A'*b;
AATb=beta2*(A*Atb);
gamma=1+beta1/beta2;beta21=beta2/beta1;bg=beta21*gamma;
for k=1:MAX_ITER
    
    %%%%theta step%%%%%
    ztemp=z-beta1*(A'*(A*z)-Atb);
    s = bg*shrinkage(ztemp, lambda*beta1)-beta21*ztemp;
    hk=Atb+s./beta2;hktemp=AATb+A*s;
   
    %%%%1-theta step %%%%
    [etak,~,~,iter,~] = cgs(@afun,hktemp,tol,200,[],[],etak);
%     history.error(k)=norm(afun(etak)-hktemp)./norm(hktemp);
    v=beta2*(hk-A'*etak);
    %%%%theta step %%%%
    vtemp=v-beta1*(A'*(A*v)-Atb);
    z=shrinkage(vtemp,lambda*beta1);
    history.objval(k)  = objective(A, b, lambda,z);
    history.cgs_iters(k) = iter;
    history.time(k)=toc(t_start);
    if history.objval(k)<=objstar
        disp('lasso_theta_cgs:');
        break;
    end  
end
history.average=sum(history.cgs_iters)./k;
history.max=max(history.cgs_iters);
end


%%%%% Comparison %%%%%%
tic
[x1, history1] = lasso_theta_lsqr(A, b, lambda,beta1,beta2);      %% Inexact theta-scheme using lsqr 
toc
objstar=history1.objval(end);

tic 
[x2,history2]=lasso_theta_cgs(A,b,lambda,beta1,beta2,alpha,objstar);    %% Inexact theta-scheme using our critertion
toc

tic 
[x3,history3]=lasso_theta_cgs1(A,b,lambda,beta1,beta2,objstar,tol1);    %% Inexact theta-scheme using fixed precison: tol1
toc
