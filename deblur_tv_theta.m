%%% By solving ||Qz-q||^2+2\eta||z||_{TV}, we can verify the efficiency of the theta-scheme.
% Q is the blurring operator
% q is the resulted image
% \eta is the balanced parameter
% TV norm is classic in this kind of problem

% The function is based on the sources in Amir Beck's homepage which are publicly available. Therefore, the inside functions like
%"denoise_bound_init" can be found therein.(Please also run with classical pacakage HNO)


function [X_out,history]=deblur_tv_theta(Bobs,P,center,lambda,l,u,beta1,beta2,X,objstar,pars)
% Assigning parameres according to pars and/or default values
flag=exist('pars');
if (flag&&isfield(pars,'MAXITER'))
    MAXITER=pars.MAXITER;
else
    MAXITER=100;
end
if(flag&&isfield(pars,'fig'))
    fig=pars.fig;
else
    fig=1;
end
if (flag&&isfield(pars,'BC'))
    BC=pars.BC;
else
    BC='reflexive';
end
if (flag&&isfield(pars,'tv'))
    tv=pars.tv;
else
    tv='iso';
end
if (flag&&isfield(pars,'mon'))
    mon=pars.mon;
else
    mon=0;
end
if (flag&&isfield(pars,'denoiseiter'))
    denoiseiter=pars.denoiseiter;
else
    denoiseiter=10;
end


% If there are two output arguments, initalize the function values vector.
t_start = tic;
[m,n]=size(Bobs);
Pbig=padPSF(P,[m,n]);

switch BC
    case 'reflexive'
        trans=@(X)dct2(X);
        itrans=@(X)idct2(X);
        % computng the eigenvalues of the blurring matrix         
        e1=zeros(m,n);
        e1(1,1)=1;
        Sbig=dct2(dctshift(Pbig,center))./dct2(e1);
    case 'periodic'
        trans=@(X) 1/sqrt(m*n)*fft2(X);
        itrans=@(X) sqrt(m*n)*ifft2(X);
        % computng the eigenvalues of the blurring matrix         
        Sbig=fft2(circshift(Pbig,1-center));
    otherwise
        error('Invalid boundary conditions should be reflexive or periodic');
end
% computing the two dimensional transform of Bobs
Btrans=trans(Bobs);

%The Lipschitz constant of the gradient of ||A(X)-Bobs||^2
% L=2*max(max(abs(Sbig).^2));


% fixing parameters for the denoising procedure 
clear parsin1 parsin2
parsin1.MAXITER=denoiseiter;
parsin1.epsilon=pars.inner_epi1;
parsin1.print=0;
parsin1.tv=tv;

parsin2.MAXITER=denoiseiter;
parsin2.epsilon=pars.inner_epi2;
parsin2.print=0;
parsin2.tv=tv;


% initialization
Z_iter=Bobs;
b12=beta1./beta2;b21=beta2./beta1;A=ones(m,n)+2*beta2*conj(Sbig).*Sbig;
A1=2*beta2*itrans(conj(Sbig).*Btrans);
for i=1:MAXITER 
    %%%%%first step %%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % gradient step
    D=Sbig.*trans(Z_iter)-Btrans;     %D=A*t*z-b
    Z_iter=Z_iter-2*beta1*itrans(conj(Sbig).*D);  %z-2*beta1*it*A^T*D
    Z_iter=real(Z_iter);
     
    %invoking the denoising procedure 
    if (i==1)
        [D_iter,Iter1,~,P]=denoise_bound_init(Z_iter,2*lambda*beta1,l,u,[],parsin1);
    else
        [D_iter,Iter1,~,P]=denoise_bound_init(Z_iter,2*lambda*beta1,l,u,P,parsin1);
    end
    % Compute the total variation and the function value and store it in
    % the function values vector fun_all if exists.
    
    %%%%%Second step%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    D_iter=b21.*((1+b12).*D_iter-Z_iter);
    V_iter=real(itrans(trans(A1+D_iter)./A));
    
    %%%%%Third step %%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D=Sbig.*trans(V_iter)-Btrans;
    Z_iter=V_iter-2*beta1*itrans(conj(Sbig).*D);
    Z_iter=real(Z_iter);
    [D_iter,Iter2,~,P]=denoise_bound_init(Z_iter,2*lambda*beta1,l,u,P,parsin2);
    
    %%%%%%%Function value %%%%%%
    Z_iter=D_iter;
    t=tlv(Z_iter,tv);
    fun_val=norm(Sbig.*trans(Z_iter)-Btrans,'fro')^2+2*lambda*t;
    MSE=norm(Z_iter(:)-X(:))^2./(m*n);
    ISNR=10*log10(norm(Bobs(:)-X(:))^2./norm(Z_iter(:)-X(:))^2);
    history.obj(i)=fun_val;
    history.mse(i)=MSE;
    history.isnr(i)=ISNR;
    history.denoiseiter(i)=Iter1+Iter2;
    history.time(i)=toc(t_start);
    if fun_val<=objstar
        break;
    end
end
X_out=Z_iter;
history.denoise_mean=sum(history.denoiseiter)./i;
history.denoise_max=max(history.denoiseiter);