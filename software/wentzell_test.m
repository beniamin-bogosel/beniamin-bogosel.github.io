function lam = wentzell_test(vec,k,bet)

%Wentzell spectrum computation

% vec = [a0 a b] is a vector containing the Fourier coefficients
% of the radial function which determines the domain

% typical use

% Steklov eigenvalues of the unit disk
% wentzell_test([1 0 0],10,0)

% Wentzell eigenvalues for beta = 2 for some shape
% wentzell_test([1 0 0 0 0.2])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% choose discretization and exterior points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


M                          = 300; % number of source points

M_ext                      = M;

vec                        = vec(:);

dim                        = length(vec);
m                          = (dim-1)/2;

a0                         = vec(1);
a                          = vec(2:m+1);
b                          = vec(m+2:2*m+1);

ind                        = 1:m;

dis                        = linspace(0,2*pi,M+1);
dis                        = dis(1:M)';
evalmat                    = dis*ind;


vals                       = a0+cos(evalmat)*a+sin(evalmat)*b;
ders                       = cos(evalmat)*(b.*ind')-sin(evalmat)*(a.*ind');

ders2                      = -cos(evalmat)*(a.*(ind').^2)-sin(evalmat)*(b.*(ind').^2);


dis                        = dis';
co                         = cos(dis);
si                         = sin(dis);

vals                       = vals';
ders                       = ders';
ders2                      = ders2';

variant = 1;


    points                 = [co.*vals; si.*vals];
    
        normals           = [(vals.*co+ders.*si)./sqrt(vals.^2+ders.^2); (vals.*si-ders.*co)./sqrt(vals.^2+ders.^2)]; 
        extpts             = points+0.1*normals;
    
pp = plot(points(1,:),points(2,:));
axis equal
axis off
set(pp,'LineWidth',2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calculate the matrices %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% bsxfun(@minus,[x1 x2 x3],[y1;y2;y3])=
% [x1-y1  x2-y1  x3-y1; 
%  x1-y2  x2-y2  x3-y2;
%  x1-y3  x2-y3  x3-y3]
%  for C we need the transpose of this...

c1                         = bsxfun(@minus,points(1,:),extpts(1,:)');
c2                         = bsxfun(@minus,points(2,:),extpts(2,:)');
C                          = sqrt(c1.^2+c2.^2);
B                          = 1/(2*pi)*log(C.');
% B is the second matrix in Au=lambda*Bu

% computation of the curvature
H = (vals.^2+2*ders.^2-vals.*ders2)./(vals.^2+ders.^2).^(1.5);

% construction of A
A = zeros(size(B));
 
diffs  = zeros([2,size(c1)]);
diffs(1,:,:) = c1./(C.^2);
diffs(2,:,:) = c2./(C.^2);

n_mat  = zeros([2 1 M]);
n_mat(1,1,:) = normals(1,:);
n_mat(2,1,:) = normals(2,:);
n_mat  = repmat(n_mat,[1 M 1]);

temp  = bsxfun(@dot,diffs,n_mat);
F     = 1/(2*pi)* squeeze(temp(1,:,:));

An    = F.';

der_nor2 = zeros(size(An));

N1  = repmat(normals(1,:),[M 1]);
N2  = repmat(normals(2,:),[M 1]);

Ax  = 1./C.^2-2*squeeze(diffs(1,:,:)).^2;
Bx   = -4*squeeze(diffs(1,:,:)).*squeeze(diffs(2,:,:));

Dx   = 1./C.^2-2*squeeze(diffs(2,:,:)).^2;

new_dno2 = Ax.*N1.^2+Bx.*N1.*N2+Dx.*N2.^2;
new_dno2 = 1/(2*pi)*new_dno2;
der_nor2 = new_dno2;

curv_mat = repmat(H',[1 M]);

As = (1+bet*curv_mat).*An+bet*der_nor2.';

opts.isreal  = 1;
opts.tol     = 1e-12;

[V,D] = eigs(As,B,k+1,'sm',opts);
[D,Is]                             = sort(diag(D));
V                                  = V(:,Is);

lam   = D(1:end);
lam   = real(lam);

