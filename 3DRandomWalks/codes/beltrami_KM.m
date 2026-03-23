function [L,M,a] = beltrami_KM(p,t,Ibord)

% [L,M] = beltrami_KM(p,t)
%
% Stiffness and mass matrix for Laplace-Beltrami on surfaces
% 
% inputs:
% p: points array
% t: triangles
% Ibord: index of points where we have Dirichlet condition
%
% outputs
% L is the laplacian matrix
% M is the mass matrix
%

nt                                     = size(t,1);
np                                     = size(p,1);
dim                                    = size(p,2);

W                                      = sparse(np,np);
for i = 1:3
  i1                                   = mod(i-1,3)+1;
  i2                                   = mod(i  ,3)+1;
  i3                                   = mod(i+1,3)+1;
  pp                                   = p(t(:,i2),:) - p(t(:,i1),:);
  qq                                   = p(t(:,i3),:) - p(t(:,i1),:);
  % normalize the vectors
  vt                                   = sqrt(sum(pp.^2,2));
  pp                                   = bsxfun(@rdivide,pp,vt);
  vt                                   = sqrt(sum(qq.^2,2));
  qq                                   = bsxfun(@rdivide,qq,vt);
  
  % compute angles
  xx                                   = sum(pp.*qq,2);
  a                                    = xx./sqrt(1-xx.^2);
  a                                    = 1 ./ tan( acos(sum(pp.*qq,2)) );
  a(abs(a)<1e-8)                       = 1e-8; % avoid degeneracy
  W                                    = W + sparse(t(:,i2),t(:,i3), a, np, np );
  W                                    = W + sparse(t(:,i3),t(:,i2), a, np, np );
end

d                                      = sum(W,1);
D                                      = spdiags(d(:), 0, size(W,1),size(W,1));
L                                      = (D - W)/2;


%% compute the area of faces
e1                                     = p(t(:,2),:) - p(t(:,1),:);
e2                                     = p(t(:,3),:) - p(t(:,1),:);
a                                      = cross(e1,e2);
a                                      = sqrt( sum(a.^2,2) ) / 2;
M                                      = sparse(np,np);
for i = 1:3
  j                                    = mod(i+1,3)+1;
  M                                    = M + sparse(t(:,i),t(:,j), a, np, np )/12;
  M                                    = M + sparse(t(:,j),t(:,i), a, np, np )/12;
end
d                                      = sum(M,1);
M                                      = M + spdiags(d(:), 0, np,np);

% if Dirichlet condition is present
if nargin>2
  L   = L + sparse(Ibord,Ibord,1e15*ones(1,length(Ibord)),size(L,1),size(L,2));
end
