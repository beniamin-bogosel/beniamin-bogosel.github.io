function vals = sph_tri_eigs(k,mh,Pts,pl)

% sph_tri_eigs(k,mh,Pts,pl)
% inputs 
% k: number of eigenvalues
% mh: mesh parameter (LESS than 9 can work under a minute on a laptop)
%                    (10 and 11 take up to 10 minutes but use a lot of RAM 
%                         between 20 and 80GB !! Beware of crashes...)
% Pts: points defining the triangle: 3 lines with 3 coordonates
% pl:  if present, a plot of the eigenfunction is presented

if nargin<3
 va = 1;
end

if nargin<2
 mh =6;
end

% create mesh
[p,t,Ibord,dx] = sph_triangle_mesh_Pts(mh,Pts);

points = p';

% compute mass and stiffness matrices
[K,M] = beltrami_KM(p,t,Ibord);

opts.issym = 1;
opts.isreal = 1;
opts.tol = 1e-16;
%opts.maxit = 1e3;


% compute eigenvalues
[V,D] = eigs(K,M,k,'sm',opts);

[D,Is]                             = sort(diag(D));
V                                  = V(:,Is);

lam   = D(1:end);
lam   = real(lam);
coeffs = V(:,1:end);
coeffs = real(coeffs);
if min(coeffs)<-1e-4
  coeffs = -coeffs;
end

vals = D(:);
%fprintf('Eigenvalue: %.12f\n',vals);

% eventual plotting, makes code slower if you run on multiple triangles
if nargin>3
clf
hold on
[X,Y,Z] = sphere(100);factt=1.03;
patch('Faces',t,'Vertices',p,'FaceVertexCData',coeffs(:,k),'FaceColor','interp','EdgeColor','none');
axis equal
 h       = surf(X/factt,Y/factt,Z/factt);set(h,'FaceColor',0.9*ones(1,3),'EdgeColor','none');

hold off
view([1 pi/4 0.7]);
end
