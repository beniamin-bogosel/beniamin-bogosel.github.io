function res = accurate_bel_Silent(n,Pts,pl)

% accurate_bel_Silent(n,Pts,pl)
%
% Computes eigenvalue of spherical triangles with extrapolation procedure
% 
% inputs
% n: number of refinements (extrapolation inputs)
%      n<=9 works on a laptop. Don't try higher! You'll run out of RAM
%      n=10,11 uses up to 20 and 80 GB of RAM, respectively
% Pts: vertices of the triangle: points on 3 lines
% pl: if present, it plots the eigenfunction for the finest mesh


%vals = zeros(1,n);

for i = 1:n
  if i<n
    vals(i) = sph_tri_eigs_Run(1,i,Pts);
  else
    if nargin<3
      vals(i) = sph_tri_eigs_Run(1,i,Pts);
    else
      vals(i) = sph_tri_eigs_Run(1,i,Pts,1);
    end
  end
end

% extrapolation procedure
L = length(vals);

L2 = 2*L-1; vv = zeros(L2,1); vv(1:2:L2) = vals;
for j=2:L
 k=j:2:L2+1-j; vv(k)=vv(k)+1./(vv(k+1)-vv(k-1));
end;
result = vv(1:2:L2);

%for i=1:length(vals)
%  fprintf('%2d | %.16f | %.16f\n',i,vals(i),result(i));
%end

res = result(floor((L+1)/2));
