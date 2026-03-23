function [p,T,Ibord,A,dx] = sph_triangle_mesh_Pts(n,Pts,variant)

% sph_triangle_mesh_Pts(n,Pts,variant)
% inputs
% n: number of midpoint refinements 
% Pts: vertices of the triangle (on 3 lines)
% variant: two variants of meshing. The default one is used in the article

% two variants for meshing

plotting = 0; % if you want to see the points in the end
              % makes the code slower

if nargin<3
 variant = 1;
end
switch variant
case 1
  p = Pts;
  n1 = (p(1,:)+p(2,:))/2; n1 = n1/norm(n1);
  n2 = (p(2,:)+p(3,:))/2; n2 = n2/norm(n2);
  n3 = (p(1,:)+p(3,:))/2; n3 = n3/norm(n3);
  n4 = (p(1,:)+p(2,:)+p(3,:))/3; n4 = n4/norm(n4); 
  p  = [p; n1; n2; n3; n4];
  T  = [1 4 7;
        1 7 6;
        4 2 7;
        2 5 7;
        3 7 5;
        7 3 6];
case 2
  p = Pts;
  n1 = (p(1,:)+p(2,:))/2; n1 = n1/norm(n1);
  n2 = (p(2,:)+p(3,:))/2; n2 = n2/norm(n2);
  n3 = (p(1,:)+p(3,:))/2; n3 = n3/norm(n3);
  p  = [p; n1; n2; n3];
  T  = [1 4 6;
        2 4 5;
        4 5 6;
        3 5 6];
end

ap = size(p,1);
for i = 1:n 
    ct = size(T,1); % nb of triagles
    ap = size(p,1); % nb of points INITIALLY
    edges = [T(:,1) T(:,2); T(:,2) T(:,3); T(:,1) T(:,3)];
    edges = unique(sort(edges,2),'rows'); % pairs of connected points
    nbe   = size(edges,1);  
    mids  = 0.5*(p(edges(:,1),:)+p(edges(:,2),:));
    norms = sqrt(sum(mids.^2,2));
    mids  = mids./repmat(norms,[1 3]);
    p     = [p;mids];    % points with midpoints
    
    % T1,M,T2...  find position in edge matrix
    [~,M3] = ismember(sort([T(:,1) T(:,2)],2),edges,'rows');
    [~,M1] = ismember(sort([T(:,2) T(:,3)],2),edges,'rows');
    [~,M2] = ismember(sort([T(:,1) T(:,3)],2),edges,'rows'); 
    M1 = M1+ap;
    M2 = M2+ap;
    M3 = M3+ap;
    
    T      = [T(:,1) M3 M2; M1 M2 M3; T(:,2) M1 M3; T(:,3) M2 M1];

    ap = size(p,1);
end                 ;

A = min(sparse(T(:,1),T(:,2),1,ap,ap)+sparse(T(:,2),T(:,3),1,ap,ap)+sparse(T(:,3),T(:,1),1,ap,ap),1);
A = min(A+A',1);

B = A^2.*A==1;
Ibord = find(sum(B,2)>0);


if(plotting==1)
  plot3(p(:,1),p(:,2),p(:,3),'.');
  axis equal
  pause
end
