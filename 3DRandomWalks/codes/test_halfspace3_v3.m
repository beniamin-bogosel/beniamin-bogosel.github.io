function res = test_halfspace3_v3(mat)

% test_halfspace3_v3(nn)
%
% Tests if the columns of the 3xN matrix given as input
% represent vectors in the same halfspace going through the origin
%
% The algorithm simply takes each pair of columns and tests if all the remaining vectors are on the same side of the corresponding plane
% The algorithm is at worst O(N^2) so it is not appropriate if you have too many vectors
% Also, in our case we do not have points which are almost collinear, we work with a tolerance of 1e-6 to say that something is positive or negative
 

nv = size(mat,2);
vv = nchoosek(1:nv,2);

res = 0;

for i = 1:size(vv,1)
  dir = cross(mat(:,vv(i,1)),mat(:,vv(i,2)));
  if norm(dir)>1e-3
    dots = zeros(1,nv);    
    for i = 1:nv
      dots(i) = dot(mat(:,i),dir);
    end 
    np = sum(dots>-1e-6);
    nm = sum(dots<1e-6);
    if or(np == nv,nm == nv)
      res = 1;  
      break
    end
  end  

end



