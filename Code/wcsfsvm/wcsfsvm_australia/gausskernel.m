function output = gausskernel( vec1, vec2, p)
%GAUSSKERNEL Summary of this function goes here
%   Detailed explanation goes here
    if(nargin <3)
        p = 2^-4;
    end
    output = exp(-p * sum((vec1 - vec2).^2));
    
end

